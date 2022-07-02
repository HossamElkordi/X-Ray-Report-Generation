import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
import nlgeval


# ------ Helper Functions ------
def data_to_device(data, device='cpu'):
    if isinstance(data, torch.Tensor):
        data = data.to(device)
    elif isinstance(data, tuple):
        data = tuple(data_to_device(item, device) for item in data)
    elif isinstance(data, list):
        data = list(data_to_device(item, device) for item in data)
    elif isinstance(data, dict):
        data = dict((k, data_to_device(v, device)) for k, v in data.items())
    else:
        raise TypeError('Unsupported Datatype! Must be a Tensor/List/Tuple/Dict.')
    return data


def data_concatenate(iterable_data, dim=0):
    data = iterable_data[0]  # can be a list / tuple / dict / tensor
    if isinstance(data, torch.Tensor):
        return torch.cat([*iterable_data], dim=dim)
    elif isinstance(data, tuple):
        num_cols = len(data)
        num_rows = len(iterable_data)
        return_data = []
        for col in range(num_cols):
            data_col = []
            for row in range(num_rows):
                data_col.append(iterable_data[row][col])
            return_data.append(torch.cat([*data_col], dim=dim))
        return tuple(return_data)
    elif isinstance(data, list):
        num_cols = len(data)
        num_rows = len(iterable_data)
        return_data = []
        for col in range(num_cols):
            data_col = []
            for row in range(num_rows):
                data_col.append(iterable_data[row][col])
            return_data.append(torch.cat([*data_col], dim=dim))
        return list(return_data)
    elif isinstance(data, dict):
        num_cols = len(data)
        num_rows = len(iterable_data)
        return_data = []
        for col in data.keys():
            data_col = []
            for row in range(num_rows):
                data_col.append(iterable_data[row][col])
            return_data.append(torch.cat([*data_col], dim=dim))
        return dict((k, return_data[i]) for i, k in enumerate(data.keys()))
    else:
        raise TypeError('Unsupported Datatype! Must be a Tensor/List/Tuple/Dict.')


def data_distributor(model, source):
    if isinstance(source, torch.Tensor):
        output = model(source)
    elif isinstance(source, tuple) or isinstance(source, list):
        output = model(*source)
    elif isinstance(source, dict):
        output = model(**source)
    else:
        raise TypeError('Unsupported DataType! Try List/Tuple!')
    return output


def args_to_kwargs(args,
                   kwargs_list=None):  # This function helps distribute input to corresponding arguments in Torch models
    if kwargs_list != None:
        if isinstance(args, dict):  # Nothing to do here
            return args
        else:  # args is a list or tuple or single element
            if isinstance(args, torch.Tensor):  # single element
                args = [args]
            assert len(args) == len(kwargs_list)
            return dict(zip(kwargs_list, args))
    else:  # Nothing to do here
        return args


def rt_decode_report(captions, dataset):
    gen = {}
    for i in range(len(captions)):
        decoded = ''
        for j in range(len(captions[i])):
            tok = dataset.vocab.id_to_piece(int(captions[i, j]))
            if tok == '</s>':
                break  # Manually stop generating token after </s> is reached
            elif tok == '<s>':
                continue
            elif tok == '▁':  # space
                if len(decoded) and decoded[-1] != ' ':
                    decoded += ' '
            elif tok in [',', '.', '-', ':']:  # or not tok.isalpha():
                if len(decoded) and decoded[-1] != ' ':
                    decoded += ' ' + tok + ' '
                else:
                    decoded += tok + ' '
            else:  # letter
                decoded += tok
        gen['{}'.format(i)] = [decoded]
    return gen


# ------ Core Functions ------
def train(data_loader, model, optimizer, criterion, scheduler=None, device='cpu', kw_src=None, kw_tgt=None, kw_out=None,
          scaler=None, use_rl=False, dataset=None):
    model.train()
    running_loss = 0
    prog_bar = tqdm(data_loader)
    scorer = nlgeval.Cider()
    for i, (source, target) in enumerate(prog_bar):
        source = data_to_device(source, device)
        target = data_to_device(target, device)

        source = args_to_kwargs(source, kw_src)
        target = args_to_kwargs(target, kw_tgt)
        source['use_rl'] = use_rl

        if scaler != None:
            with torch.cuda.amp.autocast():
                output = data_distributor(model, source)
                if not use_rl:
                    output = args_to_kwargs(output, kw_out)
                    loss = criterion(output, target)
                else:
                    gen, score = output[0], output[1]
                    gen = rt_decode_report(gen, dataset)
                    gts = rt_decode_report(target[0], dataset)

                    reward = scorer.compute_score(gts, gen)[1].astype(np.float32)
                    # decode .....

            running_loss += loss.item()
            prog_bar.set_description('Loss: {}'.format(running_loss / (i + 1)))

            # Back-propagate and update weights
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
        else:
            output = data_distributor(model, source)
            output = args_to_kwargs(output, kw_out)
            loss = criterion(output, target)

            running_loss += loss.item()
            prog_bar.set_description('Loss: {}'.format(running_loss / (i + 1)))

            # Back-propagate and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

    return running_loss / len(data_loader)


def test(data_loader, model, criterion=None, device='cpu', return_results=True, kw_src=None, kw_tgt=None, kw_out=None,
         select_outputs=[]):
    model.eval()
    running_loss = 0

    outputs = []
    targets = []

    with torch.no_grad():
        prog_bar = tqdm(data_loader)
        for i, (source, target) in enumerate(prog_bar):
            source = data_to_device(source, device)
            target = data_to_device(target, device)

            source = args_to_kwargs(source, kw_src)
            target = args_to_kwargs(target, kw_tgt)

            output = data_distributor(model, source)
            output = args_to_kwargs(output, kw_out)

            if criterion != None:
                loss = criterion(output, target)
                running_loss += loss.item()
            prog_bar.set_description('Loss: {}'.format(running_loss / (i + 1)))

            if return_results:
                if len(select_outputs) == 0:
                    outputs.append(data_to_device(output, 'cpu'))
                    targets.append(data_to_device(target, 'cpu'))
                else:
                    list_output = [output[row] for row in select_outputs]
                    list_target = [target[row] for row in select_outputs]
                    outputs.append(data_to_device(list_output if len(list_output) > 1 else list_output[0], 'cpu'))
                    targets.append(data_to_device(list_target if len(list_target) > 1 else list_target[0], 'cpu'))

    if return_results:
        outputs = data_concatenate(outputs)
        targets = data_concatenate(targets)
        return running_loss / len(data_loader), outputs, targets
    else:
        return running_loss / len(data_loader)


def save(path, model, optimizer=None, scheduler=None, epoch=-1, stats=None):
    torch.save({
        # --- Model Statistics ---
        'epoch': epoch,
        'stats': stats,
        # --- Model Parameters ---
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer != None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler != None else None,
    }, path)


def load(path, model, optimizer=None, scheduler=None):
    if not os.path.exists(path):
        return -1, (1e9, 1e9)
    checkpoint = torch.load(path)
    # --- Model Statistics ---
    epoch = checkpoint['epoch']
    stats = checkpoint['stats']
    # --- Model Parameters ---
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer != None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:  # Input optimizer doesn't fit the checkpoint one --> should be ignored
            print('Cannot load the optimizer')
    if scheduler != None:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except:  # Input scheduler doesn't fit the checkpoint one --> should be ignored
            print('Cannot load the scheduler')
    return epoch, stats
