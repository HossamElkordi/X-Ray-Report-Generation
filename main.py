# --- Base packages ---
import argparse
import os

# --- PyTorch packages ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

# --- Helper Packages ---
from tqdm import tqdm
import numpy as np

# --- Project Packages ---
from utils import save, load, train, test, data_to_device, data_concatenate
from datasets import MIMIC, NLMCXR
from losses import CELossTotalEval, CELossTotal
from models import CNN, MVCNN, TNN, Classifier, ClsGen, ClsGenInt
from base_cmn import BaseCMN

# --- Metrics Evaluation ---
import nlgeval

# --- Text Trainer ---
from train_text import train_interpreter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.manual_seed(seed=123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(123)


def train_model(model, train_loader, val_loader, test_loader, optimizer, criterion, scheduler,
                best_loss, last_epoch, num_epochs, save_path, sched_type):
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(last_epoch + 1, num_epochs):
        print('Epoch:', epoch)
        train_loss = train(train_loader, model, optimizer, criterion, device='cuda',
                           kw_src=args.kwargs_sources, scaler=scaler)
        val_loss = test(val_loader, model, criterion, device='cuda', kw_src=args.kwargs_sources, return_results=False)
        test_loss = test(test_loader, model, criterion, device='cuda', kw_src=args.kwargs_sources, return_results=False)

        if sched_type == 'ROP':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if best_loss > val_loss:
            best_loss = val_loss
            save(save_path, model, optimizer, scheduler, epoch, (val_loss, test_loss))
            print('New Best Metric: {}'.format(best_loss))
            print('Saved To:', save_path)


def test_model(test_data, comment):
    # Output the file list for inspection
    out_file_img = open('/content/drive/MyDrive/outputs/{}_{}_{}_{}_Img.txt'.format(args.dataset_name, args.model_name,
                                                                                    args.visual_extractor, comment),
                        'w')
    for i in range(len(test_data.idx_pidsid)):
        out_file_img.write(test_data.idx_pidsid[i][0] + ' ' + test_data.idx_pidsid[i][1] + '\n')


def evaluate_metric(gts, gen):
    # Bleu
    scorer = nlgeval.Bleu()
    score = scorer.compute_score(gts, gen)[0]
    print('Bleu_1:', score[0])
    print('Bleu_2:', score[1])
    print('Bleu_3:', score[2])
    print('Bleu_4:', score[3])
    # Meteor
    scorer = nlgeval.Meteor()
    score = scorer.compute_score(gts, gen)
    print('Meteor:', score[0])
    # Rouge-L
    scorer = nlgeval.Rouge()
    score = scorer.compute_score(gts, gen)
    print('Rouge-L:', score[0])
    # CIDEr
    scorer = nlgeval.Cider()
    score = scorer.compute_score(gts, gen)
    print('CIDEr:', score[0])


def decode_report(captions, i, dataset):
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
    return decoded


def infer_model(model, dataset, test_data, test_loader, comment):
    txt_test_outputs, txt_test_targets = infer(test_loader, model, device='cuda', threshold=0.25)
    gen_outputs = txt_test_outputs[0]
    gen_targets = txt_test_targets[0]

    out_file_ref = open(
        '/content/drive/MyDrive/outputs/x_{}_{}_{}_{}_Ref.txt'.format(args.dataset_name, args.model_name,
                                                                      args.visual_extractor, comment), 'w')
    out_file_hyp = open(
        '/content/drive/MyDrive/outputs/x_{}_{}_{}_{}_{}_Hyp.txt'.format(args.dataset_name, args.model_name,
                                                                      args.visual_extractor, comment, args.trial), 'w')
    out_file_lbl = open(
        '/content/drive/MyDrive/outputs/x_{}_{}_{}_{}_{}_Lbl.txt'.format(args.dataset_name, args.model_name,
                                                                      args.visual_extractor, comment, args.trial), 'w')

    gts, gen = {}, {}

    for i in range(len(gen_outputs)):
        candidate = decode_report(gen_outputs, i, dataset)
        out_file_hyp.write(candidate + '\n')
        gen['{}'.format(i)] = [candidate]

        reference = decode_report(gen_targets, i, dataset)
        out_file_ref.write(reference + '\n')
        gts['{}'.format(i)] = [reference]

    for i in tqdm(range(len(test_data))):
        target = test_data[i][1]  # caption, label
        out_file_lbl.write(' '.join(map(str, target[1])) + '\n')
    evaluate_metric(gts, gen)


def infer(data_loader, model, device='cpu', threshold=None):
    model.eval()
    outputs = []
    targets = []

    with torch.no_grad():
        prog_bar = tqdm(data_loader)
        for i, (source, target) in enumerate(prog_bar):
            source = data_to_device(source, device)
            target = data_to_device(target, device)

            # Use single input if there is no clinical history
            if threshold is not None:
                output, _ = model(image=source[0], history=source[3], threshold=threshold)
                # output = model(image=source[0], threshold=threshold)
                # output = model(image=source[0], history=source[3], label=source[2])
                # output = model(image=source[0], label=source[2])
            else:
                # output = model(source[0], source[1])
                output, _ = model(source[0])

            outputs.append(data_to_device(output))
            targets.append(data_to_device(target))

        outputs = data_concatenate(outputs)
        targets = data_concatenate(targets)

    return outputs, targets


def main(args):
    print('Loading', args.dataset_name, 'dataset...')
    vocab_size, posit_size, comment, dataset = None, None, None, None
    train_data, val_data, test_data = None, None, None
    if args.dataset_name == 'MIMIC':
        dataset = MIMIC(args.dataset_dir, (args.input_size, args.input_size), view_pos=['AP', 'PA', 'LATERAL'],
                        max_views=args.max_views, sources=args.sources, targets=args.targets)
        train_data, val_data, test_data = dataset.get_subsets(pvt=0.9, seed=0, generate_splits=True, debug_mode=False,
                                                              train_phase=(args.phase == 'train'))
        vocab_size = len(dataset.vocab)
        posit_size = dataset.max_len
        comment = 'MaxView{}_NumLabel{}_{}History'.format(args.max_views, args.decease_related_topics,
                                                          'No' if 'history' not in args.sources else '')
    elif args.dataset_name == 'NLMCXR':
        dataset = NLMCXR(args.dataset_dir, (args.input_size, args.input_size), view_pos=['AP', 'PA', 'LATERAL'],
                         max_views=args.max_views, sources=args.sources, targets=args.targets)
        train_data, val_data, test_data = dataset.get_subsets(seed=123)

        vocab_size = len(dataset.vocab)
        posit_size = dataset.max_len
        comment = 'MaxView{}_NumLabel{}_{}History'.format(args.max_views, args.decease_related_topics,
                                                          'No' if 'history' not in args.sources else '')
    print('Done Loading Dataset')

    backbone, fc_features = None, None
    if args.visual_extractor == 'DenseNet121':
        backbone = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        fc_features = 1024

    elif args.visual_extractor == 'DenseNet121':
        backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
        fc_features = 2048

    print('Creating Model', args.model_name, '...')
    model, criterion = None, None
    if args.model_name == 'ClsGen':
        visual_extractor = CNN(backbone, args.visual_extractor)
        visual_extractor = MVCNN(visual_extractor)
        text_feat_extractor = TNN(embed_dim=args.num_embed, num_heads=args.num_heads, fwd_dim=args.fd_dim,
                                  dropout=args.dropout,
                                  num_layers=1, num_tokens=vocab_size, num_posits=posit_size)
        cls_model = Classifier(num_topics=args.decease_related_topics, num_states=args.num_classes,
                               cnn=visual_extractor,
                               tnn=text_feat_extractor, fc_features=fc_features, embed_dim=args.num_embed,
                               num_heads=args.num_heads,
                               dropout=args.dropout)
        gen_model = BaseCMN(args, dataset.vocab)
        model = ClsGen(cls_model, gen_model, args.decease_related_topics, args.num_embed)
        criterion = CELossTotal(ignore_index=3)
    elif args.model_name == 'ClsGenInt':
        visual_extractor = CNN(backbone, args.visual_extractor)
        visual_extractor = MVCNN(visual_extractor)
        text_feat_extractor = TNN(embed_dim=args.num_embed, num_heads=args.num_heads, fwd_dim=args.fd_dim,
                                  dropout=args.dropout,
                                  num_layers=1, num_tokens=vocab_size, num_posits=posit_size)
        cls_model = Classifier(num_topics=args.decease_related_topics, num_states=args.num_classes,
                               cnn=visual_extractor, tnn=text_feat_extractor, fc_features=fc_features,
                               embed_dim=args.num_embed, num_heads=args.num_heads, dropout=args.dropout)
        gen_model = BaseCMN(args, dataset.vocab)

        cls_gen_model = ClsGen(cls_model, gen_model, args.decease_related_topics, args.num_embed)
        cls_gen_model = nn.DataParallel(cls_gen_model).cuda()

        if not bool(args.reload):
            checkpoint_path_from = \
                '/content/drive/MyDrive/checkpoints/{}_ClsGen_{}_{}_{}.pt'.format(args.dataset_name,
                                                                                  args.visual_extractor,
                                                                                  comment, args.trial)
            last_epoch, (best_metric, test_metric) = load(checkpoint_path_from, cls_gen_model)
            print('Reload From: {} | Last Epoch: {} | Validation Metric: {} | Test Metric: {}'.format(
                checkpoint_path_from,
                last_epoch,
                best_metric,
                test_metric))

        int_text_feat_extractor = TNN(embed_dim=args.num_embed, num_heads=args.num_heads, fwd_dim=args.fd_dim,
                                      dropout=args.dropout, num_layers=1,
                                      num_tokens=vocab_size, num_posits=posit_size)
        int_model = Classifier(num_topics=args.decease_related_topics, num_states=args.num_classes, cnn=None,
                               tnn=int_text_feat_extractor, embed_dim=args.num_embed,
                               num_heads=args.num_heads, dropout=args.dropout)
        int_model = nn.DataParallel(int_model).cuda()

        if not bool(args.reload):
            checkpoint_path_from = '/content/drive/MyDrive/checkpoints/{}_Int_MaxView2_NumLabel{}_{}.pt'.format(
                args.dataset_name,
                args.decease_related_topics, args.num_embed)
            last_epoch, (best_metric, test_metric) = load(checkpoint_path_from, int_model)
            print('Reload From: {} | Last Epoch: {} | Validation Metric: {} | Test Metric: {}'.format(
                checkpoint_path_from,
                last_epoch,
                best_metric,
                test_metric))

        model = ClsGenInt(cls_gen_model.module.cpu(), int_model.module.cpu(), freeze_evaluator=True)
        criterion = CELossTotalEval(ignore_index=3)

    print('Done Creating Model')

    train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   drop_last=True)
    val_loader = data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = nn.DataParallel(model).cuda()

    gen_params = list(map(id, model.module.clsgen.generator.parameters()))
    clsint_params = filter(lambda x: id(x) not in gen_params, model.parameters())
    optim_arguments = []
    clsint_args = {'params': clsint_params, 'lr': args.lr_clsint, 'weight_decay': args.weight_decay_clsint}
    gen_args = {'params': model.module.clsgen.generator.parameters(), 'lr': args.lr_gen,
                'weight_decay': args.weight_decay_gen}
    optim_arguments.append(clsint_args)
    optim_arguments.append(gen_args)
    optimizer = optim.AdamW(optim_arguments)

    scheduler = None
    if args.scheduler == 'ROP':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.patience, verbose=True)
    elif args.scheduler == 'EXP':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    elif args.scheduler == 'COS':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    print('Total Parameters:', sum(p.numel() for p in model.parameters()))
    last_epoch = -1
    best_metric = 1e9

    checkpoint_path_from = '/content/drive/MyDrive/checkpoints/{}_{}_{}_{}_{}.pt'.format(args.dataset_name,
                                                                                         args.model_name,
                                                                                         args.visual_extractor, comment,
                                                                                         args.trial)
    checkpoint_path_to = '/content/drive/MyDrive/checkpoints/{}_{}_{}_{}_{}.pt'.format(args.dataset_name,
                                                                                       args.model_name,
                                                                                       args.visual_extractor, comment,
                                                                                       args.trial)

    if bool(args.reload):
        last_epoch, (best_metric, test_metric) = load(checkpoint_path_from, model, optimizer, scheduler)  # Reload
        # last_epoch, (best_metric, test_metric) = load(checkpoint_path_from, model) # Fine-tune
        print('Reload From: {} | Last Epoch: {} | Validation Metric: {} | Test Metric: {}'.format(checkpoint_path_from,
                                                                                                  last_epoch,
                                                                                                  best_metric,
                                                                                                  test_metric))
    if args.phase == 'train':
        train_model(model, train_loader, val_loader, test_loader, optimizer, criterion, scheduler,
                    best_metric, last_epoch, args.epochs, checkpoint_path_to, args.scheduler)
    elif args.phase == 'test':
        test_model(test_data, comment)
    elif args.phase == 'infer':
        infer_model(model, dataset, test_data, test_loader, comment)


def parse_agruments():
    parser = argparse.ArgumentParser()

    # Operation Phase
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'infer', 'test'])
    parser.add_argument('--reload', type=int, default=1, help='whether to load a saved model or not')
    parser.add_argument('--trial', type=int, default=1, help='tuning trial number')

    # Data input settings
    parser.add_argument('--dataset_dir', type=str, default='/content/x_ray_report_generation/open-i',
                        help='the path to the dataset.')
    parser.add_argument('--input_size', type=int, default=256, help='Input Image Size.')
    parser.add_argument('--max_views', type=int, default=2, help='Max Number of X-Ray Views per dataset sample.')
    parser.add_argument('--num_classes', type=int, default=2, help='Positive and Negative Classification.')
    parser.add_argument('--decease_related_topics', type=int, default=114, help='Decease Related Topics.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='NLMCXR', choices=['NLMCXR', 'MIMICCXR'],
                        help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='DenseNet121', choices=['DenseNet121', 'resnet101'],
                        help='the visual extractor to be used.')

    parser.add_argument('--model_name', type=str, default='ClsGenInt', choices=['ClsGenInt', 'ClsGen', 'Int'],
                        help='Model Type.')
    parser.add_argument('--sources', type=list, default=['image', 'caption', 'label', 'history'], help='Source Texts.')
    parser.add_argument('--targets', type=list, default=['caption', 'label'], help='Target Texts.')
    parser.add_argument('--kwargs_sources', type=list, default=['image', 'caption', 'label', 'history'],
                        help='Kwargs Source Texts.')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')

    # for Cross-modal Memory
    parser.add_argument('--topk', type=int, default=32, help='the number of k.')
    parser.add_argument('--cmm_size', type=int, default=2048, help='the numebr of cmm size.')
    parser.add_argument('--cmm_dim', type=int, default=512, help='the dimension of cmm dimension.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search',
                        help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    parser.add_argument('--num_embed', type=int, default=2048, help='visual and textual embedding size')
    parser.add_argument('--fd_dim', type=int, default=256, help='forward dimension')

    # Trainer settings
    parser.add_argument('--epochs', type=int, default=50, help='the number of training epochs.')
    parser.add_argument('--lr_gen', type=float, default=5e-4, help='the learning rate for NLM-CXR.')
    parser.add_argument('--lr_clsint', type=float, default=3e-5, help='the learning rate for NLM-CXR.')
    parser.add_argument('--weight_decay_gen', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--weight_decay_clsint', type=float, default=1e-2, help='the weight decay.')
    parser.add_argument('--scheduler', type=str, default='ROP', choices=['ROP', 'COS', 'EXP'], help='scheduler type.')
    parser.add_argument('--patience', type=int, default=3,
                        help='Reduce LR by 10 after reaching patience epochs if ROP is used')
    parser.add_argument('--gamma', type=float, default=0.5, help='Reduce LR by gamma each epoch if EXP is used')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_agruments()
    if args.model_name == 'Int':
        train_interpreter(args)
    else:
        main(args)
