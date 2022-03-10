# --- Base packages ---
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

# --- PyTorch packages ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# --- Project Packages ---
from utils import save, load, train, test
from datasets import MIMIC, NLMCXR
from losses import CELoss
from models import Classifier, TNN
from baselines.transformer.models import LSTM_Attn

# --- Hyperparameters ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.manual_seed(seed=0)


def train_interpreter(args):
    print('Loading', args.dataset_name, 'dataset...')

    vocab_size, posit_size, comment, dataset = None, None, None, None
    train_data, val_data, test_data = None, None, None
    # --- Choose a Dataset ---
    if args.dataset_name == 'MIMIC':
        dataset = MIMIC(args.dataset_dir, (args.input_size, args.input_size), view_pos=['AP', 'PA', 'LATERAL'],
                        max_views=args.max_views, sources=['caption'], targets=['label'])
        train_data, val_data, test_data = dataset.get_subsets(pvt=0.9, seed=0, generate_splits=True, debug_mode=False,
                                                              train_phase=(args.phase == 'train'))

        vocab_size = len(dataset.vocab)
        posit_size = dataset.max_len
        comment = 'MaxView{}_NumLabel{}'.format(args.max_views, args.decease_related_topics)
    elif args.dataset_name == 'NLMCXR':
        dataset = NLMCXR(args.dataset_dir, (args.input_size, args.input_size), view_pos=['AP', 'PA', 'LATERAL'],
                         max_views=args.max_views, sources=['caption'], targets=['label'])
        train_data, val_data, test_data = dataset.get_subsets(seed=123)

        vocab_size = len(dataset.vocab)
        posit_size = dataset.max_len
        comment = 'MaxView{}_NumLabel{}'.format(args.max_views, args.decease_related_topics)
    else:
        raise ValueError('Invalid dataset name')

    print('Done Loading Dataset')
    print('Creating Model', args.model_name, '...')

    tnn = TNN(embed_dim=args.num_embed, num_heads=args.num_heads, fwd_dim=args.fd_dim, dropout=args.dropout,
              num_layers=1, num_tokens=vocab_size, num_posits=posit_size)
    model = Classifier(num_topics=args.decease_related_topics, num_states=args.num_classes, cnn=None, tnn=tnn,
                       embed_dim=args.num_embed, num_heads=args.num_heads, dropout=args.dropout)
    criterion = CELoss()

    # --- Main program ---
    train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   drop_last=True)
    val_loader = data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = nn.DataParallel(model).cuda()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr_nlm if args.dataset_name == 'NLMCXR' else args.lr_mimic,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epoch_milestone])

    print('Done Creating Model')
    print('Total Parameters:', sum(p.numel() for p in model.parameters()))

    last_epoch = -1
    best_metric = 0

    checkpoint_path_from = '/content/drive/MyDrive/checkpoints/{}_{}_{}_{}.pt'.format(args.dataset_name,
                                                                                      args.model_name,
                                                                                      comment, args.trial)
    checkpoint_path_to = '/content/drive/MyDrive/checkpoints/{}_{}_{}_{}.pt'.format(args.dataset_name, args.model_name,
                                                                                    comment, args.trial)

    if bool(args.reload):
        last_epoch, (best_metric, test_metric) = load(checkpoint_path_from, model, optimizer, scheduler)
        print('Reload From: {} | Last Epoch: {} | Validation Metric: {} | Test Metric: {}'.format(checkpoint_path_from,
                                                                                                  last_epoch,
                                                                                                  best_metric,
                                                                                                  test_metric))

    if args.phase == 'train':
        scaler = torch.cuda.amp.GradScaler()  # Reduce floating to 16 bits instead of 32 bits

        for epoch in range(last_epoch + 1, args.epochs):
            print('Epoch:', epoch)
            train_loss = train(train_loader, model, optimizer, criterion, device='cuda', kw_src=['txt'],
                               scaler=scaler)
            val_loss, val_outputs, val_targets = test(val_loader, model, criterion, device='cuda',
                                                      kw_src=['txt'])
            test_loss, test_outputs, test_targets = test(test_loader, model, criterion, device='cuda',
                                                         kw_src=['txt'])
            scheduler.step()

            val_metric = []
            test_metric = []
            for i in range(args.decease_related_topics):
                try:
                    val_metric.append(metrics.roc_auc_score(val_targets.cpu()[..., i], val_outputs.cpu()[..., i, 1]))
                except:
                    pass
                try:
                    test_metric.append(metrics.roc_auc_score(test_targets.cpu()[..., i], test_outputs.cpu()[..., i, 1]))
                except:
                    pass
            val_metric = np.mean([x for x in val_metric if str(x) != 'nan'])
            test_metric = np.mean([x for x in test_metric if str(x) != 'nan'])

            print('Validation Metric: {} | Test Metric: {}'.format(val_metric, test_metric))

            if best_metric < val_metric:
                best_metric = val_metric
                save(checkpoint_path_to, model, optimizer, scheduler, epoch, (val_metric, test_metric))
                print('New Best Metric: {}'.format(best_metric))
                print('Saved To:', checkpoint_path_to)

    elif args.phase == 'test':
        test_loss, test_outputs, test_targets = test(test_loader, model, criterion, device='cuda',
                                                     kw_src=['txt'])

        test_auc = []
        test_f1 = []
        test_prc = []
        test_rec = []
        test_acc = []

        threshold = 0.5
        num_labels = 14
        for i in range(num_labels):
            try:
                test_auc.append(metrics.roc_auc_score(test_targets.cpu()[..., i], test_outputs.cpu()[..., i, 1]))
                test_f1.append(metrics.f1_score(test_targets.cpu()[..., i], test_outputs.cpu()[..., i, 1] > threshold))
                test_prc.append(
                    metrics.precision_score(test_targets.cpu()[..., i], test_outputs.cpu()[..., i, 1] > threshold))
                test_rec.append(
                    metrics.recall_score(test_targets.cpu()[..., i], test_outputs.cpu()[..., i, 1] > threshold))
                test_acc.append(
                    metrics.accuracy_score(test_targets.cpu()[..., i], test_outputs.cpu()[..., i, 1] > threshold))

            except:
                print('An error occurs for label', i)

        test_auc = np.mean([x for x in test_auc if str(x) != 'nan'])
        test_f1 = np.mean([x for x in test_f1 if str(x) != 'nan'])
        test_prc = np.mean([x for x in test_prc if str(x) != 'nan'])
        test_rec = np.mean([x for x in test_rec if str(x) != 'nan'])
        test_acc = np.mean([x for x in test_acc if str(x) != 'nan'])

        print('Test AUC      : {}'.format(test_auc))
        print('Test F1       : {}'.format(test_f1))
        print('Test Precision: {}'.format(test_prc))
        print('Test Recall   : {}'.format(test_rec))
        print('Test Accuracy : {}'.format(test_acc))

    else:
        raise ValueError('Invalid PHASE')
