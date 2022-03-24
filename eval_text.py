# --- Base packages ---
import argparse
import os
import numpy as np
import sklearn.metrics as metrics

# --- PyTorch packages ---
import torch
import torch.nn as nn
import torch.utils.data as data

# --- Project Packages ---
from utils import load, test
from datasets import TextDataset
from models import Classifier, TNN


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.manual_seed(seed=0)


def parse_agruments():
    parser = argparse.ArgumentParser()

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
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    parser.add_argument('--model_name', type=str, default='Int', help='Model Type.')
    parser.add_argument('--sources', type=list, default=['caption'], help='Source Texts.')
    parser.add_argument('--targets', type=list, default=['label'], help='Target Texts.')
    parser.add_argument('--kwargs_sources', type=list, default=['txt'], help='Kwargs Source Texts.')

    parser.add_argument('--num_embed', type=int, default=2048, help='visual and textual embedding size')
    parser.add_argument('--fd_dim', type=int, default=256, help='forward dimension')

    return parser.parse_args()


def main(args):
    hyp_file = '/content/drive/MyDrive/outputs/x_{}_ClsGenInt_DenseNet121_MaxView2_NumLabel114_History_{}_Hyp.txt'.\
        format(args.dataset_name, args.trial)
    lbl_file = '/content/drive/MyDrive/outputs/x_{}_ClsGenInt_DenseNet121_MaxView2_NumLabel114_History_{}_Lbl.txt'.\
        format(args.dataset_name, args.trial)

    vocab_size, posit_size, comment, dataset = None, None, None, None
    if args.dataset_name == 'MIMIC':
        dataset = TextDataset(text_file=hyp_file, label_file=lbl_file, sources=args.sources, targets=args.targets,
                              vocab_file=args.dataset_dir+'mimic_unigram_1000.model', max_len=1000)

        vocab_size = len(dataset.vocab)
        posit_size = dataset.max_len
        comment = 'MaxView{}_NumLabel{}'.format(args.max_views, args.decease_related_topics)

    elif args.dataset_name == 'NLMCXR':
        dataset = TextDataset(text_file=hyp_file, label_file=lbl_file, sources=args.sources, targets=args.targets,
                              vocab_file=args.dataset_dir+'nlmcxr_unigram_1000.model', max_len=1000)

        vocab_size = len(dataset.vocab)
        posit_size = dataset.max_len
        comment = 'MaxView{}_NumLabel{}'.format(args.max_views, args.decease_related_topics)

    tnn = TNN(embed_dim=args.num_embed, num_heads=args.num_heads, fwd_dim=args.fd_dim, dropout=args.dropout,
              num_layers=1, num_tokens=vocab_size, num_posits=posit_size)
    model = Classifier(num_topics=args.decease_related_topics, num_states=args.num_classes, cnn=None, tnn=tnn,
                       embed_dim=args.num_embed, num_heads=args.num_heads, dropout=args.dropout)

    data_loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    model = nn.DataParallel(model).cuda()
    checkpoint_path_from = '/content/drive/MyDrive/checkpoints/{}_{}_{}_{}.pt'.format(args.dataset_name,
                                                                                      args.model_name,
                                                                                      comment, args.num_embed)
    last_epoch, (_, _) = load(checkpoint_path_from, model)
    test_loss, test_outputs, test_targets = test(data_loader, model, device='cuda', kw_src=args.kwargs_sources)

    # --- Evaluation ---
    test_auc = []
    test_f1 = []
    test_prc = []
    test_rec = []
    test_acc = []

    threshold = 0.5
    NUM_LABELS = 14
    for i in range(NUM_LABELS):
        try:
            test_auc.append(metrics.roc_auc_score(test_targets.cpu()[..., i], test_outputs.cpu()[..., i, 1]))
            test_f1.append(metrics.f1_score(test_targets.cpu()[..., i], test_outputs.cpu()[..., i, 1] > threshold))
            test_prc.append(
                metrics.precision_score(test_targets.cpu()[..., i], test_outputs.cpu()[..., i, 1] > threshold))
            test_rec.append(metrics.recall_score(test_targets.cpu()[..., i], test_outputs.cpu()[..., i, 1] > threshold))
            test_acc.append(
                metrics.accuracy_score(test_targets.cpu()[..., i], test_outputs.cpu()[..., i, 1] > threshold))

        except:
            print('An error occurs for label', i)

    test_auc = np.mean([x for x in test_auc if str(x) != 'nan'])
    test_f1 = np.mean([x for x in test_f1 if str(x) != 'nan'])
    test_prc = np.mean([x for x in test_prc if str(x) != 'nan'])
    test_rec = np.mean([x for x in test_rec if str(x) != 'nan'])
    test_acc = np.mean([x for x in test_acc if str(x) != 'nan'])

    print('Accuracy       : {}'.format(test_acc))
    print('Macro AUC      : {}'.format(test_auc))
    print('Macro F1       : {}'.format(test_f1))
    print('Macro Precision: {}'.format(test_prc))
    print('Macro Recall   : {}'.format(test_rec))
    print('Micro AUC      : {}'.format(
        metrics.roc_auc_score(test_targets.cpu()[..., :NUM_LABELS] == 1, test_outputs.cpu()[..., :NUM_LABELS, 1],
                              average='micro')))
    print('Micro F1       : {}'.format(
        metrics.f1_score(test_targets.cpu()[..., :NUM_LABELS] == 1, test_outputs.cpu()[..., :NUM_LABELS, 1] > threshold,
                         average='micro')))
    print('Micro Precision: {}'.format(metrics.precision_score(test_targets.cpu()[..., :NUM_LABELS] == 1,
                                                               test_outputs.cpu()[..., :NUM_LABELS, 1] > threshold,
                                                               average='micro')))
    print('Micro Recall   : {}'.format(metrics.recall_score(test_targets.cpu()[..., :NUM_LABELS] == 1,
                                                            test_outputs.cpu()[..., :NUM_LABELS, 1] > threshold,
                                                            average='micro')))


if __name__ == "__main__":
    args = parse_agruments()
    main(args)
