# --- Base packages ---
import argparse
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
from datasets import MIMIC, NLMCXR, TextDataset
from models import Classifier, TNN
from baselines.transformer.models import LSTM_Attn

# --- Hyperparameters ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.manual_seed(seed=0)


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
    parser.add_argument('--epoch_milestone', type=int, default=25,
                        help='Reduce LR by 10 after reaching milestone epochs')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='DenseNet121', choices=['DenseNet121', 'resnet101'],
                        help='the visual extractor to be used.')

    parser.add_argument('--model_name', type=str, default='Int', choices=['ClsGenInt', 'ClsGen', 'Int'],
                        help='Model Type.')
    parser.add_argument('--sources', type=list, default=['caption'], help='Source Texts.')
    parser.add_argument('--targets', type=list, default=['label'], help='Target Texts.')
    parser.add_argument('--kwargs_sources', type=list, default=['txt'], help='Kwargs Source Texts.')

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
    parser.add_argument('--lr_nlm', type=float, default=3e-5, help='the learning rate for NLM-CXR.')
    parser.add_argument('--lr_mimic', type=float, default=3e-6, help='the learning rate for MIMIC-CXR.')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='the weight decay.')

    return parser.parse_args()


def main(args):
    TEXT_FILE = '/content/drive/MyDrive/outputs/x_{}_ClsGenInt_DenseNet121_MaxView2_NumLabel114_History_Hyp.txt'.format(
        args.dataset_name)
    LABEL_FILE = '/content/drive/MyDrive/outputs/x_{}_ClsGenInt_DenseNet121_MaxView2_NumLabel114_History_Lbl.txt'.format(
        args.dataset_name)

    vocab_size, posit_size, comment, dataset = None, None, None, None
    if args.dataset_name == 'MIMIC':
        dataset = TextDataset(text_file=TEXT_FILE, label_file=LABEL_FILE, sources=args.sources, targets=args.targets,
                              vocab_file=args.dataset_dir+'mimic_unigram_1000.model', max_len=1000)

        vocab_size = len(dataset.vocab)
        posit_size = dataset.max_len
        comment = 'MaxView{}_NumLabel{}'.format(args.max_views, args.decease_related_topics)

    elif args.dataset_name == 'NLMCXR':
        dataset = TextDataset(text_file=TEXT_FILE, label_file=LABEL_FILE, sources=args.sources, targets=args.targets,
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
                                                                                      comment, args.trial)
    last_epoch, (best_metric, test_metric) = load(checkpoint_path_from, model)
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