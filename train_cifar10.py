import argparse
import os
import time
import contextlib  # used only for contextlib.nullcontext which requires python > 3.7
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.optim.lr_scheduler
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter

from tme6_utils import *

PRINT_INTERVAL = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    cudnn.benchmark = True

class ConvNet(nn.Module):
    """CNN architecture close to AlexNet from Krizhevsky et al (2012)."""

    def __init__(self, dropout_p=0., batch_norm=False):
        super(ConvNet, self).__init__()

        layers = [
            nn.Conv2d(3, 32, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0),
            nn.Conv2d(32, 64, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0),
            nn.Conv2d(64, 64, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2, padding=0, ceil_mode=True)
        ]
        if not batch_norm:  # remove batch norm layers
            layers = [l for l in layers if not isinstance(l, nn.BatchNorm2d)]

        self.features = nn.Sequential(*layers)
        layers = [
            nn.Linear(1024, 1000),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(1000, 10),
        ]
        if dropout_p == 0.:  # remove dropout
            layers = [l for l in layers if not isinstance(l, nn.Dropout)]
        self.classifier = nn.Sequential(*layers)

    def forward(self, input):
        bsize = input.size(0)
        output = self.features(input)
        output = output.view(bsize, -1)
        output = self.classifier(output)
        return output


def get_dataloaders(batch_size, path, data_augment=False, normalize=False):
    """
    This function loads the MNIST dataset and add transformations on every image
    (listed in `transform=...`).
    """
    train_tfms, test_tfms = [], []
    if data_augment:
        train_tfms.extend([
            transforms.RandomCrop(28),
            transforms.RandomHorizontalFlip()])
        test_tfms.extend([
            transforms.CenterCrop(28)])
    train_tfms.extend([
        transforms.ToTensor()])
    test_tfms.extend([
        transforms.ToTensor()])
    if normalize:
        train_tfms.extend([
            transforms.Normalize([0.491, 0.482, 0.447], [0.202, 0.199, 0.201])])
        test_tfms.extend([
            transforms.Normalize([0.491, 0.482, 0.447], [0.202, 0.199, 0.201])])

    train_dataset = datasets.CIFAR10(path, train=True, download=True,
        transform=transforms.Compose(train_tfms))
    test_dataset = datasets.CIFAR10(path, train=False, download=True,
        transform=transforms.Compose(test_tfms))

    # train_dataset = datasets.CIFAR10(path, train=True, download=True,
    #     transform=transforms.Compose([
    #         transforms.RandomCrop(28),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.491, 0.482, 0.447], [0.202, 0.199, 0.201])

    #     ]))
    # test_dataset = datasets.CIFAR10(path, train=False, download=True,
    #     transform=transforms.Compose([
    #         transforms.CenterCrop(28),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.491, 0.482, 0.447], [0.202, 0.199, 0.201]),
    #     ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=(device.type == 'cuda'),
        num_workers=torch.multiprocessing.cpu_count())
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        pin_memory=(device.type == 'cuda'),
        num_workers=torch.multiprocessing.cpu_count())

    return train_loader, test_loader


def train(model, criterion, optimizer, train_loader, val_loader, epochs, lr_sched=None,
          savefig_dir='./', writer=None):
    """Full training loop"""

    def epoch(loader, model, criterion, optimizer=None):
        """
        Execute a pass over the dataset (an epoch).
        If `optimizer` is given, do a training epoch using the optimizer, otherwise do an
        evaluation epoch of the model (no bckward).
        """
        # objets to store averages and metrics
        avg_loss = AverageMeter()
        avg_top1_acc = AverageMeter()
        avg_top5_acc = AverageMeter()
        avg_batch_time = AverageMeter()
        if optimizer is None:
            # if in evaluation mode, use a torch.no_grad() context manager
            # else, use a null context manager
            torch_cm = torch.no_grad()
            model.eval()
            ROLE = 'Eval'
        else:
            torch_cm = contextlib.nullcontext()
            model.train()
            ROLE = 'Train'
        nonlocal ite
        # iterate on dataset batches
        tic = time.time()
        with torch_cm:
            for idx, (input, target) in enumerate(loader):
                input, target = input.to(device), target.to(device)
                # forward
                output = model(input)
                loss = criterion(output, target)
                # backward if we are in train
                if optimizer:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    ite += 1

                # compute metrics
                prec1, prec5 = accuracy(output, target, topk=(1, 5))
                batch_time = time.time() - tic
                tic = time.time()
                # update averages
                avg_loss.update(loss.item())
                avg_top1_acc.update(prec1.item())
                avg_top5_acc.update(prec5.item())
                avg_batch_time.update(batch_time)
                if optimizer:
                    loss_plot.update(avg_loss.val)
                    if writer:
                        writer.add_scalar('Loss/Train-batch-loss', avg_loss.val, ite)
                # log infos
                if idx % PRINT_INTERVAL == 0:
                    print('[{0:s} Batch {1:03d}/{2:03d}]\t'
                          'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:5.1f} ({top1.avg:5.1f})\t'
                          'Prec@5 {top5.val:5.1f} ({top5.avg:5.1f})'
                          .format(ROLE, idx, len(loader),
                                  batch_time=avg_batch_time, loss=avg_loss,
                                  top1=avg_top1_acc, top5=avg_top5_acc)
                          )
                    # if optimizer:
                    #     loss_plot.plot()
        if writer:
            writer.add_scalar(f'Loss/{ROLE}-epoch-loss', avg_loss.avg, i_epoch)
            writer.add_scalar(f'Accuracy/{ROLE}-epoch-acc', avg_top1_acc.avg, i_epoch)
            writer.add_scalar(f'Accuracy/{ROLE}-epoch-top5-acc', avg_top5_acc.avg, i_epoch)

        # Log infos on epoch
        print('\n===============> Total time {batch_time:d}s\t'
              'Avg loss {loss.avg:.4f}\t'
              'Avg Prec@1 {top1.avg:5.2f} %\t'
              'Avg Prec@5 {top5.avg:5.2f} %\n'
              .format(batch_time=int(avg_batch_time.sum), loss=avg_loss,
                      top1=avg_top1_acc, top5=avg_top5_acc)
              )
        return avg_top1_acc, avg_top5_acc, avg_loss


    # init plots
    plot = AccLossPlot()
    loss_plot = TrainLossPlot()
    print(f"Training on {'GPU' if device.type == 'cuda' else 'CPU'}\n")
    ite = 0  # number of batch updates
    # Iterate on epochs
    for i_epoch in range(1, epochs+1):
        print("=================\n=== EPOCH "+str(i_epoch)+" =====\n=================\n")
        # train phase
        top1_acc, avg_top5_acc, loss = epoch(train_loader, model, criterion, optimizer)
        # evaluation phase
        top1_acc_test, top5_acc_test, loss_test = epoch(val_loader, model, criterion)
        # plot
        plot.update(loss.avg, loss_test.avg, top1_acc.avg, top1_acc_test.avg)
        # plot.plot()
        # modify learning rate
        if lr_sched:
            lr_sched.step()

    plot.savefig(savefig_dir/'Accuracy-loss.svg')
    loss_plot.savefig(savefig_dir/'Train-batch-loss.svg')
    # get max accuracy obtained
    max_epoch, max_acc = max(enumerate(plot.acc_test), key=lambda x: x[1])
    max_epoch += 1
    return max_epoch, max_acc


def main(args):

    # define model, loss, optimizer
    model = ConvNet(args.dropout_p, args.batch_norm).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    if args.momentum:  # use SGD with a momentum of 0.9
        optim_kwargs = dict(momentum=0.9)
    else:
        optim_kwargs = dict()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, **optim_kwargs)
    # use an exponential LR scheduler
    if args.exponential_lr_sched:
        lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        lr_sched = None
    # Get all data
    train_loader, test_loader = \
        get_dataloaders(args.batch_size, args.data_path, args.data_augment,
                        args.normalize)

    ignore_keys = {'data_path', 'no_tensorboard'}
    # get hyperparameters with values in a dict
    hparams = {key.replace('_','-'): val for key, val in vars(args).items() if key not in ignore_keys}
    # generate a name for the experiment
    expe_name = '_'.join([f"{key}={val}" for key, val in hparams.items()])
    # path where to save the figures
    savefig_dir = Path('./CIFAR10_expes/')/expe_name
    savefig_dir.mkdir(parents=True, exist_ok=True)
    # Tensorboard summary writer
    if args.no_tensorboard:
        writer = None
    else:
        writer = SummaryWriter(comment='__CIFAR10__'+expe_name, flush_secs=10)
        # log a batch of input images
        input, target = next(iter(train_loader))
        grid = torchvision.utils.make_grid(input)
        writer.add_image('Train-input-images', grid, 0)

    max_epoch, max_acc = train(model, criterion, optimizer, train_loader, test_loader,
                               args.epochs, lr_sched, savefig_dir, writer)
    print(f"End. Max Acc: {max_acc:5.2f} %\t"
          f"Max Epoch: {max_epoch}"
          )
    # log accuracy for hyperparameters
    if writer:
        writer.add_hparams(hparams, {'accuracy': max_acc})

if __name__ == '__main__':

    # Command-line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='./data', type=str,
                        help="path to dataset (default: ./data")
    parser.add_argument('--no-tensorboard', action='store_true',
                        help="if specified, do not log metrics to tensorboard")

    parser.add_argument('--epochs', default=30, type=int,
                        help="number of total epochs to run (default: 30)")
    parser.add_argument('--batch-size', default=128, type=int,
                        help="mini-batch size (default: 128)")
    parser.add_argument('--lr', default=0.1, type=float,
                        help="learning rate (default: 0.1)")
    # optional improvements
    parser.add_argument('--normalize', action='store_true',
                        help="if specified, normalize data")
    parser.add_argument('--data-augment', action='store_true',
                        help="if specified, do some data augmentation")
    parser.add_argument('--momentum', action='store_true',
                        help="if specified, use SGD with a momentum of 0.9")
    parser.add_argument('--exponential-lr-sched', action='store_true',
                        help="if specified, use an exponential LR scheduler")
    parser.add_argument('--dropout_p', default=0., type=float,
                        help="proba of dropout between layers fc4 and fc5 of the CNN (default: 0. - no dropout)")
    parser.add_argument('--batch-norm', action='store_true',
                        help="if specified, use batch normalization after conv layers of the CNN")
    args = parser.parse_args()
    torch.manual_seed(42)  # random seed for reproducibility
    main(args)

    print('Done.')
