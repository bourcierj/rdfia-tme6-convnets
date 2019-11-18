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
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from tme6 import *

PRINT_INTERVAL = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    cudnn.benchmark = True

class ConvNet(nn.Module):
    """
    CNN architecture close to AlexNet from Krizhevsky et al (2012).
    """
    def __init__(self):
        super(ConvNet, self).__init__()

        self.features = nn.Sequential(
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
            nn.MaxPool2d((2, 2), stride=2, padding=0, ceil_mode=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 1000),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1000, 10),
        )

    def forward(self, input):
        bsize = input.size(0)
        output = self.features(input)
        output = output.view(bsize, -1)
        output = self.classifier(output)
        return output


def get_dataloaders(batch_size, path):
    """
    This function loads the MNIST dataset and add transformations on every image
    (listed in `transform=...`)
    """
    train_dataset = datasets.CIFAR10(path, train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(28),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.491, 0.482, 0.447], [0.202, 0.199, 0.201])

        ]))
    test_dataset = datasets.CIFAR10(path, train=False, download=True,
        transform=transforms.Compose([
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            transforms.Normalize([0.491, 0.482, 0.447], [0.202, 0.199, 0.201]),
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=(device.type == 'cuda'),
        num_workers=torch.multiprocessing.cpu_count())
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        pin_memory=(device.type == 'cuda'),
        num_workers=torch.multiprocessing.cpu_count())

    return train_loader, test_loader


def epoch(loader, model, criterion, optimizer=None):
    """
    Execute a pass over the dataset (an epoch).
    If `optimizer` is given, do a training epoch using the optimizer, otherwise do an
    evaluation epoch of the model (no backward).
    """
    model.eval() if optimizer is None else model.train()

    # objets to store averages and metrics
    avg_loss = AverageMeter()
    avg_top1_acc = AverageMeter()
    avg_top5_acc = AverageMeter()
    avg_batch_time = AverageMeter()
    global loss_plot

    # if in evaluation mode, use a torch.no_grad() context manager
    # else, use a null context manager
    if criterion is None:
        torch_cm = torch.no_grad()
    else:
        torch_cm = contextlib.nullcontext()
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
            # log infos
            if idx % PRINT_INTERVAL == 0:
                print('[{0:s} Batch {1:03d}/{2:03d}]\t'
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:5.1f} ({top1.avg:5.1f})\t'
                      'Prec@5 {top5.val:5.1f} ({top5.avg:5.1f})'
                      .format("EVAL" if optimizer is None else "TRAIN", idx, len(loader),
                              batch_time=avg_batch_time, loss=avg_loss,
                              top1=avg_top1_acc, top5=avg_top5_acc)
                      )
                if optimizer:
                    loss_plot.plot()

    # Log infos on epoch
    print('\n===============> Total time {batch_time:d}s\t'
          'Avg loss {loss.avg:.4f}\t'
          'Avg Prec@1 {top1.avg:5.2f} %\t'
          'Avg Prec@5 {top5.avg:5.2f} %\n'
          .format(batch_time=int(avg_batch_time.sum), loss=avg_loss,
                  top1=avg_top1_acc, top5=avg_top5_acc)
          )
    return avg_top1_acc, avg_top5_acc, avg_loss


def main(params):

    # example of parameters:
    #   {"batch_size": 128, "epochs": 5, "lr": 0.1, "path": '/tmp/datasets/mnist'}

    # define model, loss, optimizer
    model = ConvNet()
    criterion = nn.CrossEntropyLoss()
    # use SGD with Netsterov's momentum of 0.9
    optimizer = torch.optim.SGD(model.parameters(), params.lr, momentum=0.9,
                                nesterov=True)
    # use an exponential LR scheduler
    lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    model = model.to(device)
    criterion = criterion.to(device)
    # Get all data
    train_loader, test_loader = get_dataloaders(params.batch_size, params.path)
    # path where to save figures
    savefig_path = Path(params.savefig_path)
    savefig_path.mkdir(parents=True, exist_ok=True)

    # init plots
    plot = AccLossPlot()
    global loss_plot
    loss_plot = TrainLossPlot()
    print(f"Training on {'GPU' if device.type == 'cuda' else 'CPU'}")
    # Iterate on epochs
    for i_epoch in range(params.epochs):
        print("=================\n=== EPOCH "+str(i_epoch+1)+" =====\n=================\n")
        # train phase
        top1_acc, avg_top5_acc, loss = epoch(train_loader, model, criterion, optimizer)
        # evaluation phase
        top1_acc_test, top5_acc_test, loss_test = epoch(test_loader, model, criterion)
        # plot
        plot.update(loss.avg, loss_test.avg, top1_acc.avg, top1_acc_test.avg)
        # modify learning rate
        lr_sched.step()

    plot.savefig(savefig_path/'Acc_loss.svg')
    loss_plot.savefig(savefig_path/'Train_loss.svg')

if __name__ == '__main__':

    # Command-line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./data', type=str,
                        metavar='DIR', help='path to dataset')
    parser.add_argument('--epochs', default=5, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=128, type=int, metavar='N',
                        help='mini-batch size (default: 128)')
    parser.add_argument('--lr', default=0.1, type=float, metavar='LR',
                        help='learning rate')
    parser.add_argument('--savefig-path', default='./cifar10_expes', type=str, metavar='DIR',
                        help='where to save figures')
    args = parser.parse_args()

    main(args)

    print('Done.')
q
