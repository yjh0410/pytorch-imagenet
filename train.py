import argparse
import os
import random
import shutil
import time
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils.misc import ModelEMA, FLOPs_and_Params
import backbone as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# basic
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--max_epoch', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr_epoch', nargs='+', default=[30, 60], type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total ')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--save_folder', default='weights/', type=str,
                    help='path to save model. ')
parser.add_argument('--ema', action='store_true', default=False,
                    help='use ema training trick')
# warmup
parser.add_argument('--wp_epoch', default=1, type=int, 
                    help='iteration of warmup stage')
# lr schedule
parser.add_argument('--lr_schedule', default='step', type=str,
                    help='step, cos. ')
# optimizer
parser.add_argument('--optimizer', default='sgd', type=str,
                    help='sgd, adamw')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
# dataset
parser.add_argument('--data_root', metavar='DIR', default='./data/imagenet/',
                    help='path to dataset')
# model
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--clip_max_norm', default=-1.0, type=float,
                    help='gradient clipping max norm')

best_acc1 = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


def main():
    global save_folder
    args = parser.parse_args()

    save_folder = os.path.join(args.save_folder, args.arch)
    os.makedirs(save_folder, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # Simply call main_worker function
    main_worker(args)


def main_worker(args):
    global best_acc1, save_folder
    
    print(args)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
        print(model)
        
    model.to(device)

    # FLOPs and Params
    FLOPs_and_Params(model, size=224, device=device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    # optionally resume from a checkpoint
    if args.resume:
        print('continue training ...')
        model.load_state_dict(torch.load(args.resume))

    lr_epoch = args.lr_epoch
    lr = args.lr

    # optimizer
    if args.optimizer == 'sgd':
        print('Optimizer: SGD')
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    elif args.optimizer == 'adamw':
        print('Optimizer: AdamW')
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                    weight_decay=args.weight_decay)

    # data dir
    traindir = os.path.join(args.data_root, 'train')
    valdir = os.path.join(args.data_root, 'val')

    # train dataset and dataloader
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing()
        ]))
    train_loader = torch.utils.data.DataLoader(
                        train_dataset, 
                        batch_size=args.batch_size, 
                        shuffle=True,
                        num_workers=args.workers, 
                        pin_memory=True)
    # val dataset and dataloader
    val_dataset = datasets.ImageFolder(
        valdir, 
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))                  
    val_loader = torch.utils.data.DataLoader(
                        val_dataset,
                        batch_size=args.batch_size, 
                        shuffle=False,
                        num_workers=args.workers, 
                        pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    # EMA
    ema = ModelEMA(model) if args.ema else None

    epoch_size = len(train_dataset) // args.batch_size
    print("total training epochs: %d " % (args.max_epoch))
    if args.lr_schedule == 'step':
        print("lr step epoch: ", lr_epoch)
    elif args.lr_schedule == 'cos':
        print("Cos lr schedule")

    for epoch in range(args.start_epoch, args.max_epoch):
        # use step lr decay
        if args.lr_schedule == 'step':
            if epoch in args.lr_epoch:
                print('lr decay ...')
                lr = lr * 0.1
                set_lr(optimizer, lr)
        # use cos lr decay
        elif args.lr_schedule == 'cos':
            lr = 1e-5 + 0.5*(lr - 1e-5)*(1 + math.cos(math.pi*epoch / args.max_epoch))
            set_lr(optimizer, lr)

        # train for one epoch
        lr = train(train_loader, ema, model, criterion, optimizer, epoch, lr, epoch_size, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            torch.save(model.state_dict(), save_folder+'/'+ str(args.arch) + '_' + str(epoch + 1)+'_'+str(acc1.item())+'.pth')


def train(train_loader, ema, model, criterion, optimizer, epoch, lr, epoch_size, args):
    print("start training ......")
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    lrp = AverageMeter('lr', ':6f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, lrp],
        prefix="Epoch: [{}]".format(epoch))

    lrp.update(lr)
    # switch to train mode
    model.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # warmup
        ni = i + epoch * epoch_size
        if epoch < args.wp_epoch:
            nw = args.wp_epoch * epoch_size
            lr = args.lr * pow(ni / nw, 4)
            set_lr(optimizer, lr)
            lrp.update(lr)
        elif epoch == args.wp_epoch and i == 0:
            # warmup is over
            lr = args.lr
            set_lr(optimizer, lr)
            lrp.update(lr)

        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        optimizer.step()

        # ema update
        if args.ema:
            ema.update(model)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return lr


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
