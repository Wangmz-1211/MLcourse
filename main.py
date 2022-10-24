import argparse #定义与用户交互参数的包
import random
import time
import warnings
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import StepLR
from model import *
from data import *
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from utils import *

parser = argparse.ArgumentParser(description='AI-based loafing system')
parser.add_argument('--epochs', '-e', default=50, type=int, metavar='E', help='number of total epochs to run')
parser.add_argument('--batch_size', '-b', default=64, type=int, metavar='B', help='mini-batch size')
parser.add_argument('--learning_rate', '-lr', default=0.01, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to the latest checkpoint')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
best_acc = 0

def main():
    global best_acc
    args = parser.parse_args()
    if args.seed is not None: #如果人为设定了初始化的随机数seed
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    model = Model()
    loss_fun = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model.cuda()
        loss_fun.cuda()

    optim = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optim, step_size=20, gamma=0.1)#每20个epoch lrx0.1

    #从断点处恢复
    if args.resume:
        if os.path.isfile(args.resume):#如果checkpoint文件已经存在（是一个文件）
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    #加载data
    train_dataset = myDataset('./Dataset32', ToTensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = myDataset('./Dataset32', ToTensor)######改数据集文件
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    #训练时保存checkpoint
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train(train_loader, model, loss_fun, optim, epoch,  args)

        # evaluate on validation set
        acc = validate(val_loader, model, loss_fun, args)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict()
        }, is_best)


#定义train函数
def train(train_loader, model, loss_fun, optim, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f') #对每个epoch取平均
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accurate = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(########这个函数是干啥的
        len(train_loader),
        [batch_time, data_time, losses, accurate],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda()
        target = target.cuda()
        # compute output
        output = model(images)
        #print("train=", output)
        loss = loss_fun(output, target)

        # measure accuracy and record loss
        correct_k, acc = accuracy(output, target)
        #print("train=", correct_k)
        losses.update(loss.item(), images.size(0))
        accurate.update(acc.item(), images.size(0))
        #这里是调用了AverageMeter类中的方法，可以进行以batch为单位的参数更新和平均值更新，image.size[0]指的应该是batchsize

        # compute gradient and do SGD step
        optim.zero_grad()
        loss.backward()
        optim.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)#进度条



#验证集
def validate(val_loader, model, loss_fun, args):

    def run_validate(loader):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                #i = base_progress + i
                images = images.cuda()
                target = target.cuda()
                #这里为啥要.cuda


                # compute output
                output = model(images)
                #print("vali=", output)
                loss = loss_fun(output, target)

                # measure accuracy and record loss
                correct_k, acc = accuracy(output, target)
                #print("vali=", correct_k)
                losses.update(loss.item(), images.size(0))
                accurate.update(acc.item(), images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accurate = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, accurate],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)

    progress.display_summary()

    return accurate.avg



if __name__ == '__main__':
    main()