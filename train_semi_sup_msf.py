import builtins
import os
import sys
import time
import argparse
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from PIL import ImageFilter
from util import adjust_learning_rate, AverageMeter, subset_classes
import models.resnet as resnet
from tools import get_logger


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('data', type=str, help='path to dataset')
    parser.add_argument('--sup_split', type=str, default='1percent', required=True,
                        choices=['1percent', '10percent', '20percent', '50percent'],
                        help='use full or subset of the dataset')
    parser.add_argument('--debug', action='store_true', help='whether in debug mode or not')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=24, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='90,120', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--cos', action='store_true',
                        help='whether to cosine learning rate or not')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--sgd_momentum', type=float, default=0.9, help='SGD momentum')

    # model definition
    parser.add_argument('--arch', type=str, default='alexnet',
                        choices=['alexnet', 'resnet18', 'resnet50', 'mobilenet'])

    # Mean Shift
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--unsup_mbs', type=int, default=128000)
    parser.add_argument('--sup_mbs', type=int, default=12800)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--weak_strong', action='store_true',
                        help='whether to strong/strong or weak/strong augmentation')

    parser.add_argument('--weights', type=str, help='weights to initialize the model from')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')

    # GPU setting
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

    parser.add_argument('--checkpoint_path', default='output/mean_shift_default', type=str,
                        help='where to save checkpoints. ')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    return opt


# Extended version of ImageFolder to return index of image too.
class ImageFolderEx(datasets.ImageFolder):
    def __init__(self, root, sup_split_file, *args, **kwargs):
        super(ImageFolderEx, self).__init__(root, *args, **kwargs)

        with open(sup_split_file, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        sup_set = set(lines)
        samples = []
        for image_path, image_class in self.samples:
            image_name = image_path.split('/')[-1]
            samples.append((image_path, image_class if image_name in sup_set else -1))
        self.samples = samples

    def __getitem__(self, index):
        sample, target = super(ImageFolderEx, self).__getitem__(index)
        return index, sample, target


def get_mlp(inp_dim, hidden_dim, out_dim):
    mlp = nn.Sequential(
        nn.Linear(inp_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )
    return mlp


class MeanShift(nn.Module):
    def __init__(self, arch, m=0.99, unsup_mbs=128000, sup_mbs=12800, topk=5):
        super(MeanShift, self).__init__()

        # save parameters
        self.m = m
        self.unsup_mbs = unsup_mbs
        self.sup_mbs = sup_mbs
        self.topk = topk

        # create encoders and projection layers
        # both encoders should have same arch
        if 'resnet' in arch:
            self.encoder_q = resnet.__dict__[arch]()
            self.encoder_t = resnet.__dict__[arch]()

        # save output embedding dimensions
        # assuming that both encoders have same dim
        feat_dim = self.encoder_q.fc.in_features
        hidden_dim = feat_dim * 2
        proj_dim = feat_dim // 4

        # projection layers
        self.encoder_t.fc = get_mlp(feat_dim, hidden_dim, proj_dim)
        self.encoder_q.fc = get_mlp(feat_dim, hidden_dim, proj_dim)

        # prediction layer
        self.predict_q = get_mlp(proj_dim, hidden_dim, proj_dim)

        # copy query encoder weights to target encoder
        for param_q, param_t in zip(self.encoder_q.parameters(), self.encoder_t.parameters()):
            param_t.data.copy_(param_q.data)
            param_t.requires_grad = False

        print("using unsup mbs {}".format(self.unsup_mbs))
        print("using   sup mbs {}".format(self.sup_mbs))

        # setup queue (For Storing Random Targets)
        self.register_buffer('unsup_queue', torch.randn(self.unsup_mbs, proj_dim))
        self.register_buffer('sup_queue', torch.randn(self.sup_mbs, proj_dim))

        # normalize the queue embeddings
        self.unsup_queue = nn.functional.normalize(self.unsup_queue, dim=1)
        self.sup_queue = nn.functional.normalize(self.sup_queue, dim=1)

        # initialize the labels queue (For Purity measurement)
        self.register_buffer('sup_labels', -1*torch.ones(self.sup_mbs).long())

        # setup the queue pointer
        self.register_buffer('unsup_queue_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('sup_queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        for param_q, param_t in zip(self.encoder_q.parameters(), self.encoder_t.parameters()):
            param_t.data = param_t.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def data_parallel(self):
        self.encoder_q = torch.nn.DataParallel(self.encoder_q)
        self.encoder_t = torch.nn.DataParallel(self.encoder_t)
        self.predict_q = torch.nn.DataParallel(self.predict_q)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, targets, labels):
        u_ptr = int(self.unsup_queue_ptr)
        s_ptr = int(self.sup_queue_ptr)

        # separate sup targets and labels
        sup_targets = targets[torch.where(labels != -1)]
        sup_labels = labels[torch.where(labels != -1)]

        # if all sup data points won't fit in the queue
        # we slide back the pointer to accomodate them
        # TODO: handle wrap around?
        if (s_ptr + len(sup_targets)) > self.sup_mbs:
            s_ptr = self.sup_mbs - len(sup_targets)

        # add unsup targets, sup targets, and sup labels to the queues
        self.unsup_queue[u_ptr:u_ptr + len(targets)] = targets
        self.sup_queue[s_ptr:s_ptr + len(sup_targets)] = sup_targets
        self.sup_labels[s_ptr:s_ptr + len(sup_targets)] = sup_labels

        # update pointer positions
        self.unsup_queue_ptr[0] = (u_ptr + len(targets)) % self.unsup_mbs
        self.sup_queue_ptr[0] = (s_ptr + len(sup_targets)) % self.sup_mbs


    def forward(self, im_q, im_t, labels):
        # compute query features
        feat_q = self.encoder_q(im_q)
        # compute predictions for instance level regression loss
        query = self.predict_q(feat_q)
        query = nn.functional.normalize(query, dim=1)

        # compute target features
        with torch.no_grad():
            # update the target encoder
            self._momentum_update_target_encoder()

            # shuffle targets
            shuffle_ids, reverse_ids = get_shuffle_ids(im_t.shape[0])
            im_t = im_t[shuffle_ids]

            # forward through the target encoder
            current_target = self.encoder_t(im_t)
            current_target = nn.functional.normalize(current_target, dim=1)

            # undo shuffle
            current_target = current_target[reverse_ids].detach()
            self._dequeue_and_enqueue(current_target, labels)


        ########## UNSUP ###########

        iU = torch.where(labels == -1)
        cU = iU[0].shape[0]

        Q = query[iU]
        K = current_target[iU]
        M = self.unsup_queue.clone().detach()

        k = self.topk

        # 1. distances
        Dk = 2 - 2 * (K @ M.T)
        Dq = 2 - 2 * (Q @ M.T)

        # 2. topk with key distances
        _, iNDk = Dk.topk(k, dim=1, largest=False)

        # 3. gather query distances with topk inds
        NDq = torch.gather(Dq, 1, iNDk)

        # 4. unsup loss
        Lu = NDq.mean(dim=1).sum()

        ############################

        ########### SUP ############

        iS = torch.where(labels != -1)
        cS = iS[0].shape[0]

        if cS == 0:
            L = Lu / cU
            return L, (Lu / cU, cU), (None, cS), torch.tensor(0.0)

        Q = query[iS]
        K = current_target[iS]
        M = self.sup_queue.clone().detach()

        Lx = labels[iS]
        Lm = self.sup_labels.clone().detach()

        b = Q.shape[0]
        m = M.shape[0]
        k = self.topk

        Lx1 = Lx.unsqueeze(1).expand((b, m))
        Lm1 = Lm.unsqueeze(0).expand((b, m))

        # 1. distances
        Dk = 2 - 2 * (K @ M.T)
        Dq = 2 - 2 * (Q @ M.T)

        # 2. mask out non category distances
        Dk[torch.where(Lx1 != Lm1)] = 5.0

        # 3. topk with key distances
        _, iNDk = Dk.topk(k, dim=1, largest=False)

        # 4. gather query distances with topk inds
        NDq = torch.gather(Dq, 1, iNDk)

        # 5. unsup loss
        Ls = NDq.mean(dim=1).sum()

        ############################

        L = (Lu + Ls) / (cU + cS)

        return L, (Lu / cU, cU), (Ls / cS, cS), torch.tensor(0.0)


def get_shuffle_ids(bsz):
    """generate shuffle ids for ShuffleBN"""
    forward_inds = torch.randperm(bsz).long().cuda()
    backward_inds = torch.zeros(bsz).long().cuda()
    value = torch.arange(bsz).long().cuda()
    backward_inds.index_copy_(0, forward_inds, value)
    return forward_inds, backward_inds


class TwoCropsTransform:
    """Take two random crops of one image as the query and target."""
    def __init__(self, weak_transform, strong_transform):
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        print(self.weak_transform)
        print(self.strong_transform)

    def __call__(self, x):
        q = self.strong_transform(x)
        t = self.weak_transform(x)
        return [q, t]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


# Create train loader
def get_train_loader(opt):
    traindir = os.path.join(opt.data, 'train')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    augmentation_strong = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    augmentation_weak = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]

    sup_split_file = os.path.join('subsets', '{}.txt'.format(opt.sup_split))

    if opt.weak_strong:
        train_dataset = ImageFolderEx(
            traindir,
            sup_split_file,
            TwoCropsTransform(transforms.Compose(augmentation_weak),
                transforms.Compose(augmentation_strong)),
        )
    else:
        train_dataset = ImageFolderEx(
            traindir,
            sup_split_file,
            TwoCropsTransform(transforms.Compose(augmentation_strong),
                transforms.Compose(augmentation_strong)),
        )

    print('==> train dataset')
    print(train_dataset)

    # NOTE: remove drop_last
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True)

    return train_loader


def main():
    args = parse_option()
    os.makedirs(args.checkpoint_path, exist_ok=True)

    if not args.debug:
        os.environ['PYTHONBREAKPOINT'] = '0'
        logger = get_logger(
            logpath=os.path.join(args.checkpoint_path, 'logs'),
            filepath=os.path.abspath(__file__)
        )

        def print_pass(*arg):
            logger.info(*arg)
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    print(args)

    train_loader = get_train_loader(args)

    mean_shift = MeanShift(
        args.arch,
        m=args.momentum,
        unsup_mbs=args.unsup_mbs,
        sup_mbs=args.sup_mbs,
        topk=args.topk
    )
    mean_shift.data_parallel()
    mean_shift = mean_shift.cuda()
    print(mean_shift)

    params = [p for p in mean_shift.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=args.learning_rate,
                                momentum=args.sgd_momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True
    args.start_epoch = 1

    if args.weights:
        print('==> load weights from checkpoint: {}'.format(args.weights))
        ckpt = torch.load(args.weights)
        print('==> resume from epoch: {}'.format(ckpt['epoch']))
        if 'model' in ckpt:
            sd = ckpt['model']
        else:
            sd = ckpt['state_dict']
        msg = mean_shift.load_state_dict(sd, strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        args.start_epoch = ckpt['epoch'] + 1
        print(msg)

    if args.resume:
        print('==> resume from checkpoint: {}'.format(args.resume))
        ckpt = torch.load(args.resume)
        print('==> resume from epoch: {}'.format(ckpt['epoch']))
        mean_shift.load_state_dict(ckpt['state_dict'], strict=True)
        if not args.restart:
            optimizer.load_state_dict(ckpt['optimizer'])
            args.start_epoch = ckpt['epoch'] + 1

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        train(epoch, train_loader, mean_shift, optimizer, args)

        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # saving the model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'state_dict': mean_shift.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }

            save_file = os.path.join(args.checkpoint_path, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

            # help release GPU memory
            del state
            torch.cuda.empty_cache()


def train(epoch, train_loader, mean_shift, optimizer, opt):
    """
    one epoch training for CompReSS
    """
    mean_shift.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    lu_meter = AverageMeter()
    ls_meter = AverageMeter()
    purity_meter = AverageMeter()

    end = time.time()
    for idx, (indices, (im_q, im_t), labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        im_q = im_q.cuda(non_blocking=True)
        im_t = im_t.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # ===================forward=====================
        loss, (lu, cu), (ls, cs), purity = mean_shift(im_q=im_q, im_t=im_t, labels=labels)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), im_q.size(0))
        lu_meter.update(lu.item(), cu)
        if cs > 0:
            ls_meter.update(ls.item(), cs)
        purity_meter.update(purity.item(), im_q.size(0))

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]{space}'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f}){space}'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f}){space}'
                  'loss {loss.val:.3f} ({loss.avg:.3f}){space}'
                  'Lu {lu.val:.3f} {cu:3d} ({lu.avg:.3f}){space}'
                  'Ls {ls.val:.3f} {cs:2d} ({ls.avg:.3f}){space}'
                  # 'purity {purity.val:.3f} ({purity.avg:.3f})    '
                  .format(
                   epoch, idx + 1, len(train_loader),
                   batch_time=batch_time,
                   data_time=data_time,
                   purity=purity_meter,
                   loss=loss_meter,
                   lu=lu_meter,
                   cu=cu,
                   ls=ls_meter,
                   cs=cs,
                   space=' '*4
                   ))
            sys.stdout.flush()
            sys.stdout.flush()

    return loss_meter.avg


if __name__ == '__main__':
    main()
