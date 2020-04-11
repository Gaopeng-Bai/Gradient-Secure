#!/usr/bin/python3
# -*-coding:utf-8 -*-

# Reference:**********************************************
# @Time    : 4/7/2020 7:23 PM
# @Author  : Gaopeng.Bai
# @File    : syft_main.py
# @User    : gaopeng bai
# @Software: PyCharm
# @Description: 
# Reference:**********************************************
import argparse
import syft as sy

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

from utils.dataloader import data_loader
from utils.model import model_select
from utils.Average import AverageMeter

parser = argparse.ArgumentParser(description='PyTorch Cifar100 Training')
parser.add_argument('--model', default="resnet20", type=str,
                    metavar='N', help='choose a model to use (eg..resnet20, resnet32, resnet44, resnet110'
                                      'preact_resnet110, resnet164, resnet1001, preact_resnet164, preact_resnet1001'
                                      'wide_resnet, resneXt, densenet)')
parser.add_argument('--epochs', default=200, type=int,
                    metavar='N', help='number of total epochs to run')
parser.add_argument('--worker_iter', default=5, type=int,
                    metavar='N', help='worker iterations(times of training in specify worker)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1,
                    type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4,
                    type=float, metavar='W', help='weight decay (default: 1e-4)')
args = parser.parse_args()


class syft_model:
    def __init__(self, arg):
        self.arg = arg
        hook = sy.TorchHook(torch)
        #  connect to two remote workers that be call alice and bob and request
        #  another worker called the crypto_provider who gives all the crypto primitives we may need
        self.bob = sy.VirtualWorker(hook, id="bob")
        self.alice = sy.VirtualWorker(hook, id="alice")
        self.secure_worker = sy.VirtualWorker(hook, id="secure_worker")

        # self.compute_nodes = [self.bob, self.alice]
        # load data
        self.train_loader, self.test_loader = data_loader(self.arg)
        # self.data_to_workers()
        # pre model prepare
        self.model = model_select(self.arg.model)
        self.bobs_model = self.model
        self.alice_model = self.model
        self.reload_model()

    def __call__(self):
        for i in range(self.arg.epochs):
            self.train()
            bob_loss, prc = self.test(self.bobs_model)
            print('Epoch: [{}/{}]\t'
                  'Loss_bob: ({:.3})\t'
                  'Prec_bob {top1.val:.1f}% ({top1.avg:.1f}%))'.format(
                i, self.arg.epochs, bob_loss, top1=prc))

    def reload_model(self):
        """
        set all model
        Returns:

        """
        self.bobs_optimizer = optim.SGD(self.bobs_model.parameters(), self.arg.lr,
                                        momentum=self.arg.momentum, weight_decay=self.arg.weight_decay)
        self.alice_optimizer = optim.SGD(self.alice_model.parameters(), self.arg.lr,
                                         momentum=self.arg.momentum, weight_decay=self.arg.weight_decay)
        self.params = [list(self.bobs_model.parameters()), list(self.alice_model.parameters())]

    def data_to_workers(self):
        # split training data into two workers bob and alice
        self.train_distributed_dataset = []

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.send(self.compute_nodes[batch_idx % len(self.compute_nodes)])
            target = target.send(self.compute_nodes[batch_idx % len(self.compute_nodes)])
            self.train_distributed_dataset.append((data, target))

    def train(self):
        for batch_idx, (data, target) in enumerate(self.train_loader):
            print(batch_idx)
            split = batch_idx % 2
            if split:
                self.alice_model.train()
                self.alice_optimizer.zero_grad()
                alice_pred = self.alice_model(data)
                alice_loss = F.cross_entropy(alice_pred, target)
                alice_loss.backward()
                self.alice_optimizer.step()

            else:
                self.bobs_model.train()
                self.bobs_optimizer.zero_grad()
                bobs_pred = self.bobs_model(data)
                bobs_loss = F.cross_entropy(bobs_pred, target)
                bobs_loss.backward()
                self.bobs_optimizer.step()

        # encrypted aggregation
        new_params = list()
        for param_i in range(len(self.params[0])):
            spd_params = list()
            # iterate all workers
            for index in range(2):
                clip_grad_norm_(self.bobs_model.parameters(), max_norm=20)
                clip_grad_norm_(self.alice_model.parameters(), max_norm=20)
                # aggregation same parameters from every workers. Then encrypt it into individual worker depends on trusted worker for all.
                spd_params.append(self.params[index][param_i].copy().fix_precision().share(self.bob, self.alice,
                                                                                           crypto_provider=self.secure_worker))
            # decrypt parameter.
            new = (spd_params[0] + spd_params[1]).get().float_precision() / 2
            new_params.append(new)
        # clean up
        with torch.no_grad():
            # iterate all parameters
            for model in self.params:
                for param in model:
                    param *= 0

            # set new parameters in all sub workers, bob and alice.
            for remote_index in range(2):
                for param_index in range(len(self.params[remote_index])):
                    self.params[remote_index][param_index].set_(new_params[param_index])

    def test(self, model):
        model.eval()
        test_loss = 0
        acc = AverageMeter()
        for data, target in self.test_loader:
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            prc = accuracy(output, target)[0]
            acc.update(prc.item(), data.size(0))

        test_loss /= len(self.test_loader.dataset)
        return test_loss, acc


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    a = syft_model(args)
    a()
