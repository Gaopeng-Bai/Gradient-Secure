#!/usr/bin/python3
# -*-coding:utf-8 -*-

# Reference:**********************************************
# @Time    : 4/7/2020 7:23 PM
# @Author  : Gaopeng.Bai
# @File    : secure_gradient.py
# @User    : gaopeng bai
# @Software: PyCharm
# @Description:
# Reference:**********************************************
import argparse
import numpy as np
import syft as sy

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

from utils.dataloader import data_loader
from utils.model import model_select
from utils.Average import AverageMeter
from utils.vhe import *

parser = argparse.ArgumentParser(
    description='PyTorch secure gradient Training')
parser.add_argument('--dataset', default="mnist", type=str,
                    metavar='N', help='mnist or cifar100')
parser.add_argument(
    '--model',
    default="lenet5",
    type=str,
    metavar='N',
    help='choose a model to use mnist(lenet5, simply_cnn, alexnet) or for cifar100 datasets(resnet20, resnet32, resnet44, resnet110'
    'preact_resnet110, resnet164, resnet1001, preact_resnet164, preact_resnet1001'
    'wide_resnet, resneXt, densenet)')
parser.add_argument('--epochs', default=15, type=int,
                    metavar='N', help='number of total epochs to run')
parser.add_argument(
    '--worker_iter',
    default=5,
    type=int,
    metavar='N',
    help='worker iterations(times of training in specify worker)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1,
                    type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='M', help='momentum')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4, lenet with mnist suggest:1e-2)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
args = parser.parse_args()


def update(data, target, model, optimizer):
    model.train()
    optimizer.zero_grad()
    alice_pred = model(data)
    alice_loss = F.cross_entropy(alice_pred, target)
    alice_loss.backward()
    optimizer.step()


class syft_model:
    def __init__(self, arg):
        self.arg = arg
        hook = sy.TorchHook(torch)
        #  connect to two remote workers that be call alice and bob and request
        # another worker called the crypto_provider who gives all the crypto
        # primitives we may need
        self.bob = sy.VirtualWorker(hook, id="bob")
        self.alice = sy.VirtualWorker(hook, id="alice")
        self.secure_worker = sy.VirtualWorker(hook, id="secure_worker")

        # load data
        self.train_loader, self.test_loader = data_loader(self.arg)
        # pre model prepare
        self.model = model_select(self.arg.model)
        self.bobs_model = self.model
        self.alice_model = self.model
        self.reload_model()

    def __call__(self):
        for i in range(self.arg.epochs):
            self.train(epoch=i)

    def reload_model(self):
        """
        set all model
        Returns:
        """
        self.bobs_optimizer = optim.SGD(
            self.bobs_model.parameters(),
            self.arg.lr,
            momentum=self.arg.momentum,
            weight_decay=self.arg.weight_decay)
        self.alice_optimizer = optim.SGD(
            self.alice_model.parameters(),
            self.arg.lr,
            momentum=self.arg.momentum,
            weight_decay=self.arg.weight_decay)

        self.params = [
            list(
                self.bobs_model.parameters()), list(
                self.alice_model.parameters())]

    def train(self, epoch):
        for batch_idx, (data, target) in enumerate(self.train_loader):

            if batch_idx % self.arg.print_freq == 0:
                bob_loss, bob_prc = self.test(self.bobs_model)
                alice_loss, alice_prc = self.test(self.alice_model)
                print(
                    'Epoch: [{}/{}]\t'
                    'Loss_bob: ({:.3})\t'
                    'Loss_alice: ({:.3})\t'
                    'Prec_bob {top1.avg:.1f}%\t'
                    'Prec_alice {top2.avg:.1f}%'.format(
                        epoch,
                        self.arg.epochs,
                        bob_loss,
                        alice_loss,
                        top1=bob_prc,
                        top2=alice_prc))

            if batch_idx % 2:
                update(data, target, self.alice_model, self.alice_optimizer)

            else:
                update(data, target, self.bobs_model, self.bobs_optimizer)

            # encrypted aggregation
            new_params = list()
            # gradients clip
            clip_grad_norm_(self.bobs_model.parameters(), max_norm=20)
            clip_grad_norm_(self.alice_model.parameters(), max_norm=20)

            for param_i in range(len(self.params[0])):
                spd_params = list()
                '''
                   # 获取相关加密参数 待加密数据维度  安全参数，一般取1 随机数范围
                '''
                T = getRandomMatrix(len(self.params[0][param_i].flatten()), 1, 100)
                # 私钥
                size = self.params[0][param_i].shape
                # max_length = 0
                S = getSecretKey(T)
                # iterate all workers
                for index in range(2):

                    # aggregation same parameters from every workers. Then encrypt
                    # it into individual worker depends on trusted worker for all.
                    a = np.array(self.params[index][param_i].copy().fix_precision().tolist()).flatten()
                    # for value in a:
                    #     if "." in str(value):
                    #         _, diam = str(value).split(".")
                    #         if max_length < len(diam):
                    #             max_length = len(diam)
                    # 加密
                    c = encrypt(T, a)
                    spd_params.append(c)
                # decrypt parameter.
                new = (spd_params[0] + spd_params[1]) / 2
                new_params.append(new)
                # test for decrypt
                # 解密
                dc = decrypt(S, new)
                temp=[]
                for i in dc:
                    temp.append(i)
                tmp = torch.from_numpy(np.array(temp)).reshape(size)
                t = tmp.send(self.bob)
                self.params[index][param_i].set_(tmp)
                # tmp = torch.from_numpy(dc).float_precision()
            # clean up
            with torch.no_grad():
                # iterate all parameters
                for model in self.params:
                    for param in model:
                        param *= 0

                # set new parameters in all sub workers, bob and alice.
                for remote_index in range(2):
                    for param_index in range(len(self.params[remote_index])):
                        self.params[remote_index][param_index].set_(
                            new_params[param_index])

    def test(self, model):
        model.eval()
        test_loss = 0
        acc = AverageMeter()
        for data, target in self.test_loader:
            output = model(data)
            # sum up batch loss
            test_loss += F.cross_entropy(output,
                                         target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
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
