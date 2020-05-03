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

parser = argparse.ArgumentParser(description='PyTorch secure gradient Training')
parser.add_argument('--dataset', default="mnist", type=str, metavar='N', help='mnist or cifar100')
parser.add_argument('--model', default="lenet5", type=str, metavar='N',
                    help='choose a model to use mnist(lenet5, simply_cnn, simply_cnn2, alexnet) or for '
                         'cifar100 datasets(resnet20, resnet32, resnet44, resnet110 preact_resnet110, '
                         'resnet164, resnet1001, preact_resnet164, preact_resnet1001, wide_resnet, resneXt, densenet)')
parser.add_argument('--epochs', default=15, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--worker_iter', default=5, type=int, metavar='N',
                    help='worker iterations(times of training in specify worker)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4, lenet with mnist suggest:1e-2)')
parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 10)')
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
        # save the parameters shape to recover decrypted data.
        params_size = list()
        # save the exponential of value to convert float to int64
        max_length = list()
        # save encrypted private key to decrypt, each layer has specified key.
        Private_key = list()
        # gradients clip
        clip_grad_norm_(self.bobs_model.parameters(), max_norm=20)
        clip_grad_norm_(self.alice_model.parameters(), max_norm=20)

        for param_i in range(len(self.params[0])):
            spd_params = list()
            '''
                 from utils.vhe  Homomorphic encryption.

               # Obtain relevant encryption parameters Data dimension to be encrypted Security parameters,
                 generally take 1 random number range
            '''
            T = getRandomMatrix(len(self.params[0][param_i].flatten()), 1, 100)
            # private key generated.
            Private_key.append(getSecretKey(T))
            params_size.append(self.params[0][param_i].shape)
            # Calculate the number of decimal places.
            length = 0
            for value in np.array(self.params[0][param_i].tolist()).flatten():
                if "." in str(value):
                    _, diam = str(value).split(".")
                    if length < len(diam):
                        length = len(diam)
            max_length.append(length)

            # iterate all sub models.
            for index in range(2):
                # aggregation same parameters from every workers. Then encrypt
                # it into individual worker depends on trusted worker for all.
                parameters = np.array(
                    self.params[index][param_i].tolist()).flatten()
                # encrypt data that must be integer, so Zoom in from float to integer.
                # hint: int and int64 not same type.
                a = parameters * 10 ** (max_length[param_i] - 1)
                a_int = a.astype(int64)
                spd_params.append(encrypt(T, a_int))
            # Homomorphic encrypted sum operation.
            new_params.append((spd_params[0] + spd_params[1]))

        # clean up
        with torch.no_grad():
            # iterate all parameters
            for model in self.params:
                for param in model:
                    param *= 0
            # set new parameters in all sub workers, bob and alice.
            for remote_index in range(2):
                for param_index in range(len(self.params[remote_index])):
                    dc = decrypt(Private_key[param_index], new_params[param_index]).astype(
                        float) / 2 / 10 ** (max_length[param_index] - 1)
                    self.params[remote_index][param_index].data = torch.from_numpy(
                        np.array(dc).reshape(params_size[param_index]))

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
