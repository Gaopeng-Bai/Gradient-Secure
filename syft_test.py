#!/usr/bin/python3
# -*-coding:utf-8 -*-

# Reference:**********************************************
# @Time    : 4/9/2020 4:50 PM
# @Author  : Gaopeng.Bai
# @File    : syft_test.py
# @User    : gaope
# @Software: PyCharm
# @Description: 
# Reference:**********************************************

import argparse
import syft as sy
from syft.frameworks.torch.fl import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.dataloader import data_loader
from utils.test import data_test
from utils.model import model_select

parser = argparse.ArgumentParser(description='PyTorch syft test depends on syft tutorial-10 Training')
parser.add_argument('--dataset', default="mnist", type=str,
                    metavar='N', help='mnist or cifar100')
parser.add_argument('--model', default="lenet5", type=str,
                    metavar='N', help='choose a model to use mnist(lenet5, ) or for cifar100 datasets(resnet20, resnet32, resnet44, resnet110'
                                      'preact_resnet110, resnet164, resnet1001, preact_resnet164, preact_resnet1001'
                                      'wide_resnet, resneXt, densenet)')
parser.add_argument('--epochs', default=200, type=int,
                    metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1,
                    type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4,
                    type=float, metavar='W', help='weight decay (default: 1e-4)')
args = parser.parse_args()


def train_process(data, target, model, optimizer):
    model.send(data.location)
    optimizer.zero_grad()
    pred = model(data)
    loss = F.cross_entropy(pred, target)
    loss.backward()
    optimizer.step()
    return model


class syft_model:
    def __init__(self, arg):
        self.arg = arg
        hook = sy.TorchHook(torch)
        #  connect to two remote workers that be call alice and bob and request
        #  another worker called the crypto_provider who gives all the crypto primitives we may need
        self.bob = sy.VirtualWorker(hook, id="bob")
        self.alice = sy.VirtualWorker(hook, id="alice")
        self.secure_worker = sy.VirtualWorker(hook, id="secure_worker")

        self.compute_nodes = [self.bob, self.alice]
        # load data
        self.train_loader, self.test_loader = data_test(self.arg)
        self.data_to_workers()
        # pre model prepare
        self.model = model_select(self.arg.model)
        self.reload_model()

    def __call__(self):
        for i in range(self.arg.epochs):
            self.train()

    def reload_model(self):
        """
        set all model
        Returns:

        """
        bobs_model = self.model
        alice_model = self.model
        bobs_optimizer = optim.SGD(bobs_model.parameters(), self.arg.lr,
                                   momentum=self.arg.momentum, weight_decay=self.arg.weight_decay)
        alice_optimizer = optim.SGD(alice_model.parameters(), self.arg.lr,
                                    momentum=self.arg.momentum, weight_decay=self.arg.weight_decay)
        self.models = [bobs_model, alice_model]
        # self.params = [list(bobs_model.parameters()), list(alice_model.parameters())]
        self.optimizers = [bobs_optimizer, alice_optimizer]

    def data_to_workers(self):
        # split training data into two workers bob and alice
        self.remote_dataset = (list(), list())

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.send(self.compute_nodes[batch_idx % len(self.compute_nodes)])
            target = target.send(self.compute_nodes[batch_idx % len(self.compute_nodes)])
            self.remote_dataset[batch_idx % len(self.compute_nodes)].append((data, target))

    def train(self):
        for data_index in range(len(self.remote_dataset[0]) - 1):
            # update remote models
            for remote_index in range(len(self.compute_nodes)):
                data, target = self.remote_dataset[remote_index][data_index]
                self.models[remote_index] = train_process(data, target, self.models[remote_index], self.optimizers[remote_index])

            for model in self.models:
                model.get()
            return utils.federated_avg({
                "bob": self.models[0],
                "alice": self.models[1]
            })
            # # encrypted aggregation
            # new_params = list()
            # for param_i in range(len(self.params[0])):
            #     spd_params = list()
            #     # iterate all workers
            #     for index in range(len(self.compute_nodes)):
            #         # aggregation same parameters from every workers. Then encrypt it into individual worker depends on trusted worker for all.
            #         spd_params.append(self.params[index][param_i].copy().fix_precision().share(self.bob, self.alice,
            #                                                                                    crypto_provider=self.secure_worker))
            #     # decrypt parameter.
            #     new = (spd_params[0] + spd_params[1]).get().float_precision() / 2
            #     new_params.append(new)
            # # clean up
            # with torch.no_grad():
            #     # iterate all parameters
            #     for model in self.params:
            #         for param in model:
            #             param *= 0
            #     for model in self.models:
            #         model.get()
            #     # set new parameters in all sub workers, bob and alice.
            #     for remote_index in range(len(self.compute_nodes)):
            #         for param_index in range(len(self.params[remote_index])):
            #             self.params[remote_index][param_index].set_(new_params[param_index])

    def test(self, model):
        model.eval()
        test_loss = 0
        for data, target in self.test_loader:
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability

        test_loss /= len(self.test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
        return test_loss


if __name__ == '__main__':
    a = syft_model(args)
    a()
