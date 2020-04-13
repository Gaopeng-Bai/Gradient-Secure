#!/usr/bin/python3
# -*-coding:utf-8 -*-

# Reference:**********************************************
# @Time    : 4/13/2020 11:01 PM
# @Author  : Gaopeng.Bai
# @File    : mnist_main.py
# @User    : gaope
# @Software: PyCharm
# @Description: 
# Reference:**********************************************
import numpy as np
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from utils.model import model_select

parser = argparse.ArgumentParser(description='PyTorch mnist Training')
parser.add_argument('--model', default="lenet5", type=str,
                    metavar='N', help='lenet5, ')
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


def main(args):
    normalize = transforms.Normalize(
        mean=[0.131], std=[0.308])
    train_dataset = mnist.MNIST(root='../data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))
    test_dataset = mnist.MNIST(root='../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    model = model_select(args.model)
    sgd = SGD(model.parameters(), arg.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    cross_error = CrossEntropyLoss()

    for _epoch in range(args.epoch):
        for idx, (train_x, train_label) in enumerate(train_loader):
            label_np = np.zeros((train_label.shape[0], 10))
            sgd.zero_grad()
            predict_y = model(train_x.float())
            _error = cross_error(predict_y, train_label.long())
            if idx % 10 == 0:
                print('idx: {}, _error: {}'.format(idx, _error))
            _error.backward()
            sgd.step()

        correct = 0
        _sum = 0

        for idx, (test_x, test_label) in enumerate(test_loader):
            predict_y = model(test_x.float()).detach()
            predict_ys = np.argmax(predict_y, axis=-1)
            label_np = test_label.numpy()
            _ = predict_ys == test_label
            correct += np.sum(_.numpy(), axis=-1)
            _sum += _.shape[0]

        print('accuracy: {:.2f}'.format(correct / _sum))


if __name__ == '__main__':
    main(args)
