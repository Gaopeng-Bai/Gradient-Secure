#!/usr/bin/python3
# -*-coding:utf-8 -*-

# Reference:**********************************************
# @Time    : 5/3/2020 2:34 PM
# @Author  : Gaopeng.Bai
# @File    : simply_cnn2.py
# @User    : gaope
# @Software: PyCharm
# @Description: 
# Reference:**********************************************
import torch.nn as nn


class simply_cnn2(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.cnn2 = nn.Conv2d(16, 16, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(2)

        self.cnn3 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.cnn4 = nn.Conv2d(32, 32, 3, padding=1)
        self.relu4 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(2)

        self.fc = nn.Linear(32 * 7 * 7, 10)
        # self.sm=nn.Softmax(dim=1)

    def forward(self, x):
        h = self.cnn1(x)
        h = self.relu1(h)
        h = self.cnn2(h)
        h = self.relu2(h)
        h = self.max_pool1(h)

        h = self.cnn3(h)
        h = self.relu3(h)
        h = self.cnn4(h)
        h = self.relu4(h)
        h = self.max_pool2(h)

        h = h.view(h.size(0), -1)
        h = self.fc(h)
        # h=self.sm(h)
        return h