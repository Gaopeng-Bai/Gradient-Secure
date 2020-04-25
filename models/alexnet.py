#!/usr/bin/python3
# -*-coding:utf-8 -*-

# Reference:**********************************************
# @Time    : 4/13/2020 11:26 PM
# @Author  : Gaopeng.Bai
# @File    : alexnet.py
# @User    : gaope
# @Software: PyCharm
# @Description: 
# Reference:**********************************************
import torch
import torch.nn as nn


class AlextNet(nn.Module):
    def __init__(self, in_channel, n_class):
        super(AlextNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(1*1*256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, n_class) # AlexNet上面是1000 ...如果测试的话用MNIST则可以使用10
        )

    def forward(self, x):
        con1_x = self.conv1(x)
        con2_x = self.conv2(con1_x)
        con3_x = self.conv3(con2_x)
        lin_x = con3_x.view(con3_x.size(0), -1)
        y_hat = self.fc(lin_x)
        return y_hat