#!/usr/bin/python3
# -*-coding:utf-8 -*-

# Reference:**********************************************
# @Time    : 4/7/2020 7:43 PM
# @Author  : Gaopeng.Bai
# @File    : dataloader.py
# @User    : gaope
# @Software: PyCharm
# @Description: 
# Reference:**********************************************
import torchvision
import torchvision.transforms as transforms
import torch


def data_loader(args):
    kwargs = {}
    print('=> loading cifar100 data...')
    normalize = transforms.Normalize(
        mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

    train_dataset = torchvision.datasets.CIFAR100(
        root='../data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    test_dataset = torchvision.datasets.CIFAR100(
        root='../data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=False, **kwargs)

    return trainloader, testloader
