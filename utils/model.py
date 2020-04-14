#!/usr/bin/python3
# -*-coding:utf-8 -*-

# Reference:**********************************************
# @Time    : 4/7/2020 7:49 PM
# @Author  : Gaopeng.Bai
# @File    : model.py
# @User    : gaope
# @Software: PyCharm
# @Description: 
# Reference:**********************************************
from models import *
from models.lenet import lenet
from models.simply_cnn import simply_cnn

from models.alexnet import AlextNet


def model_select(arg):
    """
    choose a model to use in our training tasks
    Args:
        arg: the model name

    Returns: pytorch model

    """
    if arg == "resnet20":
        model = resnet20_cifar(num_classes=100)
    elif arg == "resnet32":
        model = resnet32_cifar(num_classes=100)
    elif arg == "resnet44":
        model = resnet44_cifar(num_classes=100)
    elif arg == "resnet110":
        model = resnet110_cifar(num_classes=100)
    elif arg == "preact_resnet110":
        model = preact_resnet110_cifar(num_classes=100)
    elif arg == "resnet164":
        model = resnet164_cifar(num_classes=100)
    elif arg == "resnet1001":
        model = resnet1001_cifar(num_classes=100)
    elif arg == "preact_resnet164":
        model = preact_resnet164_cifar(num_classes=100)
    elif arg == "preact_resnet1001":
        model = preact_resnet1001_cifar(num_classes=100)
    elif arg == "wide_resnet":
        model = wide_resnet_cifar(depth=26, width=10, num_classes=100)
    elif arg == "resneXt":
        model = resneXt_cifar(depth=29, cardinality=16, baseWidth=64, num_classes=100)
    elif arg == "densenet":
        model = densenet_BC_cifar(depth=190, k=40, num_classes=100)
    elif arg == "lenet5":
        model = lenet()
    elif arg == "alexnet":
        model = AlextNet(in_channel=1, n_class=10)
    elif arg == "simply_cnn":
        model = simply_cnn()

    return model






