---
    Title: Experiments on CIFAR & MNIST datasets with gradient encryption.
---
<!-- TOC -->

- [0.1. Introduction](#01-introduction)
- [0.2. Requirements](#02-requirements)
    - [0.2.1. software](#021-software)
    - [0.2.2. Hardware](#022-hardware)
- [0.3. Usage](#03-usage)
    - [0.3.1. Clone this repository](#031-clone-this-repository)
    - [0.3.2. Edit syft_main.py relevant parameters of model](#032-edit-syft_mainpy-relevant-parameters-of-model)
    - [0.3.3. Train](#033-train)
    - [0.3.4. Extension (For fun)](#034-extension-for-fun)

<!-- /TOC -->

## 0.1. Introduction

The model is built for safe deep learning gradient encryption aggregation experiments on the cifar100 & mnist dataset. This realizes to use differential dataset training on the independent models then all parameters will be encrypted aggregation into specifical parameters container. Finally, the models upload parameters after gradients of each model be encrypted and aggregate together.

This structure could be built by using [PySyft framework](https://github.com/OpenMined/PySyft), so i also wrote [Extension (For fun)](#extension-for-fun) to realizes it to test this solution, but not working very well. Therefore, the main scheme is to manually implement gradient encryption through the homomorphic encryption algorithm. Algorithm in:

    utils/vhe.py

* **Hint: this algorithm consumes extremely huge memory space lead to memory overflow even in simply CNN model. vhe.py Must be improved to manage to optimize memory space, for example, point instead of variables.**

* **Brach sql_ store variables into SQL database to save the memory.**

Reimplement state-of-the-art CNN models in Cifar100 dataset with Pysyft, now including:

- For cifar100 dataset:

  1.[ResNet](https://arxiv.org/abs/1512.03385v1)

  2.[PreActResNet](https://arxiv.org/abs/1603.05027v3)

  3.[WideResNet](https://arxiv.org/abs/1605.07146v4)

  4.[ResNeXt](https://arxiv.org/abs/1611.05431v2)

  5.[DenseNet](https://arxiv.org/abs/1608.06993v4)

  6.[Alexnet](https://arxiv.org/ftp/arxiv/papers/1803/1803.01164.pdf)

- For Mnist dataset:

  7.[Lenet](https://arxiv.org/pdf/1909.12778.pdf)

Other results will be added later.

## 0.2. Requirements

### 0.2.1. software

Requirements for [PyTorch](http://pytorch.org/)

And others package must be included in your environment specified in requirements.txt

### 0.2.2. Hardware

For now only can run on CPU.

## 0.3. Usage

### 0.3.1. Clone this repository

```
git clone https://github.com/Gaopeng-Bai/Gradient-Secure.git
```

* In this project, the network structure is defined in the "models" folder, the script `secure_gradient.py` is running on two separates models, gradient aggregation though Syft virtual machine. Then upload new parameters after aggregation to each model.

- In `secure_gradient.py`, **Mnist** dataset can train on `lenet5`, `simply_cnn`, `simply_cnn2`. but `AlexNet` model testing with Tiny ImageNet or MNIST could not be done due to their smaller feature sizes (images do not fit the input size 227 x 227). Mnist on `lenet5` model tested on learning rate 0.01 and epoch 15 reached 99% accuracy, and on simply_cnn spent more time than `lenet5`.

- **cifar100** dataset can train on `resnet20`, `resnet32`, `resnet44`, `resnet110`, `preact_resnet110`, `resnet164`, `resnet1001`, `preact_resnet164`, `preact_resnet1001`,`wide_resnet`, `resneXt`, `densenet`.

### 0.3.2. Edit syft_main.py relevant parameters of model

- In the `secure_gradient.py`, you can specify the model you want to train(for example):

    ![avatar](images/models.png)

    Then, you need specify some parameter for training process, like epoche....

### 0.3.3. Train

- Two local worker with gradient aggregation

  ```
  python secure_gradient.py
  ```

- Normal Model

  ```
  python Cifar100_main.py

  python mnist_main.py
  ```

### 0.3.4. Extension (For fun)

* In the `syft_test.py`, it is a cifar100 secure gradient aggregation model modified completely according to [Syft tutorials 10](https://github.com/OpenMined/PySyft/blob/master/examples/tutorials/Part%2010%20-%20Federated%20Learning%20with%20Secure%20Aggregation.ipynb). However, it's not working. Because when the data send to virtual machine (bob and alice).

     ![avatar](images/data_pro.png)

* In the training, the model point the data position, input data to the current model point suddenly, it broken here.
 
  ![avatar](images/pro.png)

  Don't know the reason why this happen yet. It seem like some wrong in [PySyft framework](https://github.com/OpenMined/PySyft).
