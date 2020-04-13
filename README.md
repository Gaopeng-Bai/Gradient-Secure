# Experiments on CIFAR datasets with Syft model

## Introduction

The model is built for safe deep learning gradient encryption aggregation experiments on the cifar100 dataset by using [PySyft framework](https://github.com/OpenMined/PySyft). This realizes to use differential dataset training on the independent models then all parameters will be encrypted aggregation into specifical parameters container. Finally, the models upload parameters after gradients of each model be encrypted and aggregate together.

Reimplement state-of-the-art CNN models in Cifar100 dataset with Pysyft, now including:

1.[ResNet](https://arxiv.org/abs/1512.03385v1)

2.[PreActResNet](https://arxiv.org/abs/1603.05027v3)

3.[WideResNet](https://arxiv.org/abs/1605.07146v4)

4.[ResNeXt](https://arxiv.org/abs/1611.05431v2)

5.[DenseNet](https://arxiv.org/abs/1608.06993v4)

other results will be added later.

## Requirements:software

Requirements for [PyTorch](http://pytorch.org/)

And others package must be included in your environment specified in requirements.txt

## Requirements:hardware

For now only can run on CPU.

## Usage

1. Clone this repository

```
git clone https://github.com/Gaopeng-Bai/pytorch-syft-cifar-model.git
```

In this project, the network structure is defined in the "models" folder, the script ```syft_main.py``` is running on two separates models, gradient aggregation though Syft virtual machine. Then upload new parameters after aggregation to each model.

2. Edit syft_main.py relevant parameters of model

In the ```syft_main.py```, you can specify the model you want to train(for example):
![avatar](images/models.png)

Then, you need specify some parameter for training process, like epoche....

3. Train

* Two local worker with gradient aggregation

    ```
    python syft_main.py
    ```

* Normal Model

    ```
    python Normal_main.py
    ```


4. Extension (For fun)

In the  ```encrypt_sub_machine.py```, it is a cifar100 secure gradient aggregation model modified completely according to [Syft tutorials 10](https://github.com/OpenMined/PySyft/blob/master/examples/tutorials/Part%2010%20-%20Federated%20Learning%20with%20Secure%20Aggregation.ipynb).   However, it's not working. Because when the data send to virtual machine (bob and alice). 
![avatar](images/data_pro.png)
In the training, the model point the data position, input data to the current model point suddenly, it broken here. 
![avatar](images/pro.png)

Don't know the reason why this happen yet. It seem like some wrong in [PySyft framework](https://github.com/OpenMined/PySyft).  
