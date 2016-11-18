# ImageRecognition - Torch
Repository for Image Recognition Challenges
Torch implementation.

This implements training & test results of the most popular image classifying challenges, including cifar-10, cifar-100, imagenet, and kaggle cat vs dog challenge.

## Requirements
See the [installation instruction](installation.md) for a step-by-step guide.
- Install [Torch](http://torch.ch/docs/getting-started.html)
- Install [cuda-8.0](https://developer.nvidia.com/cuda-downloads)
- Install [cudnn v5](https://developer.nvidia.com/cudnn)
- Install 'optnet'
```bash
luarocks install optnet
```

## Environments
| GPUs       | numbers | nvidia-version | dev    |
|:----------:|:-------:|:--------------:|:------:|
| GTX 980 Ti | 1       | 367.57         | local  |
| GTX 1080   | 2       | 372.20         | server |

## Directories and datasets
- checkpoints : The optimal stages and models will be saved in this directory.
- datasets : data generation & preprocessing codes are contained.
- models : directory that contains resnet structure file.
- gen : directory where generated datasets are saved in.
- scripts : directory where scripts for each datasets are contained.

## How to run
You can run each dataset which could be either cifar10, cifar100, imagenet, catdog by running the script below.
```bash
./scripts/[cifar10/cifar100/imagenet/catdog].sh
```

## CIFAR-10 Results
Below is the result of the test set accuracy for CIFAR-10 dataset training.

| network           | preprocess | Optimizer| Memory | epoch | per epoch    | accuracy(%)     |
|:-----------------:|:----------:|----------|:------:|:-----:|:------------:|:---------------:|
| wide-resnet 28x10 |    None    | Momentum | 5.08G  | 200   | 2 min 59 sec | 95.16           |
| wide-resnet 28x10 |    PCA     | Momentum | 5.08G  | 200   | 2 min 59 sec | will be updated |
| wide-resnet 28x10 |    ZCA     | Momentum | 5.08G  | 200   | 2 min 59 sec | will be updated |

CIFAR-10 was updated with the following implementation details.

|   epoch   | learning rate |  weigth decay |
|:---------:|:-------------:|:-------------:|
|   0 ~ 80  |      0.1      |     0.0005    |
|  81 ~ 120 |      0.02     |     0.0005    |
| 121 ~ 160 |     0.004     |     0.0005    |
| 161 ~ 200 |     0.0008    |     0.0005    |

## CIFAR-100 Results
Below is the result of the test set accuracy for CIFAR-100 dataset training

| network           | preprocess | Optimizer| Memory | epoch | per epoch    | Top1 acc(%)| Top5 acc(%) |
|:-----------------:|:----------:|----------|:------:|:-----:|:------------:|:----------:|:-----------:|
| wide-resnet 28x10 |    None    | Momentum | 5.08G  | 200   | 2 min 59 sec |      -     |             |
| wide-resnet 28x10 |    PCA     | Momentum | 5.08G  | 200   | 2 min 59 sec |      -     |             |
| wide-resnet 28x10 |    ZCA     | Momentum | 5.08G  | 200   | 2 min 59 sec |      -     |             |


CIFAR-100 was updated with the following implementation details.

|   epoch   | learning rate |  weigth decay |
|:---------:|:-------------:|:-------------:|
|   0 ~ 80  |      0.1      |     0.0005    |
|  81 ~ 120 |      0.02     |     0.0005    |
| 121 ~ 160 |     0.004     |     0.0005    |
| 161 ~ 200 |     0.0008    |     0.0005    |

## Cat vs Dog
