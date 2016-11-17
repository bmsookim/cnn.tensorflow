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
