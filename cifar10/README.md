# CIFAR-10 Dataset Implementation Details
Specific Description for CIFAR-10 Dataset training.

CIFAR-10 is consisted with 50,000 training images and 10,000 testing images.
Each images is consisted in an RGB format with the size of 32 x 32 pixels.

## Results
|      network      | Optimizer          | epoch | per epoch | accuracy(%)     |
|:-----------------:|--------------------|:-----:|:---------:|:---------------:|
|       vggnet      | Momentum Optimizer |  200  |     -     | 93.21           |
|      resnet200    | Momentum Optimizer |  200  |     -     | will be updated |
| wide-resnet 28x10 | Momentum Optimizer |  200  |   6m 43s  | will be updated |

![alt text](../result/cifar10_result.png "CIFAR-10 Test Results")

## Implementation Details
You can see implementation details [here](../notebook/Implementation.md)
