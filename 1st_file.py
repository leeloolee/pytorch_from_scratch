# Transpose A matrix in Pytorch
import torch

## https://www.aiworkbox.com/lessons/construct-a-pytorch-tensor


pt_matrix_ex = torch.Tensor(
    [
        [1, 2, 3],
        [0, 0, 0],
        [4, 5, 6]
    ]
)

pt_matrix_ex.T

pt_random_matrix_ex = torch.rand(3, 3, 3)

torch.__version__

## Change Tensor Type
x = torch.rand(3, 3, 3)
print(x)
type(x)
type(x.float())
type(x.double())
type(x.int())

## concat
x = torch.rand(3, 2, 2)
y = torch.rand(3, 2, 2)

torch.cat((x, y), 0).shape  # 어떤 dimension으로 concate하는지

##

x_2 = torch.rand(20, 3, 3)
print(x_2)

## elementwise multiplication
random_tensor_one_ex = (torch.rand(2, 3, 4) * 10).int()
random_tensor_two_ex = (torch.rand(2, 3, 4) * 10).int()
random_tensor_one_ex * random_tensor_two_ex

## import torch
import torch
import torchvision
import torchvision.datasets as datasets

cifar_trainset = datasets.CIFAR10(root= "./data", train=True,
                 transform=None, target_transform=None,
                 download=True)

cifar_testset = datasets.CIFAR10(root= "./data", train=False,
                 transform=None, target_transform=None,
                 download=False)