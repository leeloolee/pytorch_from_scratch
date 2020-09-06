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

pt_random_matrix_ex = torch.rand(3,3,3)

torch.__version__

## Change Tensor Type
x = torch.rand(3,3,3)
print(x)
type(x)
type(x.float())
type(x.double())
type(x.int())

##
