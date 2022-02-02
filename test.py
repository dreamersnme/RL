import time

import torch.nn as nn
import torch

st = time.time()

x = torch.rand(2,5,2)
y = x.transpose(dim0=-2, dim1=-1)
# y = x.permute(0, 2, 1)
print('input_size:', x.shape)
print(x)
print('input_size:', y.shape)
print(y)










#
# conv1d = nn.Conv1d(3, 3, kernel_size=3)
# out = conv1d(x)
# print('output_size:',out.shape)
# print(out)