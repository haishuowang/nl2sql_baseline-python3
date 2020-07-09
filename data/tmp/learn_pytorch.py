import torchtext
from torchtext.vocab import Vectors
import torch
import numpy as np
import random

USE_CUDA = torch.cuda.is_available()

# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)
if USE_CUDA:
    torch.cuda.manual_seed(53113)

BATCH_SIZE = 32
EMBEDDING_SIZE = 650
MAX_VOCAB_SIZE = 50000

in_channels = 3
class CNN_NET(torch.nn.Module):
    def __init__(self):
            super(CNN_NET,self).__init__()
            self.conv1 = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,              # input height  数字识别是灰白照片，只有一个channel
                        out_channels=16,            # n_filters
                        kernel_size=3,              # filter size
                        stride=3,                   # filter movement/step
                        padding=0,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
                    ),
                # nn.ReLU(),
                # nn.AvgPool2d(kernel_size=2),
            )
import torch.nn as nn
net_test = CNN_NET()
#
test_data = torch.randn(3,in_channels,9,9)
out1 = net_test.conv1(test_data)
print(out1.shape)
