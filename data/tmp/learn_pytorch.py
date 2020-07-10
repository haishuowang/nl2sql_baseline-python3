# import torchtext
# from torchtext.vocab import Vectors
import torch
import numpy as np
import random
import torch.nn.functional as F
import torch.nn as nn

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
torch.Tensor

class CNN_NET(torch.nn.Module):
    def __init__(self):
        super(CNN_NET, self).__init__()
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=60,  # input height  数字识别是灰白照片，只有一个channel
        #         out_channels=128,  # n_filters
        #         kernel_size=3,  # filter size
        #         stride=1,  # filter movement/step
        #         padding=1,
        #         # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
        #     ),
        #     nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1),
        #     # nn.ReLU(),
        #     # nn.AvgPool2d(kernel_size=2),
        # )
        # x = F.softmax(input=x, dim=1)

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=60, out_channels=64, kernel_size=3, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                                   nn.ReLU(),

                                   nn.MaxPool2d((2, 2)),
                                   nn.Dropout(0.2),

                                   nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                                   nn.ReLU(),

                                   nn.Dropout(0.3),

                                   nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d((2, 2)),
                                   nn.Dropout(0.5),
                                   nn.Softmax()
                                   )


net_test = CNN_NET()
#
test_data = torch.randn(200, 60, 12, 1)
out1 = net_test.conv1(test_data)
print(out1.shape)
