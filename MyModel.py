import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.checkingFlag = 0

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=3)
        
        self.fc1 = nn.Linear(in_features=50 * 5 * 5, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=10)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        '''Forward pass of the Neural Network
        
        :param x: Image Batch
        :type x: Tensor
        :return: Predictions
        :rtype: Label
        '''

        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if self.checkingFlag == 1:
            print(x.size())
            assert False

        x = x.view(x.size()[0], -1)
        
        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.logsoftmax(x,)

        return x