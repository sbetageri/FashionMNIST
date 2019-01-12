import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=64 * 4 * 4, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

        self.dropout = nn.Dropout(p=0.4)
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
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = x.view(x.size()[0], -1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.logsoftmax(x,)

        return x