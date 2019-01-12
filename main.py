import torch
import torch.nn as nn
import torchvision
import MyModel
import Utils

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

EPOCHS = 10

transforms = torchvision.transforms.ToTensor()

trainDataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms)
testDataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms)

trainLoader = DataLoader(dataset=trainDataset, batch_size=4)
testLoader = DataLoader(dataset=testDataset, batch_size=4)

criterion = nn.NLLLoss()
device = Utils.getDevice()

model = MyModel.MyModel()

model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for i in range(EPOCHS):

    running_acc = 0
    running_loss = 0
    model.train()
    for img, label in tqdm(trainLoader):
        img = img.to(device)
        label = label.to(device)
    
        optimizer.zero_grad()
        pred = model(img)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
    
        pred = torch.topk(pred, 1)[1].view(1, pred.size()[0])
        running_loss += loss.item()
        running_acc += torch.sum(pred == label).item()
        
    train_loss = running_loss / len(trainLoader)
    train_acc = running_acc / len(trainLoader)
    
    print('Training Loss : ', train_loss)
    print('Training Acc : ', train_acc)
    
    test_loss = 0
    test_acc = 0
    
    model.eval()
    for img, label in tqdm(testLoader):
        img = img.to(device)
        label = label.to(device)
    
        pred = model(img)
        loss = criterion(pred, label)
    
        pred = torch.topk(pred, 1)[1].view(1, pred.size()[0])
        test_loss += loss.item()
        test_acc += torch.sum(pred == label).item()
    
    test_loss = test_loss / len(testLoader)
    test_acc = test_acc / len(testLoader)
    
    print('Test Loss : ', test_loss)
    print('Train Acc : ', test_acc)