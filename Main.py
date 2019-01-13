import torch
import torch.nn as nn
import torchvision
import MyModel
import Utils

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

EPOCHS = 72
BATCH_SIZE = 32
MODEL_LOC = './model/'
MODEL_NAME = 'fashion_2.pt'

transforms = torchvision.transforms.ToTensor()

trainDataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms)
testDataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms)

trainLoader = DataLoader(dataset=trainDataset, batch_size=BATCH_SIZE)
testLoader = DataLoader(dataset=testDataset, batch_size=BATCH_SIZE)

criterion = nn.NLLLoss()
device = Utils.getDevice()

model = MyModel.MyModel()

model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

prev_acc = 0

for i in range(EPOCHS):
    print('\n\n\n\nEpoch : ', i)
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
        
    train_loss = running_loss / (len(trainLoader) * BATCH_SIZE)
    train_acc = running_acc / (len(trainLoader) * BATCH_SIZE)
    
    print('Training Loss : ', train_loss)
    print('Training Acc : ', train_acc * 100.0)
    
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
    
    test_loss = test_loss / (len(testLoader) * BATCH_SIZE)
    test_acc = test_acc / (len(testLoader) * BATCH_SIZE)

    print('Test Loss : ', test_loss)
    print('Test Acc : ', test_acc * 100.0)

    if prev_acc < test_acc or i == (EPOCHS - 1):
        prev_acc = test_acc
        torch.save(model.state_dict(), MODEL_LOC + MODEL_NAME)
    else:
        print('Best model is from epoch ', i - 1)
        break