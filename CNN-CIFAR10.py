#Imports
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


#Device 
device=("mps" if torch.mps.is_available() else "cpu")


#Hyper-Parameters
num_epochs=4
batch_size=4
learning_rate=0.001



#Dataset has PILimages from range [0,1]
#We transform them to tensors of normalized range [-1,1]


transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


#Loading dataset
train_data=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
test_data=torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)

train_loader=DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_data,batch_size=batch_size,shuffle=False)

classes=('plane','car','bird','cat',
        'deer','dog','frog','horse','ship','truck')

#Implement convNet

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)



    def forward(self,x):
        y=self.pool(F.relu(self.conv1(x)))
        y=self.pool(F.relu(self.conv2(y)))
        #Flatten it
        y=y.view(-1,16*5*5)
        y=F.relu(self.fc1(y))
        y=F.relu(self.fc2(y))
        y=self.fc3(y)
        return y


model=CNN().to(device)

criterion=nn.CrossEntropyLoss().to(device)
optimizer=optim.SGD(model.parameters(),lr=learning_rate)

n_total_steps=len(train_loader)
for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_loader):
        #Original shape: [4,3,32,32] = 4,3,1024
        # input_layer= 3 input channels,6 output channel, 5 kernel size
        images=images.to(device)
        labels=labels.to(device)

        # Forward prop
        output=model(images)
        loss=criterion(output,labels) 

        # Backprop and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #if (i+1)%2000==0:
            #print(f'Epoch : [{epoch+1}/{num_epochs}], Step : [{i+1}/{n_total_steps}], Loss : {loss.item():.4f}')

print("Finished training")

with torch.no_grad():
    n_correct=0
    n_samples=0
    n_class_correct=[0 for i in range(10)]
    n_class_samples=[0 for i in range(10)]
    for images,labels in train_loader:
        images=images.to(device)
        labels=labels.to(device)
        output=model(images)

        #max returns (value,index)
        _,predicted=torch.max(output,1)
        n_samples += labels.size(0)
        n_correct += (predicted==labels).sum().item()

        for i in range(batch_size):
            label=labels[i]
            pred=predicted[i]
            if (label==pred):
                n_class_correct[label]+=1
            n_class_samples[label]+=1
    
    acc=100 * n_correct/n_samples
    print(f'Accuracy : {acc:.4f} %')

    for i in range(10):
        acc= 100 * n_class_correct[i]/n_class_samples[i]
        print(f'Accuracy of {classes[i]} : {acc:.4f} %')

