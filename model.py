
import torch.nn as nn
from torch import flatten
import torch
import torch.nn.functional as F
from torchvision import models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device="cpu"
class TextAttentionCNN(nn.Module):
    def __init__(self,multitask=False):
        super(TextAttentionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 9)
        self.conv2 = nn.Conv2d(32, 96, 7)
        self.pool1 = nn.MaxPool2d(3,stride=3)
        self.conv3 = nn.Conv2d(96, 128, 5)
        
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cl0 = nn.Linear(1024, 2)
        self.cl1 = nn.Linear(1024, 11)
        self.multitask=multitask

    def forward(self, x):
        
        x = F.relu(self.conv1(x.float()))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 512)
        x = self.fc1(x)
        x = self.fc2(x)
        y0 = self.cl0(x)
        y1 = self.cl1(x)
        if self.multitask:
            return y1,y0
        return y1

def step_multitask(inputs,labels,net,optimizer,train=True):
    labels= labels.squeeze(1)
    labels_binary = torch.tensor(labels>0, dtype=torch.long,device=labels.device)
    optimizer.zero_grad()
    
    outputs_multiclass,outputs_binary= net(inputs)
    loss = nn.CrossEntropyLoss()(outputs_multiclass, labels) + nn.CrossEntropyLoss()(outputs_binary, labels_binary)
    if train:
        loss.backward()
        optimizer.step()
    acc=(labels==outputs_multiclass.max(1)[1]).sum().cpu().numpy().astype('float')/labels.shape[0]
    
    del labels,labels_binary,outputs_binary,outputs_multiclass
    return loss,acc

def step_single(inputs,labels,net,optimizer,train=True):
    labels= labels.squeeze(1)
    if train:
        optimizer.zero_grad()
    
    results= net(inputs)
    
    loss = nn.CrossEntropyLoss()(results, labels)
    if train:
        loss.backward()
        optimizer.step()

    acc=(labels==results.max(1)[1]).sum().cpu().numpy().astype('float')/labels.shape[0]
    del labels,results

    return loss,acc

class  FineTuneVGG(nn.Module):
    def __init__(self,freeze=True):
        super(FineTuneVGG, self).__init__()
        
        self.model = models.vgg16_bn(pretrained=True)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        #change outputsize and class number
        self.model.avgpool=nn.AdaptiveAvgPool2d(output_size=(1, 1))

        for i,layer in enumerate(self.model.classifier):
            if isinstance(layer,nn.Linear):
                self.model.classifier[i]=nn.Linear(in_features=layer.in_features//49,out_features=layer.out_features//49)
        self.model.classifier[6]=nn.Linear(83,11)
        
    def forward(self, x):
        x= x.float()
        y = self.model(x)
        del x
        return y

