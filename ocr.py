import torch
from torchvision import datasets
import torchvision.transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import keyboard
import os.path

path1 = r"torch2.pt"
path2 = r"torch.pt"
path = path2
#Sets the device used to the GPU CUDA if possible, otherwise uses the CPU
device : torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

#EMNIST has an error where the images are transposed, so this reverts that
transformation=torchvision.transforms.Compose([
                    lambda img: torchvision.transforms.functional.rotate(img, -90),
                    lambda img: torchvision.transforms.functional.hflip(img),
                    ToTensor()])

#EMNIST is the dataset I will be using to train my CNN
#It contains 28x28 pixel images of 62 characters (10 numbers, 26 lowercase and 26 uppercase)
#Parts of the training data will be used in SGD
#The testing data will be used to check accuracy of the model
trainingData = datasets.EMNIST(
    root = 'data',
    split = 'byclass',
    train = True,                         
    transform = transformation, 
    download = True,            
)

testingData = datasets.EMNIST(
    root = 'data',
    split = 'byclass', 
    train = False, 
    transform = transformation
)

classes = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcderfhijklmnopqrstuvwxyz')

#Enables a batch of the dataset to be loaded
trainingBatch = 64
testingBatch = 25
trainingLoader = DataLoader(trainingData, 
                            batch_size=trainingBatch, 
                            shuffle=True, 
                            num_workers=0)
    
testingLoader = DataLoader(testingData, 
                           batch_size=testingBatch, 
                           shuffle=True, 
                           num_workers=0)

#Defines a convolutional neural network with 62 outputs
class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.fc1 = nn.Linear(32*7*7,62)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  
        x = self.fc1(x)
        return x

class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.fc1 = nn.Sequential(
            nn.Linear(32*7*7, 32*7),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(32*7,62)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
#Creates a neural network in the GPU, as well as an evaluation function and and optimiser
learningRate = 0.0005
network = CNN2()
network=network.to(device=device)
lossFunction = nn.CrossEntropyLoss()
optimiser = optim.RMSprop(network.parameters(), lr=learningRate)


def saveCheckpoint(state,filename=path):
    torch.save(state,filename,_use_new_zipfile_serialization=False)

def loadCheckpoint(checkpoint):
    network.load_state_dict(checkpoint['state_dict'])
    optimiser.load_state_dict(checkpoint['optimiser'])
    return checkpoint['epoch']
 
def train():
    if os.path.isfile(path):
        epochNumber = loadCheckpoint(torch.load(path))
    else:
        epochNumber = 0
    while True:
        totalLoss=0.0
        if keyboard.is_pressed('esc'): break
        for i, (data,targets) in enumerate(trainingLoader, 0):
            data=data.to(device=device)
            targets=targets.to(device=device)
            optimiser.zero_grad()
            out=network(data)
            loss=lossFunction(out, targets)
            loss.backward()
            optimiser.step()
            totalLoss += loss.item()
            if keyboard.is_pressed('esc'): break
            if i % 2000 == 1999:
                checkpoint={'state_dict':network.state_dict(),'optimiser':optimiser.state_dict(),'epoch':epochNumber}
                saveCheckpoint(checkpoint)
                print(f'[{epochNumber+1}, {i+1:5}] loss:{round(totalLoss/2000,5)}')
                totalLoss = 0.0
        epochNumber+=1
        if keyboard.is_pressed('esc'): break

similar = [
    ['s','S','5'],
    ['o','O','0'],
    ['1','l','I','L','J'],
    ['p','P'],
    ['q','9'],
    ['W','w'],
    ['C','c'],
    ['Z','z','2'],
    ['K','k','x','X'],
    ['F','f'],
    ['U','u'],
    ['Y','y']
]
def test():
    loadCheckpoint(torch.load(path))
    figure = plt.figure()
    print('Outputting:')
    dataiter = iter(testingLoader)
    images, labels = next(dataiter)
    cuda_images=images.to(device=device)
    outputs = network(cuda_images)
    _, predicted = torch.max(outputs, 1)
    print('Truth:    ', ' '.join('%s' % classes[labels[j]] for j in range(testingBatch)))
    print('Predicted:', ' '.join('%s' % classes[predicted[j]] for j in range(testingBatch)))
    print(sum([{True:1,False:0}[classes[labels[j]]==classes[predicted[j]]] for j in range(testingBatch)])/testingBatch)
    for j in range(testingBatch):
                figure.add_subplot(5,5,j+1)
                plt.title(f'{classes[predicted[j]]}-{classes[labels[j]]}')
                plt.axis("off")
                plt.imshow(images[j].squeeze(), cmap="gray")
    plt.show()
    


if __name__ == '__main__':
    #train()
    test()
    pass