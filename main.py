import torch
import torchvision
import torchvision.transforms as transforms

data_path = 'dataset/'

transform = transforms.Compose(
    [
     transforms.RandomResizedCrop(224),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 32

classes = ('bone', 'fractured')

trainset = torchvision.datasets.ImageFolder(root=data_path+"train/", transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

testset = torchvision.datasets.ImageFolder(root=data_path+"test/", transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

import matplotlib.pyplot as plt
import numpy as np
 
# functions to show an image
def imshow(img):
    img = img     # unnormalize
    npimg = img.numpy()
    print(npimg)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

from PIL import Image, ImageFile
from PIL import UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True

def pil_loader(path):
    for _ in range(5):  # Tentativas para carregar a imagem
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')
        except (OSError, UnidentifiedImageError) as e:
            print(f"Warning: {e} - {path}. Retrying...")
    # Se falhar em todas as tentativas, retornar uma imagem preta (ou você pode escolher outra abordagem)
    return Image.new('RGB', (224, 224))

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same')
        self.fc1 = nn.Linear(in_features=32 * 56 * 56, out_features=128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = torch.flatten(x, 1) # flatten all dimensions except batch

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x


import torch.optim as optim



if __name__ == '__main__':

    # dataiter = iter(trainloader)
    # images, labels = next(dataiter)

    # print(images.shape)
    # print(labels.shape)
    # images, labels = next(dataiter)

    # imshow(torchvision.utils.make_grid(images))
    # print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
    
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    net.to(device)
    net.train()

    print("chegamos até aqui")
    for epoch in range(40):  # loop no dataset multiplas vezes

        running_loss = 0.0
        for i, data in enumerate(trainloader, start=0):
            # obter as entradas; data é uma lista de [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zerar o gradiente dos parâmetros
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # imprimindo estatísticas
            running_loss += loss.item()
            if (i+1) % 500 == 0:    # print a cada 2000
                print(f'[Época: {epoch + 1}, iter: {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    # PATH = '/savetest'
    # torch.save(net.state_dict(), PATH)
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    images, labels = images.to(device), labels.to(device)
    outputs = net(images)
    print(outputs)

    for i in range(4):
        preds = outputs[i]
        cp = torch.argmax(preds)
        cgt = labels[i]
        print(f'Classe predita (idx): {cp}, Classe gt (idx): {cgt}')
        print(f'Classe predita: {classes[cp]}, Classe gt: {classes[cgt]}\n\n')

    correct = 0
    total = 0
    # Como não estamos treinando, não é necessário calcular os gradientes
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculando as saidas da CNN
            outputs = net(images)
            # classe com maior score
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Acurácia da CNN: {100 * correct // total} %')


    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)

            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accurácia para classe: {classname:5s} é {accuracy:.1f} %')