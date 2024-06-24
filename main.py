import torch
 
import torchvision.transforms as transforms
from torchvision import models, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
data_path = 'dataset/'

transform = transforms.Compose(
    [
     transforms.RandomResizedCrop(224),
     transforms.RandomHorizontalFlip(),
     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
     transforms.Grayscale(),
     transforms.ToTensor(),
     transforms.Normalize([0.485], [0.229])
     ])

val_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

batch_size = 32

classes = ('fractured', 'not fractured')

trainset = datasets.ImageFolder(root=data_path+"train/", transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)

testset = datasets.ImageFolder(root=data_path+"test/", transform=val_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

valset = datasets.ImageFolder(root=data_path+"val/", transform=val_transform)
valloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)


# functions to show an image
def imshow(img, title = None):
    img = img     # unnormalize
    npimg = img.numpy()
    print(npimg)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title != None:
        plt.title(title)
    plt.show()

from PIL import Image, ImageFile
from PIL import UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True

def pil_loader(path):
    for _ in range(5):  # Tentativas para carregar a imagem
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('L')
        except (OSError, UnidentifiedImageError) as e:
            print(f"Warning: {e} - {path}. Retrying...")
    # Se falhar em todas as tentativas, retornar uma imagem preta (ou você pode escolher outra abordagem)
    return Image.new('L', (224, 224))

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same')
        self.fc1 = nn.Linear(in_features=32 * 56 * 56, out_features=128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

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
    
def imshow2(images, labels, class_names, fontsize=10, ncol=7, nrow=5):
    num_images = len(images)
    assert num_images <= ncol * nrow, "Número de imagens excede a capacidade da grade."
    
    fig, axes = plt.subplots(nrow + 1, ncol, figsize=(15, 15))  # +1 para títulos
    axes = axes.flatten()  # Achata a grade de eixos para fácil iteração
    
    for col in range(ncol):
        if col < len(labels):
            axes[col].set_title(class_names[labels[col]], fontsize=fontsize)
        axes[col].axis('off')  # Desativa os eixos para os títulos
    
    for i, (img, label, ax) in enumerate(zip(images, labels, axes[ncol:])):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        ax.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
        ax.axis('off')
    
    # Remove os eixos excedentes se houver menos imagens do que subplots
    for ax in axes[num_images+ncol:]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    class_names = trainset.classes
    imshow2(images, labels, class_names, fontsize=8)
    print(images.shape)
    print(labels.shape)
 
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    net = net.to(device)
    
    print("chegamos até aqui")
    num_epochs = 20
    for epoch in range(num_epochs):  
        net.train()
        train_loss = 0
        train_correct = 0
        for i, data in enumerate(trainloader, start=0):
            # obter as entradas; data é uma lista de [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(trainloader)
        train_accuracy = train_correct / len(trainloader.dataset)

        net.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for images, labels in valloader:
                try:
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    continue

        val_loss /= len(valloader)
        val_accuracy = val_correct / len(valloader.dataset)
        
        scheduler.step()

        print(f'{epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

    print('Finished Training')
    # PATH = '/savetest'
    # torch.save(net.state_dict(), PATH)
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # print images
    imshow(utils.make_grid(images))
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