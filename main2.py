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
     transforms.Resize((224, 224)),
     transforms.ColorJitter(brightness=0.2, contrast=0.2,saturation=0.2, hue=0.2),  
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

val_transform = transforms.Compose([
 
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

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding='same') # MUDEI AQUI
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same')# MUDEI AQUI
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same')# MUDEI AQUI
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # MUDEI AQUI
        self.fc1 = nn.Linear(in_features=128 * 28 * 28, out_features=256) # MUDEI AQUI
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.5) # MUDEI AQUI

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x) # MUDEI AQUI

        x = torch.flatten(x, 1) # flatten all dimensions except batch

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x) # MUDEI AQUI

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x) # MUDEI AQUI
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
    
   
    for ax in axes[num_images+ncol:]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    class_names = trainset.classes
    imshow2(images, labels, class_names, fontsize=8)
    
    net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 
    net = net.to(device)
    
  
    num_epochs = 20
    net.train()

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for i, data in enumerate(trainloader, start=0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()
            if i % 100 == 99:
                print(f'Epoch {epoch + 1}/{num_epochs}, Batch {i + 1}/{len(trainloader)}, '
                  f'Training Loss: {running_loss / 100}, Training Acc: {100 * correct_train / total_train}%')

        training_accuracy = correct_train / total_train

        # Validation
        net.eval()
        correct_val = 0
        total_val = 0
        val_running_loss = 0.0

        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                if -1 in labels:
                    continue  # Skip the batch if it contains dummy labels

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

                _, predicted_val = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted_val == labels).sum().item()

            # Calculate validation accuracy after the epoch
        validation_accuracy = correct_val / total_val
        average_val_loss = val_running_loss / len(valloader)

        print(f'Epoch {epoch + 1}/{num_epochs}, '
            f'Trn Loss: {running_loss / len(trainloader)}, '
            f'Trn Acc: {100 * training_accuracy}%, '
            f'Vall Loss: {average_val_loss}, '
            f'Val Acc: {100 * validation_accuracy}%')
        scheduler.step()

    print('Finished Training')
    # PATH = '/savetest'
    # torch.save(net.state_dict(), PATH)
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # print images
    # imshow(utils.make_grid(images))
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