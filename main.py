 
from torchvision.models import resnet50, ResNet50_Weights 
from torch import device, cuda, max, no_grad, argmax
from torch.nn import Linear, CrossEntropyLoss
from torchvision.transforms import Compose, Resize, ColorJitter, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from check_dataset import *
 
transform = Compose(
    [
     Resize((224, 224)),
     ColorJitter(brightness=0.2, contrast=0.2,saturation=0.2, hue=0.2),  
     ToTensor(),
     Normalize((0.5,), (0.5,))])

val_transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize((0.5,), (0.5,))
])

batch_size = 32

trainset = ImageFolder(root=mainpath+"train/", transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)

testset = ImageFolder(root=mainpath+"test/", transform=val_transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

valset = ImageFolder(root=mainpath+"val/", transform=val_transform)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4)

classes = trainset.classes
if __name__ == '__main__':
    print("PrediçÃO de Fraturas Osseas")
    print("Classes dividas em:", classes)
    print("QTD Imagens em:")
    print("- Treino:", len(trainset))
    print("- Teste:", len(testset))
    print("- Val:", len(valset))

    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    class_names = trainset.classes
    imshow2(images, labels, class_names, fontsize=8)
    
    net = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_ftrs = net.fc.in_features
    net.fc = Linear(num_ftrs, 2)

    criterion = CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    use_device = device('cuda:0' if cuda.is_available() else 'cpu')
    net = net.to(use_device)
  
    num_epochs = 12
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    predictions = []
    targets = []
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for i, data in enumerate(trainloader, start=0):
            inputs, labels = data[0].to(use_device), data[1].to(use_device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted_train = max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()
            if i % 50 == 99:
                print(f'Epoch {epoch}/{num_epochs}, Batch {i + 1}/{len(trainloader)}, '
                  f'Training Loss: {running_loss / 100}, Training Acc: {100 * correct_train / total_train}%')

        training_accuracy = correct_train / total_train

        # Validation
        net.eval()
        correct_val = 0
        total_val = 0
        val_running_loss = 0.0

        with no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(use_device), labels.to(use_device)
                
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

                _, predicted_val = max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted_val == labels).sum().item()

        # Calculate validation accuracy after the epoch
        validation_accuracy = correct_val / total_val
        average_val_loss = val_running_loss / len(valloader)

        print(f'Epoch {epoch}/{num_epochs} | ' f'Trn Loss: {running_loss / len(trainloader)} | ' f'Trn Acc: {100 * training_accuracy}% | '
            f'Vall Loss: {average_val_loss} | '
            f'Val Acc: {100 * validation_accuracy}%')
        
        train_losses.append(running_loss / len(trainloader))
        train_accuracies.append(training_accuracy)
        val_losses.append(average_val_loss)
        val_accuracies.append(validation_accuracy)
        scheduler.step()

    print('Finished Training')
 
    dataiter = iter(testloader)
    images, labels = next(dataiter)

 
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    images, labels = images.to(use_device), labels.to(use_device)
    outputs = net(images)
    print(outputs)

    for i in range(4):
        preds = outputs[i]
        cp = argmax(preds)
        cgt = labels[i]
        print(f'Classe predita (idx): {cp}, Classe gt (idx): {cgt}')
        print(f'Classe predita: {classes[cp]}, Classe gt: {classes[cgt]}\n\n')

    correct = 0
    total = 0

    with no_grad():
        for data in testloader:
            images, labels = data[0].to(use_device), data[1].to(use_device)
            outputs = net(images)
            _, predicted = max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Acurácia da CNN: {100 * correct // total} %')
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with no_grad():
        for data in testloader:
            images, labels = data[0].to(use_device), data[1].to(use_device)
            outputs = net(images)
            _, predictions = max(outputs, 1)

            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    for name, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[name]
        print(f'Accurácia para classe: {name:5s} é {accuracy:.1f} %')