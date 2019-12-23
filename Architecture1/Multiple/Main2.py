import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import Model as model
from torch.autograd import Variable
import time
from Logger import Logger
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
best_acc = 0
start_epoch = 0
logger = Logger('./logs2')
Train_Loss = []
Test_Loss = []
Train_Accuracy = []
Test_Accuracy = []

# Dataset
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
trainset = torchvision.datasets.CIFAR10(root='/export/livia/data/xxu/CIFAR10', train=True, download=False, transform=transform_train)
# /home/AN96120/brycexu/CIFAR10
# /export/livia/data/xxu/CIFAR10
trainset, valset = torch.utils.data.random_split(trainset, [40000, 10000])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
valloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='/export/livia/data/xxu/CIFAR10', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

# Model
print('==> Building model..')
model = model.MutipleBNN()
model = nn.DataParallel(model)
model = model.to(device)
criterion = nn.CrossEntropyLoss(reduction='mean')
optimizer1 = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
optimizer2 = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999), weight_decay=0.0003)


def update_lr(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.1


# Training
def train(epoch):
    global Train_Loss, Train_Accuracy
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for name, param in model.named_parameters():
        if name == 'module.convolutions.0.alpha' or name == 'module.convolutions.1.alpha' \
                or name == 'module.convolutions.2.alpha' or name == 'module.convolutions.3.alpha' \
                or name == 'module.convolutions.4.alpha' or name == 'module.convolutions.5.alpha' \
                or name == 'module.convolutions.6.alpha' or name == 'module.convolutions.7.alpha' \
                or name == 'module.convolutions.8.alpha' or name == 'module.convolutions.9.alpha':
            param.requires_grad = False
        else:
            param.requires_grad = True
    start = time.time()
    for batch_index, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = Variable(inputs)
        targets = Variable(targets)
        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # Backward and Optimize
        optimizer1.zero_grad()
        loss.backward()
        for p in list(model.parameters()):
            if hasattr(p, 'org'):
                p.data.copy_(p.org)
        optimizer1.step()
        for p in list(model.parameters()):
            if hasattr(p, 'org'):
                p.org.copy_(p.data.clamp_(-1, 1))
        # Results
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    train_loss = train_loss / (40000 / 128)
    end = time.time()
    print('Training Time: %.1f' % (end - start))
    print('Loss: %.3f | Accuracy: %.3f' % (train_loss, 100. * correct / total))
    # Plot the model
    info = {'train_loss': train_loss}
    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch + 1)
    Train_Loss.append(train_loss)
    Train_Accuracy.append(100. * correct / total)
    # Update lr
    if epoch == 200 or epoch == 400 or epoch == 600:
        update_lr(optimizer1)


def val(epoch):
    model.train()
    val_loss = 0
    correct = 0
    total = 0
    for name, param in model.named_parameters():
        if name == 'module.convolutions.0.alpha' or name == 'module.convolutions.1.alpha' \
                or name == 'module.convolutions.2.alpha' or name == 'module.convolutions.3.alpha' \
                or name == 'module.convolutions.4.alpha' or name == 'module.convolutions.5.alpha' \
                or name == 'module.convolutions.6.alpha' or name == 'module.convolutions.7.alpha' \
                or name == 'module.convolutions.8.alpha' or name == 'module.convolutions.9.alpha':
            param.requires_grad = True
        else:
            param.requires_grad = False
    start = time.time()
    for batch_index, (inputs, targets) in enumerate(valloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = Variable(inputs)
        targets = Variable(targets)
        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # Backward and Optimize
        optimizer2.zero_grad()
        loss.backward()
        for p in list(model.parameters()):
            if hasattr(p, 'org'):
                p.data.copy_(p.org)
        optimizer2.step()
        # Results
        val_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    end = time.time()
    print('Validating Time: %.1f' % (end - start))
    print('Accuracy: %.3f' % (100. * correct / total))


def test(epoch):
    global best_acc, Test_Loss, Test_Accuracy
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = Variable(inputs)
            targets = Variable(targets)
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Results
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    test_loss = test_loss / (10000 / 100)
    end = time.time()
    print('Testing Time: %1.f' % (end - start))
    print('Loss: %.3f | Accuracy: %.3f' % (test_loss, 100. * correct / total))

    # Save the model
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc

    # Plot the model
    info = {'test_loss': test_loss, 'test_accuracy': acc}
    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch + 1)
    Test_Loss.append(test_loss)
    Test_Accuracy.append(100. * correct / total)


epochs = 800
for epoch in range(start_epoch, start_epoch + epochs):
    train(epoch)
    val(epoch)
    test(epoch)

x1 = np.arange(0, epochs)
y10 = Train_Loss
y11 = Test_Loss
x2 = np.arange(0, epochs)
y20 = Train_Accuracy
y21 = Test_Accuracy
plt.subplot(2, 1, 1)
plt.title('Arch1_Multiple')
plt.plot(x1, y10, 'o-', color='b', label='Train_Loss')
plt.plot(x1, y11, 'o-', color='g', label='Test_Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(x2, y20, 'o-', color='k', label='Train_Acc')
plt.plot(x2, y21, 'o-', color='r', label='Test_Acc')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.tight_layout()
plt.savefig("Result2.jpg")
plt.show()
