from models import Discriminator
import torch
import torchvision
from torch import nn
from torchvision import transforms
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


net = Discriminator(1).to(device)
net.load_state_dict(torch.load(sys.argv[1]))

i = 0
for param in net.named_parameters():
    i += 1
print("total_param: "+str(i))

j = 0
for param in net.parameters():
    j += 1
    if j == i:
        break
    param.requires_grad_(False)

#for param in net.named_parameters():
#    print(param)

data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

data_dir = './transferredDataset/'

dataset = torchvision.datasets.ImageFolder(root = data_dir, transform = data_transform)
print(len(dataset))
dataset_splits = torch.utils.data.random_split(dataset, [len(dataset) - 2*int(0.2*len(dataset)), int(0.2*len(dataset)), int(0.2*len(dataset))])
train_dataset = dataset_splits[0]
valid_dataset = dataset_splits[1]
test_dataset = dataset_splits[2]

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

print(dataset.class_to_idx)

import torch.optim as optim

optimizer = optim.Adam(net.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion = nn.BCELoss()

epoch_num = 20

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs).view(-1)
            outputs = torch.round(outputs)
            diffs = torch.abs(labels - outputs)
            correct += len(diffs) - torch.sum(diffs)
            total += len(diffs)
    model.train()
    return float(correct/total)


nofinetune = check_accuracy(testloader, net)
print("Discriminator test set accuracy without finetuning: " + str(check_accuracy(testloader, net)))

print("Started finetuning.")

prev_valid_acc = 0
for epoch in range(epoch_num):  # loop over the dataset multiple times
    current_valid_acc = check_accuracy(validloader, net)
    print('Epoch: '+str(epoch)+ 'Train accuracy: ' + str(running_loss/count) +' Valid accuracy: '+str(current_valid_acc))
    if current_valid_acc > prev_valid_acc:
        best_finetuned_weight =  'finetuned_'+str(round(current_valid_acc, 5))+'.pth'
        torch.save(net.state_dict(), best_finetuned_weight)
        prev_valid_acc = current_valid_acc
    running_loss = 0.0
    count = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs).view(-1)
#        print(outputs)
#        print(labels)
        loss = criterion(outputs, labels.type(torch.FloatTensor).to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        count += 1

print('Finished finetuning')

net.load_state_dict(torch.load(best_finetuned_weight))

finetune = check_accuracy(testloader, net)
print('Discriminator test set accuracy after finetuning:' + str(check_accuracy(testloader, net)))

print('Starting from scratch')

from models import weights_init
net.apply(weights_init)
for param in net.parameters():
    param.requires_grad_(True)

optimizer = optim.Adam(net.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion = nn.BCELoss()

net.train()
prev_valid_acc = 0
for epoch in range(epoch_num):  # loop over the dataset multiple times
    current_valid_acc = check_accuracy(validloader, net)
    print('Epoch: '+str(epoch)+ 'Train accuracy: ' + str(running_loss/count) +' Valid accuracy: '+str(current_valid_acc))
    if current_valid_acc > prev_valid_acc:
        best_scratch_weight =  'scratch_'+str(round(current_valid_acc, 5))+'.pth'
        torch.save(net.state_dict(), best_scratch_weight)
        prev_valid_acc = current_valid_acc
    running_loss = 0.0
    count = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs).view(-1)
#        print(outputs)
#        print(labels)
        loss = criterion(outputs, labels.type(torch.FloatTensor).to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        count += 1

print("Finished training from scratch.")

net.load_state_dict(torch.load(best_scratch_weight))

print("Discriminator test set accuracy when trained from scratch: "+str(check_accuracy(testloader, net)))
print("Discriminator test set accuracy after finetuning : "+str(finetune))
print("Discriminator test set accuracy no finetuning: "+str(nofinetune))
