import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import syft as sy

epochs = 2
DOWNLOAD_MNIST = False

class Arguments():
    def __init__(self):
        self.batch_size = 50
        self.test_batch_size = 1000
        self.epochs = epochs
        self.lr = 0.001
        self.momentum = 0.5
        self.no_cuda = True
        self.seed = 1
        self.log_interval = 200
        self.save_model = False


args = Arguments()

use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# Initialize workers
hook = sy.TorchHook(torch)  # <-- NEW: hook PyTorch ie add extra functionalities to support Federated Learning
bob = sy.VirtualWorker(hook, id="bob")  # <-- NEW: define remote worker bob
alice = sy.VirtualWorker(hook, id="alice")  # <-- NEW: and alice


# Load and distribute dataset
def load_data():
    '''<--Load CIFAR dataset from torch vision module distribute to workers using PySyft's Federated Data loader'''

    federated_train_loader = sy.FederatedDataLoader(  # <-- this is now a FederatedDataLoader
        torchvision.datasets.MNIST('./mnist/',
                                   train=True,
                                   download=DOWNLOAD_MNIST,
                                   transform=torchvision.transforms.ToTensor(),)
            .federate((bob, alice)),
        # <-- NEW: we distribute the dataset across all the workers, it's now a FederatedDataset
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./mnist/',
                                   train=False,
                                   transform=torchvision.transforms.ToTensor()),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    return federated_train_loader, test_loader


# Define Neural Network
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.lr = nn.Sequential(
            nn.Linear(28*28, 10)
        )

    def forward(self, x):
        output = self.lr(x)
        return output, x


# Train function

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(federated_train_loader):  # <-- now it is a distributed dataset
        data = data.view(-1, 28*28)
        model.send(data.location)  # <-- NEW: send the model to the right location
        data, target = data.to(device), target.to(device)
        output = model(data)[0]
        loss = loss_func(output, target)
        optimizer.zero_grad()  #
        loss.backward()
        optimizer.step()
        model.get()  # <-- NEW: get the model back
        if batch_idx % args.log_interval == 0:
            loss = loss.get()  # <-- NEW: get the loss back
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader) * args.batch_size,
                # batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


# Test function
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target).item()  # sum up batch loss
            pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# Train neural network
# <--Load federated training data and test data
federated_train_loader, test_loader = load_data()

# <--Create Neural Network model instance
model = LogisticRegression().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)  # <--TODO momentum is not supported at the moment
loss_func = nn.CrossEntropyLoss()

# <--Train Neural network and validate with test set after completion of training every epoch
for epoch in range(1, args.epochs + 1):
    train(args, model, device, federated_train_loader, optimizer, epoch)
    test(args, model, device, test_loader)

if (args.save_model):
    torch.save(model.state_dict(), "minst_lr.pt")
