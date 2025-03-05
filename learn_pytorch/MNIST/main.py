import csv
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def load_train_data(file_path):
    df = pd.read_csv(file_path)
    data = df.iloc[:, 1:].values
    target = df.iloc[:, 0].values
    data = torch.tensor(data, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0
    target = torch.tensor(target, dtype=torch.long)
    return TensorDataset(data, target)

def load_test_data(file_path):
    df = pd.read_csv(file_path)
    data = df.iloc[:, :].values 
    data = torch.tensor(data, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0
    dummy_target = torch.zeros(len(data), dtype=torch.long)
    return TensorDataset(data, dummy_target)

def train(model, train_loader, device, optimizer, epoch):
    model.train()
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if idx % 10 == 0:
            print(f"Train Epoch: {epoch} [{idx * len(data)}/{len(train_loader.dataset)} ({100. * idx / len(train_loader):.0f}%)]\tLoss: {loss.item()}")

def test(model, device, test_loader):
    file_name = 'data/output.csv'
    headers = ['ImageId', 'Label']
    model.eval()
    predictions = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            predictions.extend(pred.cpu().numpy().flatten())
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for idx, label in enumerate(predictions, start=1):
            writer.writerow([idx, label])

def main():
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = load_train_data('data/train.csv')
    test_dataset = load_test_data('data/test.csv') 
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    model = Net().to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(1, 11):
        train(model, train_loader, device, optimizer, epoch)
        scheduler.step()
    test(model, device, test_loader)
    torch.save(model.state_dict(), "mnist_cnn.pth")

if __name__ == '__main__':
    main()