import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import ResNet18 # Import from ResNet/model.py

# Modify the ResNet18 model for MNIST
class ResNet18_MNIST(ResNet18):
    def __init__(self, num_classes=10):
        super(ResNet18_MNIST, self).__init__(num_classes=num_classes)
        # MNIST images are grayscale (1 channel), so we modify the first convolutional layer
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

def train_and_evaluate():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loading and transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

    # Model, loss function, optimizer
    model = ResNet18_MNIST(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f'Accuracy of the model on the 10000 test images: {100 * correct / total}%')

if __name__ == '__main__':
    train_and_evaluate()