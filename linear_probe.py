import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from mnist_exp import MLPHead, LinearHead
import argparse

def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--weight")
    return parser


test_transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root="./data", train=False, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

class LinearClassifier(nn.Module):
    def __init__(self, encoder, proj_dim=32, num_classes=10):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(proj_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():  # freeze encoder
            z = self.encoder(x)
        return self.classifier(z)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    weight = args.weight
    if "mlp" in weight:
        model = MLPHead()
        model.load_state_dict(torch.load(args.weight))
    else:
        model = LinearHead()
        model.load_state_dict(torch.load(args.weight))

    model.eval()  # freeze encoder
    linear_model = LinearClassifier(model).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(linear_model.classifier.parameters(), lr=1e-4)

    train_dataset = datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    for epoch in range(10):
        for x, y in train_loader:
            x, y = x.view(-1, 28*28).cuda(), y.cuda()
            logits = linear_model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")
    
    correct, total = 0, 0
    linear_model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.view(-1, 28*28).cuda(), y.cuda()
            preds = linear_model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    print(f"Test Accuracy: {correct / total:.2%}")
