import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DeepNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class FuzzyLayer(nn.Module):
    def __init__(self, input_dim, num_rules):
        super(FuzzyLayer, self).__init__()
        self.num_rules = num_rules
        self.mu = nn.Parameter(torch.rand(input_dim, num_rules)) 
        self.sigma = nn.Parameter(torch.ones(input_dim, num_rules)) 

    def forward(self, x):
        x = x.unsqueeze(2) 
        mu = self.mu.unsqueeze(0)
        sigma = self.sigma.unsqueeze(0)
        out = torch.exp(-torch.pow(x - mu, 2) / (2 * torch.pow(sigma, 2) + 1e-5))
        out = torch.prod(out, dim=1) 
        return out

class DynamicFuzzyNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DynamicFuzzyNet, self).__init__()
        self.fuzzy = FuzzyLayer(input_dim, 32) 
        self.consequent = nn.Linear(32, num_classes)

    def forward(self, x):
        rule_firing = self.fuzzy(x)
        output = self.consequent(rule_firing)
        return output
