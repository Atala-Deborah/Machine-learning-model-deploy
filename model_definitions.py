import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, W, H = x.size()
        proj_query = self.query_conv(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, W * H)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(B, -1, W * H)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)
        return self.gamma * out + x

class CapsuleLayer(nn.Module):
    def __init__(self, input_dim, num_capsules, capsule_dim, output_dim=128, image_size=32, kernel_size=9, stride=2):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.capsules = nn.ModuleList([
            nn.Conv2d(input_dim, capsule_dim, kernel_size=kernel_size, stride=stride, padding=0)
            for _ in range(num_capsules)
        ])
        self.output_spatial_size = (image_size - kernel_size) // stride + 1
        self.fc = nn.Linear(num_capsules * capsule_dim * self.output_spatial_size * self.output_spatial_size, output_dim)
        self.output_dim = output_dim  # store the output dim
        self.num_capsules = num_capsules  # store num of capsules
        self.capsule_dim = capsule_dim  # store capsule dim
        self.output_spatial_size = self.output_spatial_size  # store the output spatial size

    def forward(self, x):
        capsule_outputs = [F.relu(capsule(x)) for capsule in self.capsules] # Added ReLU
        x = torch.stack(capsule_outputs, dim=1)  # Shape: (B, num_capsules, capsule_dim, H_out, W_out)
        x = x.view(x.size(0), -1)  # Flatten the features (excluding batch size)
        features = self.fc(x)  # Fully connected layer
        return features

class SkinDiagnosisModel(nn.Module):
    def __init__(self, num_classes=23):
        super(SkinDiagnosisModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # Batch normalization
        self.attention = SelfAttention(64)
        self.capsule_layer = CapsuleLayer(64, num_capsules=8, capsule_dim=16, image_size=32, kernel_size=9, stride=2)  # Pass image_size, kernel_size, stride
        self.dropout = nn.Dropout(0.5) # Added dropout
        self.classifier = nn.Linear(128, num_classes)  # Classifier layer for final output

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # Added batch norm
        x = self.attention(x)
        features = self.capsule_layer(x)
        features = self.dropout(features)
        logits = self.classifier(features)
        if self.training:
            return features, logits  # Return both during training
        else:
            return logits  