import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from model_definitions import SkinDiagnosisModel 


app = Flask(__name__)

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

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'models/skin_diagnosis_model (1).pth'  
CLASS_NAMES = ['Light Diseases and Disorders of Pigmentation', 'Lupus and other Connective Tissue diseases', 'Bullous Disease', 'Systemic Disease', 'Eczema', 'Vascular Tumors', 'Urticaria Hives', 'Contact Dermatitis', 'Herpes HPV and other STDs', 'Atopic Dermatitis', 'Hair Loss Photos Alopecia and other Hair Diseases', 'Tinea Ringworm Candidiasis and other Fungal Infections', 'Acne or Rosacea', 'Psoriasis pictures Lichen Planus and related diseases', 'Melanoma Skin Cancer Nevi and Moles', 'Vasculitis', 'Nail Fungus and other Nail Disease', 'Scabies Lyme Disease and other Infestations and Bites', 'Exanthems and Drug Eruptions', 'Seborrheic Keratoses and other Benign Tumors', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 'Cellulitis Impetigo and other Bacterial Infections', 'Warts Molluscum and other Viral Infections'] #  actual classes
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load PyTorch model
model = SkinDiagnosisModel(num_classes=23).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("PyTorch model loaded successfully")

# Define image transformations

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    """Predict the class of an image using the loaded model"""
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        
    return preds.item(), probabilities[preds].item()


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get prediction
            predicted_idx, confidence = predict_image(filepath)
            class_name = CLASS_NAMES[predicted_idx]
            
            return render_template('index.html', 
                                filename=filename,
                                prediction=class_name,
                                confidence=round(confidence, 2))
    
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)