import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys
# Add repo root to Python path
sys.path.append(os.path.abspath('..'))
from src.classification_keypoint import AngleLSTMNet, AngleFeatureExtractor

# Load the data
train_data = pd.read_csv('../datasets/train_action_pose_keypoint.csv')
test_data = pd.read_csv('../datasets/test_action_pose_keypoint.csv')

# Debug: Print unique labels and their counts
print("Training data unique labels:")
print(train_data['label'].value_counts())
print("\nTest data unique labels:")
print(test_data['label'].value_counts())

# Debug: Check for whitespace and case issues
print("\nTraining data unique labels (with repr):")
for label in train_data['label'].unique():
    print(repr(label))
print("\nTest data unique labels (with repr):")
for label in test_data['label'].unique():
    print(repr(label))

# Clean the labels (remove leading/trailing whitespace and convert to lowercase)
train_data['label'] = train_data['label'].str.strip().str.lower()
test_data['label'] = test_data['label'].str.strip().str.lower()

# Get unique labels after cleaning
print("\nUnique labels after cleaning:")
print("Training:", train_data['label'].unique())
print("Testing:", test_data['label'].unique())

# Now proceed with label encoding
label_encoder = LabelEncoder()
train_data['label'] = label_encoder.fit_transform(train_data['label'])
test_data['label'] = label_encoder.transform(test_data['label'])

print("\nEncoded labels:")
print("Number of classes:", len(label_encoder.classes_))
print("Classes:", label_encoder.classes_)

import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.classification_keypoint import AngleLSTMNet, AngleFeatureExtractor
import numpy as np

# Initialize angle feature extractor
angle_extractor = AngleFeatureExtractor()

# Load the pre-split training and testing data
train_data = pd.read_csv('../datasets/train_action_pose_keypoint.csv')
test_data = pd.read_csv('../datasets/test_action_pose_keypoint.csv')

# Encode labels
label_encoder = LabelEncoder()
train_data['label'] = label_encoder.fit_transform(train_data['label'])
test_data['label'] = label_encoder.transform(test_data['label'])

# Extract angular features from keypoints
def extract_angular_features(data):
    keypoints = data.iloc[:, 2:].values  # Select only keypoint columns
    angles_list = []
    for keypoint_row in keypoints:
        angles = angle_extractor.calculate_angles(keypoint_row)
        angles_list.append(angles)
    return np.array(angles_list)

# Process training and testing data
X_train = extract_angular_features(train_data)
y_train = train_data['label'].values
X_test = extract_angular_features(test_data)
y_test = test_data['label'].values

# Reshape for LSTM (batch, sequence_length, features)
X_train = X_train.reshape(-1, 1, 8)  # 8 angular features
X_test = X_test.reshape(-1, 1, 8)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Hyperparameters
input_size = 8  # Number of angular features
hidden_size = 128  # Increased from 64
num_layers = 3  # Increased from 2
num_classes = len(label_encoder.classes_)
lstm_dropout = 0.3
fc_dropout = 0.5
weight_decay = 1e-5
learning_rate = 0.001
num_epochs = 150
patience = 10
save_path = '../model/trained-model.pt'
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model, loss, and optimizer
model = AngleLSTMNet(input_size=input_size, hidden_size=hidden_size,
                     num_layers=num_layers, num_classes=num_classes,
                     lstm_dropout=lstm_dropout, fc_dropout=fc_dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_model(model, train_loader, val_loader, num_epochs, patience, save_path):
    early_stopping = EarlyStopping(patience=patience)
    best_model_state = None
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {epoch_loss:.4f}, Training Acc: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        if early_stopping(val_loss):
            print("Early stopping triggered")
            break

    # Save the best model
    if best_model_state is not None:
        torch.save(best_model_state, save_path)
        print("Best model saved.")

# Train the model
train_model(model, train_loader, test_loader, num_epochs, patience, save_path)