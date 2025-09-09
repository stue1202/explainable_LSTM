import torch
import torch.nn as nn
import torch.nn.functional as F
from EfficientKAN import KAN

# --- Step 2: ChannelAttention Module Implementation ---
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        # AdaptiveAvgPool1d compresses each channel's spatial dimension into a single value
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Two fully connected layers to learn the channel weights
        self.fc1 = nn.Linear(in_channels, in_channels // ratio)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [batch_size, in_channels, seq_len]
        
        # Global average pooling on each channel
        avg_out = self.avg_pool(x).squeeze(-1)  # [batch_size, in_channels]
        
        # Pass the pooled output through the FC layers
        avg_out = self.fc1(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.fc2(avg_out)
        
        # Use a sigmoid function to scale the weights to [0, 1]
        scale = self.sigmoid(avg_out).unsqueeze(-1)  # [batch_size, in_channels, 1]
        
        # Multiply the input feature map by the learned channel-wise weights
        return x * scale

# --- Step 3: HybridModel Implementation ---
class HybridModel(nn.Module):
    def __init__(self, in_channels, num_classes, kan_in_features, kan_out_features=2):
        super(HybridModel, self).__init__()
        
        # 1D-CNN Module: Local feature extraction
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        
        # Channel Attention Module
        self.channel_attention = ChannelAttention(in_channels=128)
        
        # KAN Module: For learning relationships between features
        self.kan_layer = KAN(in_features=kan_in_features, out_features=kan_out_features)
        
        # Final Classifier Layer
        self.classifier = nn.Linear(kan_out_features, num_classes)

    def forward(self, x):
        # x shape: [batch_size, in_channels, seq_len]
        
        # 1. Pass through CNN layers to extract features
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        # x shape is now [batch_size, 128, new_seq_len]
        
        # 2. Apply channel attention to weight the feature channels
        x = self.channel_attention(x)
        
        # 3. Flatten the feature map to prepare for KAN
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # 4. Pass the flattened features to the KAN layer
        x = self.kan_layer(x)
        
        # 5. Final classification
        output = self.classifier(x)
        
        return output

# --- Example Usage ---
if __name__ == '__main__':
    # Define the dimensions for your data
    # Example: 16 ECG samples, 1 channel each, with a sequence length of 256
    batch_size = 16
    in_channels = 1
    seq_len = 256
    num_classes = 2 # e.g., 'Normal' vs. 'Abnormal'
    
    # Calculate the required input features for the KAN layer
    # After two max pooling layers with kernel_size=2, the sequence length becomes:
    # 256 -> 128 -> 64
    kan_in_features = 128 * 64
    
    # Instantiate the hybrid model
    model = HybridModel(in_channels, num_classes, kan_in_features)
    
    # Create a dummy input tensor for demonstration
    dummy_input = torch.randn(batch_size, in_channels, seq_len)
    
    # Pass the dummy data through the model
    output = model(dummy_input)
    
    print("Model output shape:", output.shape)
    print("Expected output shape: [batch_size, num_classes]")
    print(f"In this example: [{batch_size}, {num_classes}]")