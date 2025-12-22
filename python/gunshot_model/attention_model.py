"""
Attention-based neural network model for gunshot detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation attention block.
    
    Reference: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
    """
    
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.channels = channels
        self.reduction = reduction
        
        # Global average pooling (squeeze)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Excitation: two fully connected layers
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        
        self.activation = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch, channels, length)
        batch_size, channels, length = x.size()
        
        # Squeeze: global average pooling
        squeeze = self.global_pool(x).view(batch_size, channels)
        
        # Excitation
        excitation = self.fc1(squeeze)
        excitation = self.activation(excitation)
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation).view(batch_size, channels, 1)
        
        # Scale input
        return x * excitation.expand_as(x)


class ChannelAttention(nn.Module):
    """
    Channel Attention module.
    """
    
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size, channels, length = x.size()
        
        # Average pooling
        avg_out = self.fc(self.avg_pool(x).view(batch_size, channels))
        
        # Max pooling
        max_out = self.fc(self.max_pool(x).view(batch_size, channels))
        
        # Combine
        out = avg_out + max_out
        out = self.sigmoid(out).view(batch_size, channels, 1)
        
        return x * out.expand_as(x)


class SpatialAttention(nn.Module):
    """
    Spatial Attention module.
    """
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, 
                             padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch, channels, length)
        
        # Average along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        
        # Max along channel dimension
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate
        combined = torch.cat([avg_out, max_out], dim=1)
        
        # Conv layer
        attention = self.conv(combined)
        attention = self.sigmoid(attention)
        
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Combines Channel Attention and Spatial Attention.
    
    Reference: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
    """
    
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        # Apply channel attention
        x = self.channel_attention(x)
        
        # Apply spatial attention
        x = self.spatial_attention(x)
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head Self-Attention for 1D sequences.
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "Embed dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, length, embed_dim)
        batch_size, length, embed_dim = x.size()
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, length, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(batch_size, length, embed_dim)
        out = self.proj(out)
        
        return out


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise separable convolution for efficient computation.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv1d, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MobileNetBlock1D(nn.Module):
    """
    MobileNetV2 inverted residual block for 1D signals.
    """
    
    def __init__(self, in_channels, out_channels, stride, expansion_factor=6):
        super(MobileNetBlock1D, self).__init__()
        hidden_channels = in_channels * expansion_factor
        
        # Expansion
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.relu1 = nn.ReLU6(inplace=True)
        
        # Depthwise convolution
        self.conv2 = nn.Conv1d(
            hidden_channels, hidden_channels, 3,
            stride=stride, padding=1, groups=hidden_channels, bias=False
        )
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.relu2 = nn.ReLU6(inplace=True)
        
        # Projection
        self.conv3 = nn.Conv1d(hidden_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
    def forward(self, x):
        identity = x
        
        # Expansion
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        # Depthwise convolution
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        
        # Projection
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Skip connection
        if self.use_residual:
            out = out + identity
            
        return out


class AudioMobileNet1D(nn.Module):
    """
    AudioMobileNet1D: Efficient neural network for audio classification.
    Combines MobileNet blocks with attention mechanisms.
    """
    
    def __init__(self, num_classes=2, input_channels=1, dropout_rate=0.3):
        super(AudioMobileNet1D, self).__init__()
        
        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # MobileNet blocks
        self.block1 = MobileNetBlock1D(32, 64, stride=2, expansion_factor=2)
        self.attention1 = SqueezeExcitation(64, reduction=16)
        
        self.block2 = MobileNetBlock1D(64, 128, stride=2, expansion_factor=4)
        self.attention2 = CBAM(128, reduction=16)
        
        self.block3 = MobileNetBlock1D(128, 256, stride=2, expansion_factor=6)
        self.attention3 = SqueezeExcitation(256, reduction=16)
        
        self.block4 = MobileNetBlock1D(256, 512, stride=2, expansion_factor=6)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Self-attention layer
        self.self_attention = MultiHeadSelfAttention(embed_dim=512, num_heads=8)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, x):
        # x shape: (batch, channels, length)
        
        # Stem
        x = self.stem(x)
        
        # Block 1 with attention
        x = self.block1(x)
        x = self.attention1(x)
        
        # Block 2 with attention
        x = self.block2(x)
        x = self.attention2(x)
        
        # Block 3 with attention
        x = self.block3(x)
        x = self.attention3(x)
        
        # Block 4
        x = self.block4(x)
        
        # Global pooling
        x = self.global_pool(x)  # (batch, 512, 1)
        
        # Reshape for self-attention
        x = x.transpose(1, 2)  # (batch, 1, 512)
        
        # Self-attention
        x = self.self_attention(x)
        
        # Flatten
        x = x.mean(dim=1)  # (batch, 512)
        
        # Classifier
        x = self.classifier(x)
        
        return x
        
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class LightweightGunshotDetector(nn.Module):
    """
    Lightweight model for FPGA deployment.
    Optimized for low memory and computation.
    """
    
    def __init__(self, num_classes=2, input_channels=1):
        super(LightweightGunshotDetector, self).__init__()
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            # Block 1
            nn.Conv1d(input_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            
            # Block 2
            DepthwiseSeparableConv1d(16, 32, kernel_size=3, stride=2, padding=1),
            SqueezeExcitation(32, reduction=8),
            
            # Block 3
            DepthwiseSeparableConv1d(32, 64, kernel_size=3, stride=2, padding=1),
            
            # Block 4
            DepthwiseSeparableConv1d(64, 128, kernel_size=3, stride=2, padding=1),
            SqueezeExcitation(128, reduction=16),
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(32, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, x):
        # Feature extraction
        x = self.feature_extractor(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x
        
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def get_num_params(self):
        """Get number of parameters."""
        return sum(p.numel() for p in self.parameters())


class TemporalAttentionModel(nn.Module):
    """
    Temporal Attention Model for gunshot detection.
    Focuses on temporal patterns in audio signals.
    """
    
    def __init__(self, num_classes=2, input_channels=1, hidden_size=128, num_layers=2):
        super(TemporalAttentionModel, self).__init__()
        
        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # CNN feature extraction
        cnn_features = self.cnn(x)  # (batch, 128, length/16)
        
        # Reshape for LSTM
        cnn_features = cnn_features.transpose(1, 2)  # (batch, length/16, 128)
        
        # LSTM
        lstm_out, _ = self.lstm(cnn_features)  # (batch, length/16, hidden_size*2)
        
        # Attention weights
        attention_weights = self.attention(lstm_out)  # (batch, length/16, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(lstm_out * attention_weights, dim=1)  # (batch, hidden_size*2)
        
        # Classification
        output = self.classifier(context)
        
        return output


def test_models():
    """Test all model architectures."""
    print("Testing model architectures...")
    
    # Test input
    batch_size = 4
    input_channels = 1
    length = 16000  # 1 second at 16kHz
    
    # Create dummy input
    x = torch.randn(batch_size, input_channels, length)
    print(f"Input shape: {x.shape}")
    
    # Test AudioMobileNet1D
    print("\n1. AudioMobileNet1D:")
    model = AudioMobileNet1D(num_classes=2)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    
    with torch.no_grad():
        output = model(x)
        print(f"  Output shape: {output.shape}")
        print(f"  Output sample: {output[0]}")
    
    # Test LightweightGunshotDetector
    print("\n2. LightweightGunshotDetector:")
    model = LightweightGunshotDetector(num_classes=2)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    
    with torch.no_grad():
        output = model(x)
        print(f"  Output shape: {output.shape}")
        print(f"  Output sample: {output[0]}")
    
    # Test TemporalAttentionModel
    print("\n3. TemporalAttentionModel:")
    model = TemporalAttentionModel(num_classes=2)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    
    with torch.no_grad():
        output = model(x)
        print(f"  Output shape: {output.shape}")
        print(f"  Output sample: {output[0]}")
    
    # Test individual attention modules
    print("\n4. Attention Modules:")
    
    # Test SqueezeExcitation
    se = SqueezeExcitation(channels=64)
    test_input = torch.randn(batch_size, 64, length // 16)
    output = se(test_input)
    print(f"  SqueezeExcitation: {test_input.shape} -> {output.shape}")
    
    # Test CBAM
    cbam = CBAM(channels=128)
    test_input = torch.randn(batch_size, 128, length // 32)
    output = cbam(test_input)
    print(f"  CBAM: {test_input.shape} -> {output.shape}")
    
    print("\nAll models tested successfully!")


def create_fpga_compatible_model():
    """
    Create a model compatible with FPGA deployment.
    Uses fixed-point friendly operations and reduced precision.
    """
    model = LightweightGunshotDetector(num_classes=2)
    
    # Convert to quantization-aware training model
    model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
    model = torch.ao.quantization.prepare_qat(model)
    
    return model


if __name__ == "__main__":
    test_models()
