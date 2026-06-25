import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        return self.dropout(self.relu(self.bn(self.conv(x))))


class StressTransformer(nn.Module):
    def __init__(self, input_channels=4, num_classes=2, d_model=64,
                 num_heads=4, num_layers=2, dropout=0.2):
        super(StressTransformer, self).__init__()

        # CNN extracts local physiological patterns (peaks, waveform shape)
        self.cnn = nn.Sequential(
            ConvBlock(input_channels, 32, kernel_size=7, stride=2),
            ConvBlock(32, d_model, kernel_size=5, stride=2),
        )

        self.pos_dropout = nn.Dropout(p=dropout)

        # transformer captures temporal dependencies across the 10s window
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
        self.adaptivemaxpool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(d_model * 2, num_classes)

    def forward(self, x):
        # x: (batch, 4, 1000)
        x = self.cnn(x)                  # (batch, d_model, seq/4)
        x = self.pos_dropout(x)
        x = x.permute(0, 2, 1)          # (batch, seq/4, d_model)
        x = self.transformer(x)          # (batch, seq/4, d_model)
        x = x.permute(0, 2, 1)          # (batch, d_model, seq/4)
        x1 = self.adaptiveavgpool(x)     # (batch, d_model, 1)
        x2 = self.adaptivemaxpool(x)     # (batch, d_model, 1)
        x = torch.cat((x1, x2), dim=1)  # (batch, d_model*2, 1)
        x = x.view(x.size(0), -1)
        return self.fc(self.dropout(x))


class LightweightCNN(nn.Module):
    # smaller baseline — for comparison against StressTransformer
    def __init__(self, input_channels=4, num_classes=2, dropout=0.3):
        super(LightweightCNN, self).__init__()
        self.net = nn.Sequential(
            ConvBlock(input_channels, 32, kernel_size=7, stride=2),
            ConvBlock(32, 64, kernel_size=5, stride=2),
            ConvBlock(64, 64, kernel_size=3, stride=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def stress_transformer(**kwargs):
    return StressTransformer(**kwargs)


def lightweight_cnn(**kwargs):
    return LightweightCNN(**kwargs)