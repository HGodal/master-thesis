import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Classifier, self).__init__()
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = self.conv_block(c_in=in_channels, c_out=128, dropout=0.3, kernel_size=5, stride=1, padding=2)
        self.block2 = self.conv_block(c_in=128, c_out=64, dropout=0.3, kernel_size=3, stride=1, padding=1)
        self.lastcnn = nn.Conv3d(in_channels=64, out_channels=num_classes, kernel_size=32, stride=1, padding=0)

    def conv_block(self, c_in, c_out, dropout, **kwargs):
        seq_block = nn.Sequential(
            nn.Conv3d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm3d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout3d(p=dropout),
        )
        return seq_block

    def forward(self, x):
        x = self.block1(x)
        x = self.maxpool(x)
        x = self.block2(x)
        x = self.maxpool(x)
        x = self.lastcnn(x)

        x = x.squeeze()

        return x if len(x.shape) == 2 else x.unsqueeze(0)


if __name__ == '__main__':
    size = 128
    model = Classifier(in_channels=1, num_classes=3)

    x = torch.randn((2, 1, size, size, size))
    preds = model(x)
    print(preds.shape)

