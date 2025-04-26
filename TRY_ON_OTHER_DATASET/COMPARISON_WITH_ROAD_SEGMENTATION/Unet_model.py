import torch 
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU()
            )
        
        self.encoder = nn.ModuleList([
            conv_block(3, 16),
            conv_block(16, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256)
        ])
        
        self.pool = nn.MaxPool2d(2)
        
        self.upconv = nn.ModuleList([
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        ])
        
        self.decoder = nn.ModuleList([
            conv_block(256, 128),
            conv_block(128, 64),
            conv_block(64, 32),
            conv_block(32, 16)
        ])
        
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        enc_outs = []
        for enc in self.encoder:
            x = enc(x)
            enc_outs.append(x)
            x = self.pool(x)
        
        for i in range(4):
            x = self.upconv[i](x)
            enc_out = enc_outs[-(i+2)]
            if x.shape[2:] != enc_out.shape[2:]:
                x = nn.functional.interpolate(x, size=enc_out.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, enc_out], dim=1)
            x = self.decoder[i](x)
        
        return torch.sigmoid(self.final_conv(x))
    
if __name__ == "__main__":
    model = UNet()
    print(type(model))