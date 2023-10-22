import torch.nn as nn
import torch
from eegnet import Conv2dWithConstraint, ConvTranspose2dWithConstraint

class EEGNetEncoder(nn.Module):
    def __init__(self,
                 chunk_size: int = 151,
                 num_electrodes: int = 60,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 laten_size: int = 128,
                 kernel_1: int = 64,
                 kernel_2: int = 16,
                 dropout: float = 0.25):

        super(EEGNetEncoder, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.chunk_size = chunk_size
        self.laten_size = laten_size
        self.num_electrodes = num_electrodes
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout

        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernel_1), stride=1, padding=(0, self.kernel_1 // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            Conv2dWithConstraint(self.F1,
                                 self.F1 * self.D, (self.num_electrodes, 1),
                                 max_norm=1,
                                 stride=1,
                                 padding=(0, 0),
                                 groups=self.F1,
                                 bias=False), nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(), nn.AvgPool2d((1, 4), stride=4), nn.Dropout(p=dropout))

        self.block2 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D,
                      self.F1 * self.D, (1, self.kernel_2),
                      stride=1,
                      padding=(0, self.kernel_2 // 2),
                      bias=False,
                      groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3), nn.ELU(), nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout))

        self.mu = nn.Linear(self.feature_dim(), self.laten_size, bias=False)
        self.logvar = nn.Linear(self.feature_dim(), self.laten_size, bias=False)
    
    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)
            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)

        return self.F2 * mock_eeg.shape[3]

    def forward(self, x: torch.Tensor):
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)
        mu = self.mu(x)
        std = self.logvar(x)

        return mu, std

class EEGNetDecoder(nn.Module):
    def __init__(self,
                 chunk_size: int = 151,
                 num_electrodes: int = 60,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 laten_size: int = 128,
                 kernel_1: int = 64,
                 kernel_2: int = 16,
                 dropout: float = 0.25):
        super(EEGNetDecoder, self).__init__()

        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.chunk_size = chunk_size
        self.laten_size = laten_size
        self.num_electrodes = num_electrodes
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout

        self.lin = nn.Linear(self.laten_size, self.F2, bias=False)
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(self.F2, self.F1 * self.D, (1,self.kernel_2 * 4), 1, padding=(0,2), bias=False, groups=1),
            nn.ConvTranspose2d(self.F1 * self.D, self.F1 * self.D, 
                               (1, self.kernel_2 // 2), 1, 
                               padding=(0, self.kernel_2 // 2), 
                               bias=False, 
                               groups=self.F1 * self.D), 
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.AvgPool2d((1, 1), stride=2))


        self.block2 = nn.Sequential(ConvTranspose2dWithConstraint(self.F1 * self.D, self.F1, (self.num_electrodes,1),
                            max_norm=1,
                            stride=1,
                            padding=(0, 0),
                            groups=F1,
                            bias=False),
                        nn.ConvTranspose2d(self.F1, 1, (1, self.valid_final_kernal_size()), (1,2), padding=(0, 0), bias=False),
                        nn.BatchNorm2d(1, momentum=0.01, affine=True, eps=1e-3),
                        nn.ELU(),
                        nn.AvgPool2d((1, 2), stride=1),
                        nn.Dropout(p=0.25))
        
    def valid_final_kernal_size(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, self.laten_size)
            mock_eeg = self.lin(mock_eeg)
            mock_eeg = mock_eeg[:, :, None, None]
            mock_eeg = self.block1(mock_eeg)

        w = mock_eeg.shape[-1]
        return (self.chunk_size + 1) - (w - 1) * 2

    def forward(self, x: torch.Tensor):
        x = self.lin(x)
        x = x[:, :, None, None] # add missing dimension, later be upsampled with convTranspose
        x = self.block1(x)
        x = self.block2(x)

        return x
    


if __name__ == '__main__':
    mock_eeg = torch.rand(1, 1, 64, 50,)

    enc = EEGNetEncoder(50, 64, 8, 16, 4, 128, 32, 8, 0.25)
    dec = EEGNetDecoder(50, 64, 8, 16, 4, 128, 32, 8, 0.25)

    with torch.no_grad():
        print(mock_eeg.shape)
        out = enc(mock_eeg)
        print(out[0].shape, out[1].shape)
        recon = dec(out[0])
        print(recon.shape)