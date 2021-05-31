from model.config import *
from model.spectral_nomalization import SpectralNorm as SN


# Image Encoder
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True)
        )
        self.g = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64)
        )

    def forward(self, input):
        output = self.linear(input)
        return F.normalize(output, dim=-1), F.normalize(self.g(output), dim=-1)


# Gradient Reversal Layer
class GRL(torch.autograd.Function):
    def forward(self, inputs):
        return inputs

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input = -grad_input
        return grad_input
