import torch

x = torch.rand(10, 10, dtype=torch.complex64)
ifft2 = torch.fft.ifft2(x)

print(x.shape)