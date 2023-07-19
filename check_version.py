import torch
import torchvision
# import torchaudio

print("pytorch version: ", torch.__version__)
print("CUDA version: ", torch.version.cuda)

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))
print(torch.version)
print(torchvision.version)
# print(torchaudio.version)
# import torch.searchsorted