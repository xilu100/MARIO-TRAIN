import intel_extension_for_pytorch as ipex
import torch

print(ipex.__version__)
print(torch.__version__)
print(torch.xpu.is_available())
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print(f"Using device: {device}")
