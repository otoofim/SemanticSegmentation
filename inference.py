import torch
from unet import *



model = UNet(out_channels = 3)
model.load_state_dict(torch.load("/home/lunet/wsmo6/deeplab/checkpoints/test/best.pth")['model_state_dict'])
model.eval()
