import torchvision.models as models
from torchvision import transforms
from dataloader import *
import numpy as np
import torch
from torch.utils.data import DataLoader


preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


deeplab = models.segmentation.deeplabv3_resnet101(pretrained=False,
            progress=True, num_classes=21, aux_loss=True)


print(deeplab)


loader = MapillaryLoader("/home/lunet/wsmo6/mapillary/dataset", "v2.0", preprocess)
train_loader = DataLoader(dataset = loader, batch_size = 16, shuffle = True)

#print(train_loader[0])

for i in train_loader:
    print(i[0])
    break
    print(torch.max(input_tensor))

    break


# for epoch in range(epochs):
#     for batch in range(mini-batches):
#         out = model(batch)
