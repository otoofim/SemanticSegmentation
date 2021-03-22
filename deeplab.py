import torchvision.models as models
from torchvision import transforms
from dataloader import *
import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision
import wandb


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        #plt.imshow(npimg)



epochs = 10
batch_size = 20
val_every = 2
lr = 0.001
momentum = 0.9
img_w = 224
img_h = 224


#writer = SummaryWriter('runs/mapillary')



preprocess_in = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize((img_w, img_h))
])

preprocess_ou = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((img_w, img_h))
])



deeplab = models.segmentation.deeplabv3_resnet101(pretrained=False,
            progress=True, num_classes=21, aux_loss=True)
unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)


tr_loader = MapillaryLoader("/home/lunet/wsmo6/mapillary/dataset", "v2.0",
                            preprocess_in, preprocess_ou, mode = 'tra')
train_loader = DataLoader(dataset = tr_loader, batch_size = batch_size, shuffle = True)

val_loader = MapillaryLoader("/home/lunet/wsmo6/mapillary/dataset", "v2.0",
                            preprocess_in, preprocess_ou, mode = 'val')
val_loader = DataLoader(dataset = val_loader, batch_size = batch_size, shuffle = True)

optimizer = optim.SGD(unet.parameters(), lr = lr, momentum = momentum)
criterion = torch.nn.MSELoss(reduction='mean')


tr_loss = 0.0
val_loss = 0.0

with tqdm(range(epochs), unit="epoch", leave = True, position = 0) as epobar:
    for epoch in epobar:
            epobar.set_description("Epoch {}".format(epoch + 1))
            epobar.set_postfix(ordered_dict = {'tr_loss':tr_loss, 'val_loss':val_loss})

            tr_loss = 0.0
            with tqdm(train_loader, unit="batch", leave = False) as batbar:
                for i, batch in enumerate(batbar):

                    batbar.set_description("Batch {}".format(i + 1))

                    optimizer.zero_grad()

                    out = unet(batch['image'])
                    loss = criterion(out, batch['label'])

                    loss.backward()
                    optimizer.step()
                    tr_loss += loss.item()

            tr_loss /= len(train_loader)
            #print(len(train_loader))

            if ((epoch+1) % val_every == 0):
                with tqdm(val_loader, unit="batch", leave = False) as valbar:
                    with torch.no_grad():
                        val_loss = 0.0
                        for i, batch in enumerate(valbar):

                            valbar.set_description("Val_batch {}".format(i + 1))

                            out = unet(batch['image'])
                            loss = criterion(out, batch['label'])

                            val_loss += loss.item()


                    val_loss /= len(val_loader)
