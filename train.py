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
from unet import *
import os
import efemarai as ef




def train(batch_size, epoch, learning_rate, run_name, data_path):


    hyperparameter_defaults = {
        "batch_size": batch_size,
        "lr": learning_rate,
        "epochs": epoch,
        "momentum": 0.9,
        "architecture": "unet",
        "dataset": "Mapillary",
        "run": run_name
    }

    base_add = os.getcwd()

    wandb.init(config = hyperparameter_defaults, project = 'unet', entity = 'moh1371',
                name = hyperparameter_defaults['run'])

    val_every = 2
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


    #deeplab = models.segmentation.deeplabv3_resnet101(pretrained=False,
    #            progress=True, num_classes=21, aux_loss=True)
    #unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    #    in_channels=3, out_channels=3, init_features=32, pretrained=False)

    unet = UNet(out_channels = 3)
    model = unet

    tr_loader = MapillaryLoader(data_path, "v2.0",
                                preprocess_in, preprocess_ou, mode = 'tra')
    train_loader = DataLoader(dataset = tr_loader, batch_size = wandb.config.batch_size, shuffle = True)

    val_loader = MapillaryLoader(data_path, "v2.0",
                                preprocess_in, preprocess_ou, mode = 'val')
    val_loader = DataLoader(dataset = val_loader, batch_size = wandb.config.batch_size, shuffle = True)

    optimizer = optim.SGD(model.parameters(), lr = wandb.config.lr, momentum = wandb.config.momentum)
    criterion = torch.nn.MSELoss(reduction='mean')


    tr_loss = 0.0
    val_loss = 0.0
    best_val = 1e10
    wandb.watch(model)

    with tqdm(range(wandb.config.epochs), unit="epoch", leave = True, position = 0) as epobar:
        for epoch in epobar:
                epobar.set_description("Epoch {}".format(epoch + 1))
                epobar.set_postfix(ordered_dict = {'tr_loss':tr_loss, 'val_loss':val_loss})

                tr_loss = 0.0
                with tqdm(train_loader, unit="batch", leave = False) as batbar:
                    for i, batch in enumerate(batbar):


                        batbar.set_description("Batch {}".format(i + 1))

                        optimizer.zero_grad()

                        #ef.add_view(batch['image'], ef.View.IMAGE)

                        #with ef.scan():
                        out = model(batch['image'])
                        loss = criterion(out, batch['label'])
                        loss.backward()

                        optimizer.step()
                        tr_loss += loss.item()

                        org_img = {'input':wandb.Image(batch['image']),
                         "ground truth":wandb.Image(batch['label']),
                         "prediction":wandb.Image(out)}
                        wandb.log(org_img)


                tr_loss /= len(train_loader)
                wandb.log({"tr_loss": tr_loss, "epoch": epoch + 1})


                if ((epoch+1) % val_every == 0):
                    with tqdm(val_loader, unit="batch", leave = False) as valbar:
                        with torch.no_grad():
                            val_loss = 0.0
                            for i, batch in enumerate(valbar):

                                valbar.set_description("Val_batch {}".format(i + 1))

                                out = model(batch['image'])
                                loss = criterion(out, batch['label'])

                                val_loss += loss.item()


                        val_loss /= len(val_loader)
                        wandb.log({"val_loss": val_loss, "epoch": epoch + 1})
                        if val_loss < best_val:

                            newpath = base_add + "/checkpoints/{}".format(hyperparameter_defaults['run'])

                            if not os.path.exists(base_add + "/checkpoints"):
                                os.makedirs(ase_add + "/checkpoints")

                            if not os.path.exists(newpath):
                                os.makedirs(newpath)

                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'tr_loss': tr_loss,
                                'val_loss': val_loss,
                                }, newpath + "/best.pth")
