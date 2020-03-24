import torch
import torchvision
import torch.nn.functional as F
from math import sqrt
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def train_FaceFeature2Face(G, D, dataloader, num_epoch, batch_size):
    Dloss_hist = []
    Gloss_hist = []
    criterionBCE = nn.BCELoss().cuda()
    criterionL1 = nn.L1Loss().cuda()
    Doptimizer = optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.999))
    Goptimizer = optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))
    # noise = np.zeros(7)/100
    lambda1 = 100
    Dninner = 10
    Gninner = 2
    for iters in range(num_epoch):
        for role in ["train", "val"]:
            Dloss_avg = 0
            Gloss_avg = 0
            inner = 0
            for feature, image in tqdm(dataloader[role]):
                # cv2.imwrite("gt.jpg", np.array(255*image[0,:,:,:].permute(1,2,0)))
                if role == "train":
                    D.train()
                    G.train()
                else:
                    D.eval()
                    G.eval()

                inner += 1
                if role=="train":
                    for _ in range(Dninner):
                        set_requires_grad(D, requires_grad=True)
                        set_requires_grad(G, requires_grad=False)
                        noise = torch.Tensor(np.random.randn(feature.shape[0], 1, 64, 64))/10
                        fake_image = G(feature, noise)
                        Dout_fake = D(fake_image)
                        Dout_true = D(image)
                        Dloss = 1 / 2 * (criterionBCE(Dout_fake, torch.zeros_like(Dout_fake)) + criterionBCE(Dout_true, torch.ones_like(Dout_true)))
                        Doptimizer.zero_grad()
                        Dloss.backward()
                        Doptimizer.step()
                    Dloss_avg += Dloss
                else:
                    set_requires_grad(D, requires_grad=False)
                    set_requires_grad(G, requires_grad=False)
                    noise = torch.Tensor(np.random.randn(feature.shape[0], 1, 64, 64))/10
                    fake_image = G(feature, noise)
                    Dout_fake = D(fake_image)
                    Dout_true = D(image)
                    Dloss = 1 / 2 * (criterionBCE(Dout_fake, torch.zeros_like(Dout_fake)) + criterionBCE(Dout_true, torch.ones_like(Dout_true)))
                    Dloss_avg += Dloss

                if role=="train":
                    for _ in range(Gninner):
                        set_requires_grad(D, requires_grad=False)
                        set_requires_grad(G, requires_grad=True)
                        noise = torch.Tensor(np.random.randn(feature.shape[0], 1, 64, 64))/10
                        fake_image = G(feature, noise)
                        Dout_fake = D(fake_image)
                        Gloss = criterionBCE(Dout_fake, torch.ones_like(Dout_fake)) + lambda1 * criterionL1(fake_image, image)
                        Goptimizer.zero_grad()
                        Gloss.backward()
                        Goptimizer.step()
                    Gloss_avg += Gloss

                else:
                    set_requires_grad(D, requires_grad=False)
                    set_requires_grad(G, requires_grad=False)
                    noise = torch.Tensor(np.random.randn(feature.shape[0], 1, 64, 64))/10
                    fake_image = G(feature, noise)
                    Dout_fake = D(fake_image)
                    Gloss = criterionBCE(Dout_fake, torch.ones_like(Dout_fake)) + lambda1 * criterionL1(fake_image,image)
                    Gloss_avg += Gloss

            Dloss_avg /= inner
            Gloss_avg /= inner
            print("Epoch: {}. The average loss in {} for D is {}".format(iters, role, Dloss_avg))
            print("The average loss for G is {}".format(Gloss_avg))
            pred_fake_image = fake_image.permute(0, 2, 3, 1).detach().cpu().numpy()
            SaveImage(pred_fake_image, pred_fake_image.shape[0])
            # Batch_Visualization(pred_fake_image, pred_fake_image.shape[0])
            if (iters+1)%5 == 0:
                if not os.path.exists("G_Feature2Face"):
                    os.makedirs("G_Feature2Face")
                if not os.path.exists("D_Feature2Face"):
                    os.makedirs("D_Feature2Face")
                torch.save(G.state_dict(), f"G_Feature2Face/model_{iters+1}.pth")
                torch.save(D.state_dict(), f"D_Feature2Face/model_{iters+1}.pth")
        Dloss_hist.append(Dloss_avg)
        Gloss_hist.append(Gloss_avg)
    return Dloss_hist, Gloss_hist

def Batch_Visualization(fake_image, batch_size):
    if batch_size>=6:
        fig,ax = plt.subplots(2,3)
        for i in range(2):
            for j in range(3):
                ax[i, j].imshow(fake_image[i*3+j, :, :, :])
    else:
        fig,ax = plt.subplots(batch_size)
        for k in range(batch_size):
            ax[k].imshow(fake_image[k, :, :, :])

def SaveImage(batch_fake_image, batch_size):
    for i in range(batch_size):
        new_im = Image.fromarray(np.uint8(batch_fake_image[i, :, :, :]*255))
        new_im.save("test_{}.png".format(i))
