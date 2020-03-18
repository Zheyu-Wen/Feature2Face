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


def train_FaceFeature2Face(G, D, dataloader, num_epoch):
    Dloss_hist = []
    Gloss_hist = []
    criterionBCE = nn.BCELoss().cuda()
    criterionL1 = nn.L1Loss().cuda()
    Doptimizer = optim.Adam(D.parameters(), lr=1e-4)
    Goptimizer = optim.Adam(G.parameters(), lr=1e-4)
    noise = np.zeros(7)/100
    lambda1 = 100
    for iters in range(num_epoch):
        Dloss_avg = 0
        Gloss_avg = 0
        for role in ["train", "val"]:
            inner = 0
            for feature, image in tqdm(dataloader[role]):
                if role == "train":
                    D.train()
                    G.train()
                else:
                    D.eval()
                    G.eval()

                inner += 1

                set_requires_grad(D, requires_grad=True)
                set_requires_grad(G, requires_grad=False)
                fake_image = G(feature, noise)
                Dout_fake = D(fake_image)
                Dout_true = D(image)
                Dloss = 1 / 2 * (criterionBCE(Dout_fake, torch.zeros_like(Dout_fake)) + criterionBCE(Dout_true,
                                                                                                     torch.ones_like(
                                                                                                         Dout_true)))
                Dloss_avg += Dloss
                if role == "train":
                    Doptimizer.zero_grad()
                    Dloss.backward()
                    Doptimizer.step()

                set_requires_grad(D, requires_grad=False)
                set_requires_grad(G, requires_grad=True)
                fake_image = G(feature, noise)
                Dout_fake = D(fake_image)
                Gloss = criterionBCE(Dout_fake, torch.ones_like(Dout_fake)) + lambda1 * criterionL1(fake_image, image)
                Gloss_avg += Gloss
                if role == "train":
                    Goptimizer.zero_grad()
                    Gloss.backward()
                    Goptimizer.step()

            Dloss_avg /= inner
            Gloss_avg /= inner
            print("The average loss for D is {}".format(Dloss_avg))
            print("The average loss for G is {}".format(Gloss_avg))
            pred_fake_image = fake_image.view(0,2,3,1).detach().cpu().numpy()
            Batch_Visualization(pred_fake_image, pred_fake_image.shape[0])
            if iters % 5 == 0:
                if not os.path.exists("G_Feature2Face"):
                    os.makedirs("G_Feature2Face")
                if not os.path.exists("D_Feature2Face"):
                    os.makedirs("D_Feature2Face")
                torch.save(G.state_dict(), f"G_Feature2Face/model_{iters}.path")
                torch.save(D.state_dict(), f"D_Feature2Face/model_{iters}.path")
        Dloss_hist.append(Dloss_avg)
        Gloss_hist.append(Gloss_avg)
    return Dloss_hist, Gloss_hist

def Batch_Visualization(fake_image, batch_size):
    if batch_size>=6:
        fig,ax = plt.subplots(2,3)
        for i in range(2):
            for j in range(3):
                ax[i, j].imshow(fake_image[i*3+j,:,:,:])
    else:
        fig,ax = plt.subplots(batch_size)
        for k in range(batch_size):
            ax[k].imshow(fake_image[k,:,:,:])
