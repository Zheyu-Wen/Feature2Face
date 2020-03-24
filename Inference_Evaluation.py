import matplotlib.pyplot as plt
import numpy as np


def Inference(G, dataloader):
  G.eval()
  photos = []
  noise = np.random.randn(7)/100
  for feature, _ in dataloader["test"]:
    image_pred = G(feature, noise)
    photos += [image_pred]
  return photos

def Face_Visualization(photos, num_col, num_row):
  fig, ax = plt.subplots(num_row, num_col)
  for ir in range(num_row):
    for ic in range(num_col):
      ax[ir, ic].imshow(photos[ir*num_col+ic].transpose(1,2,0))
  return 0

def Loss_Visualization(Dloss_hist, Gloss_hist, num_epoch):
  plt.figure(1)
  plt.plot(np.arange(0,num_epoch), Dloss_hist)
  plt.title("D loss history")
  plt.xlabel("iteration")
  plt.ylabel("loss")

  plt.figure(2)
  plt.plot(np.arange(0,num_epoch), Gloss_hist)
  plt.title("G loss history")
  plt.xlabel("iteration")
  plt.ylabel("loss")