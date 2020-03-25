import torch
from F2FDataloader import *
from F2FGenerator import *
from F2FDiscriminator import *
from train_gan import *
from Inference_Evaluation import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataloader_dict = {}
dataloader_dict = GAN_dataloader_dict(dataloader_dict, path="Feature2Face_data",
                                      subfolder="train", data_size=256, batch_size=1)
dataloader_dict = GAN_dataloader_dict(dataloader_dict, path="Feature2Face_data",
                                      subfolder="val", data_size=256, batch_size=1)
G = Generator(64)
D = Discriminator()

num_epoch = 1
Dloss_hist, Gloss_hist = train_FaceFeature2Face(G, D, dataloader_dict, num_epoch=num_epoch, batch_size=1)
# Loss_Visualization(Dloss_hist, Gloss_hist, num_epoch)

# G = Generator()
# G.load_state_dict(torch.load(f'G_Feature2Face/model_{num_epoch}.pth'))
# dataloader_dict = {}
# dataloader_dict = GAN_dataloader_dict(dataloader_dict, path="Feature2Face_data", subfolder="test", data_size=256, batch_size=20)
# photos = Inference(G, dataloader_dict)
# Face_Visualization(photos[0].detach().cpu().numpy(), 4, 4)
