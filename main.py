
import torch
from model_component import *
from F2FDataloader import *
from F2FGenerator import *
from F2FDiscriminator import *
from train_gan import *
from Inference_Evaluation import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataloader_dict = {}
dataloader_dict = GAN_dataloader_dict(dataloader_dict, path="Feature2Face_data_viability_test",
                                      subfolder="train", data_size=256, batch_size=20)
dataloader_dict = GAN_dataloader_dict(dataloader_dict, path="Feature2Face_data_viability_test",
                                      subfolder="val", data_size=256, batch_size=5)
G = Generator()
D = Discriminator()

num_epoch = 10
Dloss_hist, Gloss_hist = train_FaceFeature2Face(G, D, dataloader_dict, num_epoch=num_epoch)
Loss_Visualization(Dloss_hist, Gloss_hist, num_epoch)

data = torch.load(f'Feature2Face/model_{num_epoch}.pth')
G = Generator()
G.load_state_dict(data["model_state_dict"])
dataloader_dict = {}
dataloader_dict = GAN_dataloader_dict(dataloader_dict, path="Feature2Face_data_viability_test", subfolder="test", data_size=256, batch_size=20)
photos = Inference(G, dataloader_dict)
Face_Visualization(photos, 4, 4)