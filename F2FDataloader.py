import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torch

class Dataloader_for_Inverse(Dataset):
    def __init__(self, data_dir, train=True, size=256, n_hidden=128):
        super().__init__()
        self.embeds = sorted(glob.glob(f'{data_dir}/*face.npy', recursive=True))
        self.images = [direc.replace("face", "jpg") for direc in self.embeds]
        self.train = train
        self.n_hidden = n_hidden
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.embeds)

    def __getitem__(self, idx):
        embedding = np.load(self.embeds[idx])
        image = Image.fromarray(np.uint8(np.load(self.images[idx])))
        image = self.transform(image)

        # if self.train: embedding = embedding + np.random.randn(self.n_hidden) * 0.0001
        embedding = embedding.astype(np.float32)

        return embedding, image


def GAN_dataloader_dict(data_loader_dict, path, subfolder, data_size, batch_size):
    data = Dataloader_for_Inverse(path + "/" + subfolder, size=data_size)
    data_loader_dict[subfolder] = torch.utils.data.DataLoader(data, batch_size=batch_size)
    return data_loader_dict