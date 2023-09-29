from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image

class EmotionNLPDataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.sent_labels = pd.read_csv(annotations_file, delimiter=';')
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        sent = self.sent_labels.iloc[idx, 0]
        label = self.sent_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label