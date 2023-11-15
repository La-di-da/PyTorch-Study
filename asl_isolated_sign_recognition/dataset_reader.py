from torch.utils.data import ConcatDataset, Dataset
import pandas as pd
import os
from typing import List

def element_to_vec(element : pd.DataFrame) -> List[float]:
    vec = []
    for x, y in zip(element["x"], element["y"]):
        vec.append(x)
        vec.append(y)
    return vec

class AslSignData(Dataset):
    def __init__(self, train_csv, transform=None, target_transform=None, root_path=None):
        self.parquet_labels = pd.read_csv(train_csv, delimiter=',')
        self.transform = transform
        self.target_transform = target_transform
        self.root = root_path

    def __len__(self):
        return len(self.parquet_labels)

    def __getitem__(self, idx):
        parquet = self.parquet_labels.iloc[idx, 0] # column: path
        label = self.parquet_labels.iloc[idx, 3] # column: sign

        contents = pd.read_parquet(os.path.join(self.root, parquet))
        contents = contents.fillna(0) # replace NaN with something that doesn't break the neural network

        content_grouped = contents.groupby("frame")
        content_by_frame = [content_grouped.get_group(group) for group in sorted(content_grouped.groups.keys())]
        frame_vecs = []
        for frame in content_by_frame:
            frame_grouped = frame.groupby("type")
            face, pose, left_hand, right_hand = [frame_grouped.get_group(group) for group in frame_grouped.groups.keys()]

            joined_vec = element_to_vec(face)
            joined_vec.extend(element_to_vec(pose))
            joined_vec.extend(element_to_vec(right_hand))
            joined_vec.extend(element_to_vec(left_hand))

            frame_vecs.append(joined_vec)
        
        if self.transform:
            frame_vecs = self.transform(frame_vecs)
        if self.target_transform:
            label = self.target_transform(label)
        return frame_vecs, label
