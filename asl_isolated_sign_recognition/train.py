import dataset_reader
import os

root = '/Users/ark/projects/PyTorch-Study/PyTorch-Study/data/asl-signs/'
train_csv = os.path.join(root, 'train.csv')

dataset = dataset_reader.AslSignData(train_csv, root_path=root)

print(len(dataset))

for input, label in dataset:
    for frame in input:
        print(len(frame))
        break
    break