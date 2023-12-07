import dataset_reader
import os
from neural_model import AslSignModel
from torch.utils.data import DataLoader
from torch import nn, optim, Tensor

root = '/Users/ark/projects/PyTorch-Study/PyTorch-Study/data/asl-signs/'
csv_dir = os.path.join(root, "split-csvs")
train_csv = os.path.join(root, 'train.csv')
sign_mapping = os.path.join(root, 'sign_to_prediction_index_map.json')

training_csvs = sorted([os.path.join(csv_dir, file) for file in os.listdir(csv_dir) if file.endswith('.csv')])

dataset = dataset_reader.AslSignData(training_csvs[0], root_path=root, sign_to_pred=sign_mapping)

train_dataloader = DataLoader(dataset, shuffle=True)

for parq_vec, label, _ in dataset:
    first_vec = parq_vec
    break

EMBEDDING_DIM = len(parq_vec[0])
HIDDEN_DIM = 64
tagset_size = dataset.tagset_size

model = AslSignModel(EMBEDDING_DIM, HIDDEN_DIM, tagset_size)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(5):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tag, id in train_dataloader:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 3. Run our forward pass.
        tag_scores = model(sentence)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss : Tensor = loss_function(tag_scores, tag) # tag needs to be a 250 length vector
        loss.backward()
        optimizer.step()

        # prints the training line ID for samples above the loss threshold
        # might be useful for diagnostics?
        if loss > 1:
            print(id)