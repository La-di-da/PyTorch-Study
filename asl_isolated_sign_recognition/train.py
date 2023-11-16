import dataset_reader
import os
from neural_model import AslSignModel
import torch
from torch import nn, optim

root = '/Users/ark/projects/PyTorch-Study/PyTorch-Study/data/asl-signs/'
train_csv = os.path.join(root, 'train.csv')

dataset = dataset_reader.AslSignData(train_csv, root_path=root)

print(len(dataset))

for parq_vec, label in dataset:
    first_vec = parq_vec
    break

EMBEDDING_DIM = len(parq_vec[0])
HIDDEN_DIM = 64
tagset_size = 1

model = AslSignModel(EMBEDDING_DIM, HIDDEN_DIM, tagset_size)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tag in dataset:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 3. Run our forward pass.
        tag_scores = model(sentence)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, tag) # tag needs to be a 250 length vector
        loss.backward()
        optimizer.step()