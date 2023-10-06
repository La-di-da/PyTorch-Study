from dataset_reader import EmotionNLPDataset
from collections import OrderedDict
from pandas import DataFrame
import os
from neural_model import EmotionClassifer
import torch
import json

class Lexicon:
    def __init__(self, saved_lexicon : dict = {}):
        self.lexicon = saved_lexicon
        self.length = len(saved_lexicon)

    def build_lexicon(self, data : DataFrame) -> None:
        column = [data['sents']]
        for sent in data['sents']:
            for token in sent.split(' '):
                if token not in self.lexicon:
                    self.lexicon[token] = self.length
                    self.length += 1

    # this should create a vector based on the lexicon, using the tokens of the sentence, which is your
    # input to the model. It might work better using a dataframe, or some other vectorization strategy
    # the vector length has to be consistent. Everything else is free
    def vectorize_sent(self, sent):
        sent_vec = torch.zeros(self.length, dtype=torch.float)
        for token in sent:
            sent_vec[self.lexicon[token]] = 1

        return sent_vec

def label_transform(label):
    labels = {
        "sadness" : 0,
        "anger" : 1,
        "love" : 2,
        "surprise" : 3,
        "fear" : 4,
        "joy" : 5
    }
    return labels[label]

lexicon = Lexicon()

# these have to get functions - like "lexicon.vectorize_sent" to transform the input into vectors correctly
transform = lexicon.vectorize_sent
target_transform = label_transform

train = EmotionNLPDataset('data/emotion_nlp/train.txt',
                          transform=transform,
                          target_transform=target_transform)

test = EmotionNLPDataset('data/emotion_nlp/test.txt',
                          transform=transform,
                          target_transform=target_transform)

val = EmotionNLPDataset('data/emotion_nlp/val.txt',
                          transform=transform,
                          target_transform=target_transform)

lexicon.build_lexicon(train.sent_labels)
lexicon.build_lexicon(test)
lexicon.build_lexicon(val)

lexicon_path = 'models/emotion_nlp/lexicon.json'
os.makedirs(os.path.dirname(lexicon_path), exist_ok=True)
with open(lexicon_path, mode='w') as file:
    json.dump(lexicon.lexicon, file)

device = "cpu"
print(f"Using {device} device")
model = EmotionClassifer(lexicon.length).to(device)
print(model)

# add training here, look in the quickstart
def train(dataset, model, loss_fn, optimizer):
    size = len(dataset)
    model.train()
    for batch, (X, y) in enumerate(dataset):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

""" def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n") """


loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

train(test, model, loss_fn, optimizer)