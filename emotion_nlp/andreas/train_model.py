from dataset_reader import EmotionNLPDataset
from collections import OrderedDict
from pandas import DataFrame
import os
from neural_model import EmotionClassifer
import torch

class Lexicon:
    def __init__(self):
        self.lexicon = OrderedDict()

    def build_lexicon(self, data : DataFrame) -> None:
        for sent, _ in data:
            for token in sent.split(' '):
                self.lexicon[token] = ""

    def get_tokens(self):
        return self.lexicon.keys()
    
    # this should create a vector based on the lexicon, using the tokens of the sentence, which is your
    # input to the model. It might work better using a dataframe, or some other vectorization strategy
    # the vector length has to be consistent. Everything else is free
    def vectorize_sent(self, sent):
        tokens = self.get_tokens()
        torch.zeros(len(tokens), dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1)

# these have to get functions - like "lexicon.vectorize_sent" to transform the input into vectors correctly
transform = None
target_transform = None

train = EmotionNLPDataset('data/emotion_nlp/train.txt',
                          transform=transform,
                          target_transform=target_transform)

test = EmotionNLPDataset('data/emotion_nlp/test.txt',
                          transform=transform,
                          target_transform=target_transform)

val = EmotionNLPDataset('data/emotion_nlp/val.txt',
                          transform=transform,
                          target_transform=target_transform)

lexicon = Lexicon()
lexicon.build_lexicon(train)
lexicon.build_lexicon(test)
lexicon.build_lexicon(val)
tokens = lexicon.get_tokens()

lexicon_path = 'models/emotion_nlp/lexicon.txt'
os.makedirs(os.path.dirname(lexicon_path), exist_ok=True)
with open(lexicon_path, mode='w') as file:
    file.write('\n'.join(tokens))

model = EmotionClassifer(len(tokens))

# add training here, look in the quickstart