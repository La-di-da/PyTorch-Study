import pandas as pd

input_csv = 'data/asl-signs/train.csv'

chunk_size = 100

csv = pd.read_csv(input_csv, delimiter=',')

