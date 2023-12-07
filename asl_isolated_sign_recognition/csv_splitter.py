import pandas as pd
import os

input_csv = 'data/asl-signs/train.csv'
output_folder = 'data/asl-signs/split-csvs/'

os.makedirs(output_folder, exist_ok=True)

chunk_size = 100

csv = pd.read_csv(input_csv, delimiter=',')

header = ["path", "participant_id", "sequence_id", "sign"]

for index_start in range(1, len(csv), chunk_size):
    subset = csv[index_start:index_start+chunk_size]
    
    out_path = os.path.join(output_folder, f'train_{index_start}_{index_start+chunk_size-1}.csv')

    subset.to_csv(out_path, header=header)
