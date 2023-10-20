import pandas

parquet_file = "data/asl-signs/train_landmark_files/2044/635217.parquet"

parquet_contents = pandas.read_parquet(parquet_file)

print(parquet_contents)