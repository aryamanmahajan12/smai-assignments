import pandas as pd

# Option 1: Using raw string
data = pd.read_feather(r"C:\Users\aryma\Desktop\SMAI\smai\smai-m24-assignments-aryamanmahajan123\data\external\word-embeddings.feather")

# Option 2: Using double backslashes
# data = pd.read_feather("C:\\Users\\aryma\\Desktop\\SMAI\\smai\\smai-m24-assignments-aryamanmahajan123\\data\\external\\word-embeddings.feather")

numsample, numfet = data.shape
print(numsample)
print(numfet)
print(data[50:60])
