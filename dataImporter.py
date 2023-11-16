import pandas as pd

column_names = ["input", "target"]
df = pd.read_csv('./data/C4_200M.tsv-00000-of-00010',sep = '\t')
df = df.shift(periods=1,fill_value=None)
df.loc[0] = df.columns
df.columns = column_names
# print(df)
# df.to_csv("./CSVdata/data1.csv")