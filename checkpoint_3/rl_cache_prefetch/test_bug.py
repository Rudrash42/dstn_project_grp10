import pandas as pd
df = pd.read_csv("data/traces_rag.csv")
print("RAG length:", len(df))
print(df.head())
