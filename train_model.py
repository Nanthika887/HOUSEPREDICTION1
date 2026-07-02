import pandas as pd
import pickle
from sklearn.linear_model import ElasticNet

# Read the CSV
df = pd.read_csv("HouseData.csv")   # or house_price_data.csv

print(df.head())
print(df.columns)

X = df[["area", "bedrooms", "bathrooms"]]
y = df["price"]

model = ElasticNet(alpha=0.1, l1_ratio=0.5)
model.fit(X, y)

with open("model_pickle", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")