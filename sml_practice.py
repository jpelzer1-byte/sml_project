
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('sample_videos.csv')

df["views"].sum()

df["engagement_rate"].mean()

top_views = df.groupby("title")["views"].max().sort_values(ascending = False)

top_engagement = df.groupby("title")["engagement_rate"].max().sort_values(ascending = False) 

df["performance_label"] = np.where(df["engagement_rate"] >= .10, "Strong", "Needs Improvement")

df.to_csv("creator_report.csv", index = False)

df["strong"] = (df["engagement_rate"] >= 0.10).astype(int)

features = ["views", "length_seconds", "post_hour"]

x = df[features]
y = df["strong"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=42)

model = DecisionTreeClassifier(random_state=42)

model.fit(x_train, y_train) #teaches the model the information. learns clues and answers. 

y_pred = model.predict(x_test) #predicts from unseen clues

print(accuracy_score(y_test, y_pred)) #compares predictions with answers

print(classification_report(y_test, y_pred))








