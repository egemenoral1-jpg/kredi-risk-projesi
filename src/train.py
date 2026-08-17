import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


df = pd.read_csv("data/processed/german_credit_clean.csv")


X = df.drop(columns = ["target"])
y = df["target"]

kategorik_sutunlar = X.select_dtypes(include=["object"]).columns.tolist()
sayisal_sutunlar = X.select_dtypes(exclude=["object"]).columns.tolist()


X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,stratify=y,test_size=0.25)

print(kategorik_sutunlar)
print(sayisal_sutunlar)

preprocessor = ColumnTransformer(transformers=[
    ("num",StandardScaler(),sayisal_sutunlar),  
    ("cat",OneHotEncoder(drop="first",handle_unknown="ignore"),kategorik_sutunlar)

])


pipeline = Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("model",RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight="balanced"))
])
pipeline.fit(X_train,y_train)


y_pred = pipeline.predict(X_test)
print(classification_report(y_test,y_pred))


scores = cross_val_score(pipeline,X,y,cv = 5,scoring="recall")
print(scores)
print(f"ortalama = {scores.mean()}")
print(f"Std sapma = {scores.std()}") 


joblib.dump(pipeline, "models/credit_risk_model.pkl")
print("Model kaydedildi.")