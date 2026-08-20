import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score


df = pd.read_csv("data/processed/german_credit_clean.csv")


X = df.drop(columns = ["target"])
y = df["target"]

kategorik_sutunlar = X.select_dtypes(include=["object"]).columns.tolist()
sayisal_sutunlar = X.select_dtypes(exclude=["object"]).columns.tolist()


X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,stratify=y,test_size=0.25)




#print(kategorik_sutunlar)
#print(sayisal_sutunlar)

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


scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
pipeline_xgb = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42))
])

param_grid = {
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.01, 0.1, 0.3],
    "model__n_estimators": [100, 200]
}
grid_search = GridSearchCV(pipeline_xgb, param_grid, cv=5, scoring="recall", n_jobs=-1)
grid_search.fit(X_train, y_train)

print("En iyi parametreler:", grid_search.best_params_)
best_xgb = grid_search.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)
print(classification_report(y_test, y_pred_xgb))
scores_xgb = cross_val_score(best_xgb, X, y, cv=5, scoring="recall")
print(scores_xgb)
print(f"ortalama = {scores_xgb.mean()}")
print(f"Std sapma = {scores_xgb.std()}")




f2_rf = fbeta_score(y_test, y_pred, beta=2)
f2_xgb = fbeta_score(y_test, y_pred_xgb, beta=2)
print(f"RF F2: {f2_rf}")
print(f"XGBoost F2: {f2_xgb}")






joblib.dump(pipeline, "models/credit_risk_model.pkl")
print("Model kaydedildi.")