import joblib
import pandas as pd

pipeline = joblib.load("models/credit_risk_model.pkl")



def kredi_riski_tahmin_et(musteri_bilgileri):
    df_musteri = pd.DataFrame([musteri_bilgileri])
    tahmin = pipeline.predict(df_musteri)[0]
    olasilik = pipeline.predict_proba(df_musteri)[0][1]  #olasilik[0][1] = kötü kredi olasılığı [0][0] ' da iyi kredi 
    return tahmin,olasilik


musteri_1 ={
    "checking_account_status": "<0 DM",
    "duration_months": 24,
    "credit_history": "simdiye odendi",
    "purpose": "araba(yeni)",
    "credit_amount": 3500,
    "savings_account": "<100 DM",
    "employment_since": "1-4 yil",
    "installment_rate_pct": 3,
    "personal_status_sex": "erkek:bekar",
    "other_debtors": "A101",
    "residence_since": 2,
    "property": "A121",
    "age": 29,
    "other_installment_plans": "A143",
    "housing": "kendi evi",
    "existing_credits": 1,
    "job": "vasifli",
    "num_dependents": 1,
    "telephone": "A191",
    "foreign_worker": "A201",
}

tahmin,olasilik = kredi_riski_tahmin_et(musteri_1)

print(f"Tahmin: {tahmin}, Kotu kredi olasiligi: %{olasilik*100:.1f}")



