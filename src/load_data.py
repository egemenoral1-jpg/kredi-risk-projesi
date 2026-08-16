import pandas as pd
import pandas as pd

columns = [
    "checking_account_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_account", "employment_since", "installment_rate_pct",
    "personal_status_sex", "other_debtors", "residence_since", "property",
    "age", "other_installment_plans", "housing", "existing_credits",
    "job", "num_dependents", "telephone", "foreign_worker", "target"
]

df = pd.read_csv("data/raw/german.csv", header=None, names=columns)

df["target"] = df["target"].map({1: 0, 2: 1})

print(df["target"].value_counts())

checking_map = {"A11": "<0 DM", "A12": "0-200 DM", "A13": ">=200 DM", "A14": "hesap yok"}
credit_hist_map = {"A30": "kredi yok/hepsi odendi", "A31": "bu bankada odendi", "A32": "simdiye odendi",
                    "A33": "gecmiste gecikme", "A34": "kritik hesap"}
purpose_map = {"A40": "araba(yeni)", "A41": "araba(eski)", "A42": "mobilya", "A43": "radyo/TV",
               "A44": "ev aleti", "A45": "tamirat", "A46": "egitim", "A48": "yeniden egitim",
               "A49": "is", "A410": "diger"}
savings_map = {"A61": "<100 DM", "A62": "100-500 DM", "A63": "500-1000 DM", "A64": ">=1000 DM", "A65": "bilinmiyor"}
employment_map = {"A71": "issiz", "A72": "<1 yil", "A73": "1-4 yil", "A74": "4-7 yil", "A75": ">=7 yil"}
housing_map = {"A151": "kira", "A152": "kendi evi", "A153": "ucretsiz"}
job_map = {"A171": "issiz/vasifsiz", "A172": "vasifsiz-yerlesik", "A173": "vasifli", "A174": "yonetici/uzman"}


df["checking_account_status"] = df["checking_account_status"].map(checking_map)
df["credit_history"] = df["credit_history"].map(credit_hist_map)
df["purpose"] = df["purpose"].map(purpose_map)
df["savings_account"] = df["savings_account"].map(savings_map)
df["employment_since"] = df["employment_since"].map(employment_map)
df["housing"] = df["housing"].map(housing_map)
df["job"] = df["job"].map(job_map)



df.to_csv("data/processed/german_credit_clean.csv", index=False)
print("Kaydedildi.")




