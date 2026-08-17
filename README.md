# Kredi Riski Tahmin Projesi

Almanya'daki gerçek kredi başvurusu verileriyle (German Credit Data, UCI Machine Learning Repository), bir başvuranın **iyi mi kötü mü kredi riski** taşıdığını tahmin eden bir makine öğrenmesi modeli.

## Veri Seti

- **Kaynak:** UCI Machine Learning Repository — German Credit Data
- **Boyut:** 1000 gerçek kredi başvurusu, 20 özellik (yaş, meslek, kredi miktarı, hesap durumu vb.)
- **Hedef değişken:** `target` (0 = iyi kredi, 1 = kötü kredi)
- Sınıf dağılımı dengesiz: %70 iyi kredi, %30 kötü kredi

## Proje Yapısı

kredi-risk-projesi/
├── data/
│ ├── raw/ # ham veri
│ └── processed/ # temizlenmiş veri
├── notebooks/
│ └── 01_eda.ipynb # kesifsel veri analizi
├── src/
│ ├── load_data.py # veri yukleme ve temizleme
│ ├── train.py # model egitimi
│ └── predict.py # yeni basvuru icin tahmin
├── models/
│ └── credit_risk_model.pkl # egitilmis model
├── requirements.txt
└── README.md


## Keşifsel Veri Analizi — Öne Çıkan Bulgular

- **Vadesiz hesap durumu** en güçlü sinyal: hesabı olmayanlarda kötü kredi oranı %12, eksi bakiyeli hesabı olanlarda %49.
- **Yaş:** 18-25 yaş grubunda kötü kredi oranı %42, diğer gruplarda %24-30.
- **Kredi miktarı:** kötü kredi verilenlerin medyan kredi tutarı, iyi kredi verilenlere göre belirgin şekilde yüksek.
- **Kredi amacı (purpose):** yeni araba kredilerinde risk oranı (%38), ikinci el araba kredilerine (%16.5) göre çok daha yüksek — bu fark kredi tutarı, yaş veya vade ile açıklanamadı, muhtemelen bağımsız bir sinyal.

## Metodoloji

1. **Veri temizleme:** eksik değer, tekrarlanan satır ve aykırı değer kontrolü yapıldı (hiçbiri veri kalitesini bozacak düzeyde bulunmadı).
2. **Ön işleme:** `ColumnTransformer` ile sayısal değişkenler `StandardScaler`, kategorik değişkenler `OneHotEncoder` (`drop="first"`, `handle_unknown="ignore"`) kullanılarak dönüştürüldü.
3. **Model karşılaştırması:** Logistic Regression ve Random Forest, 5-fold cross-validation ile recall metriği üzerinden karşılaştırıldı.
4. **Hiperparametre optimizasyonu:** `GridSearchCV` ile Random Forest'ın `n_estimators` ve `max_depth` parametreleri optimize edildi.

## Model Performansı

| Model | Recall (CV ortalaması) |
|---|---|
| Logistic Regression | %48.0 |
| Random Forest (elle ayarlanmış) | %68.7 |
| **Random Forest (GridSearch, final)** | **%70.2** |

**Final model** (`n_estimators=100, max_depth=5`) test setinde:
- Precision (kötü kredi): 0.52
- Recall (kötü kredi): 0.73
- Accuracy: 0.72

**Neden recall önceliklendirildi:** Bankacılık bağlamında, riskli bir müşteriyi kaçırmak (false negative — banka parasını geri alamaz), iyi bir müşteriyi yanlışlıkla reddetmekten (false positive — banka sadece kâr kaybeder) daha maliyetlidir.

## En Etkili Özellikler (Feature Importance)

1. Vadesiz hesap durumu (hesap yok)
2. Kredi miktarı
3. Kredi süresi (ay)
4. Yaş
5. Kredi geçmişi (kritik hesap)

Bu sıralama, EDA'da elle bulunan bulgularla tutarlıdır.

## Kurulum ve Çalıştırma

```powershell
python -m venv venv
venv\Scripts\Activate
pip install -r requirements.txt
```

**Modeli eğitmek:**
```powershell
python src/train.py
```

**Yeni bir başvuru için tahmin yapmak:**
```powershell
python src/predict.py
```

## Geliştirme Fikirleri

- SMOTE gibi tekniklerle sınıf dengesizliğini ayrıca ele almak
- XGBoost/LightGBM gibi farklı modelleri denemek
- `purpose` değişkenindeki açıklanamayan risk sinyalini daha derin incelemek
- Basit bir web arayüzü (Streamlit/Flask) ile canlı tahmin servisi kurmak