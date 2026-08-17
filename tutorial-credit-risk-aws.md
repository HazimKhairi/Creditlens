# CreditLens MY: Credit Risk Scoring on AWS (Experian-style)

Tutorial end-to-end untuk projek loan default prediction dengan explainable reason codes.
Target siap: 2-3 weekend. Setiap phase ada deliverable sendiri, boleh stop dan sambung.

**Kenapa projek ni:** dia isi 3 gap dalam resume awak sekarang: SageMaker endpoint deployment,
model monitoring, dan pengalaman data relational berskala besar. Plus SHAP reason codes tu
signature industri kredit (adverse action reasons, dituntut regulator).

---

## Phase 0: Persediaan (buat HARI INI, sebab ada waiting time)

### 0.1 Request SageMaker quota

Kali lepas quota training instance awak 0. Endpoint pun sama. Request sekarang sebab
approval ambil 1-3 hari:

1. AWS Console > Service Quotas > Amazon SageMaker
2. Request increase untuk:
   - `ml.m5.large for endpoint usage` -> 1
   - `ml.m5.large for training job usage` -> 1 (optional, boleh train dalam Studio notebook macam dulu)
3. Alternatif kalau ditolak: **Serverless Inference** (quota berasingan, selalunya dah ada,
   dan bayar per-request je, sesuai untuk demo portfolio)

### 0.2 Kaggle API

```bash
pip install kaggle
# Kaggle.com > Settings > API > Create New Token, letak kat ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Pergi ke page competition dan klik "Join Competition" dulu (wajib sebelum boleh download):
https://www.kaggle.com/competitions/home-credit-default-risk

### 0.3 Download dataset

```bash
mkdir -p ~/creditlens/data && cd ~/creditlens/data
kaggle competitions download -c home-credit-default-risk
unzip home-credit-default-risk.zip
```

Files yang penting (± 2.7GB unzipped):

| File | Isi | Rows |
|---|---|---|
| application_train.csv | Permohonan + label TARGET (1 = default) | 307k |
| bureau.csv | Rekod credit bureau (macam CTOS/Experian) | 1.7M |
| bureau_balance.csv | Baki bulanan setiap rekod bureau | 27M |
| previous_application.csv | Permohonan terdahulu dengan Home Credit | 1.6M |
| installments_payments.csv | Sejarah bayaran ansuran | 13.6M |
| credit_card_balance.csv | Baki kad kredit bulanan | 3.8M |
| POS_CASH_balance.csv | Baki POS/cash loan bulanan | 10M |

Ini sebab kenapa Glue/PySpark ada justifikasi: 7 table, berjuta rows, kena join dan
aggregate ke satu row per applicant.

---

## Phase 1: Data ke S3 (versi web console)

### 1.1 Cipta bucket

1. Console > taip **S3** kat search bar > tekan enter
2. Klik butang oren **Create bucket**
3. Bucket name: pilih nama unik global, huruf kecil sahaja (corak: `creditlens-namaawak-tahun`)
4. Region: pastikan **ap-southeast-1 (Singapore)**, sama dengan region kerja awak
5. SEMUA setting lain biar default. "Block all public access" kekal ON (data kewangan)
6. Scroll bawah > **Create bucket**

### 1.2 Buat struktur folder

1. Klik nama bucket yang baru dicipta
2. Klik **Create folder** > namakan `raw` > Create folder
3. Ulang untuk `features` dan `models`

(Nota: "folder" S3 sebenarnya prefix je, tapi console layan dia macam folder biasa.)

### 1.3 Upload data

1. Klik masuk folder **raw/**
2. Klik **Upload** > **Add files**
3. Pilih 8 CSV dari `~/creditlens/data/` (SEMUA KECUALI `sample_submission.csv`, `HomeCredit_columns_description.csv`, dan file .zip). Kenapa exclude: dua file tu bukan data training, upload cuma tambah kos dan sepah
4. Scroll bawah > **Upload**. 2.6GB, jadi biarkan tab tu terbuka sampai siap (browser upload akan buat multipart sendiri)
5. Checkpoint: lepas siap, folder raw/ patut ada **8 objek**

Kalau browser upload menyeksa atau putus, alternatif CLI satu baris (dari Terminal):
`cd ~/creditlens/data && aws s3 sync . s3://NAMA-BUCKET/raw/ --exclude "*.zip" --exclude "sample_submission.csv" --exclude "HomeCredit_columns_description.csv"`

## Phase 2: Feature engineering dalam Glue (versi web console)

Corak yang sama untuk setiap child table: **groupBy applicant, aggregate, join balik ke application**.

### 2.0 Setup job dalam Glue Studio (setiap klik)

1. Console > search **AWS Glue** > masuk
2. Sidebar kiri > **ETL jobs** > klik **Script editor** (BUKAN Visual ETL, script editor bagi kawalan penuh dan tak trigger sesi preview yang berbayar)
3. Engine: **Spark** > Options: **Start fresh** > **Create script**
4. Sebelum tulis code, pergi tab **Job details** dan set:
   - **Name:** `creditlens-features`
   - **IAM Role:** kalau takde, klik dropdown > kalau kosong, buka tab baru ke IAM:
     IAM > Roles > **Create role** > Trusted entity: **AWS service** > Use case: pilih **Glue** > Next >
     tick polisi **AmazonS3FullAccess** dan **AWSGlueServiceRole** > Next > namakan `glue-creditlens-role` > Create.
     Balik ke tab Glue, refresh dropdown, pilih role tu
   - **Glue version:** biar default (4.0 atau 5.0)
   - **Worker type:** G 1X
   - **Requested number of workers:** 2 (paling minimum, cukup untuk 2.6GB)
   - **Job timeout:** tukar ke 30 minit (penghadang kos kalau job sangkut)
5. **Save** (atas kanan). Lepas ni semua kerja kat tab **Script**

### 2.1 Kitaran kerja awak

Tulis code kat tab Script > **Save** > **Run** > pergi tab **Runs** > tengok status.
Klik run > **Output logs** untuk print output, **Error logs** kalau merah. Setiap run 2 workers
ambil beberapa minit dan kos berapa sen sahaja, jadi jangan takut nak iterate.

Checkpoint kos: satu run G.1X 2 workers x 10 minit lebih kurang USD 0.15.

### 2.1 Contoh: aggregate bureau.csv

```python
from pyspark.sql import functions as F

app = spark.read.csv("s3://creditlens-<nama>/raw/application_train.csv", header=True, inferSchema=True)
bureau = spark.read.csv("s3://creditlens-<nama>/raw/bureau.csv", header=True, inferSchema=True)

bureau_agg = bureau.groupBy("SK_ID_CURR").agg(
    F.count("*").alias("bureau_loan_count"),
    F.sum(F.when(F.col("CREDIT_ACTIVE") == "Active", 1).otherwise(0)).alias("bureau_active_loans"),
    F.sum("AMT_CREDIT_SUM_DEBT").alias("bureau_total_debt"),
    F.sum("AMT_CREDIT_SUM").alias("bureau_total_credit"),
    F.max("CREDIT_DAY_OVERDUE").alias("bureau_max_overdue_days"),
    F.avg("DAYS_CREDIT").alias("bureau_avg_loan_age"),
)

df = app.join(bureau_agg, "SK_ID_CURR", "left")
```

### 2.2 Feature domain kredit yang WAJIB ada (ni yang interviewer cari)

```python
df = (df
    .withColumn("debt_to_income", F.col("bureau_total_debt") / F.col("AMT_INCOME_TOTAL"))
    .withColumn("credit_to_income", F.col("AMT_CREDIT") / F.col("AMT_INCOME_TOTAL"))
    .withColumn("annuity_to_income", F.col("AMT_ANNUITY") / F.col("AMT_INCOME_TOTAL"))
    .withColumn("employed_years", -F.col("DAYS_EMPLOYED") / 365.25)
    .withColumn("age_years", -F.col("DAYS_BIRTH") / 365.25)
    .withColumn("credit_utilization", F.col("bureau_total_debt") / (F.col("bureau_total_credit") + F.lit(1)))
)
```

Gotcha dataset ni: `DAYS_EMPLOYED` ada nilai sentinel 365243 (maknanya "tak bekerja/pencen").
Tukar ke null dulu sebelum kira, kalau tak, feature awak rosak senyap:

```python
df = df.withColumn("DAYS_EMPLOYED",
    F.when(F.col("DAYS_EMPLOYED") == 365243, None).otherwise(F.col("DAYS_EMPLOYED")))
```

### 2.3 Ulang untuk previous_application dan installments (pilih 2-3 table je, tak perlu semua 7)

Aggregate berguna: bilangan permohonan lepas yang ditolak, ratio bayaran lewat,
purata hari lewat bayar. Then tulis output:

```python
df.write.mode("overwrite").parquet("s3://creditlens-<nama>/features/")
```

**Deliverable Phase 2:** satu Parquet, satu row per applicant, ~50-80 features.
Ini dah cukup untuk satu post LinkedIn ("feature engineering belongs in the ETL layer, part 2").

---

## Phase 3: Training dalam SageMaker Studio (versi web console)

Cara masuk (awak dah pernah buat masa demand forecasting):

1. Console > search **SageMaker AI** > **Studio** (sidebar) > **Open Studio**
2. Sidebar Studio > **JupyterLab** > pilih space awak atau **Create JupyterLab space** >
   namakan `creditlens` > instance **ml.t3.medium** (murah, cukup) > **Run space** > tunggu > **Open**
3. Dalam JupyterLab, buka notebook baru. Baca Parquet features terus dari S3:
   `df = pd.read_parquet("s3://NAMA-BUCKET/features/")` (pandas dalam Studio dah ada akses S3)
4. Install yang tiada: `%pip install lightgbm shap`
5. PENTING: bila habis sesi, sidebar JupyterLab spaces > **Stop space**. Billing jalan selagi dia hidup

## Phase 3 (sambungan): resipi training

### 3.1 Split yang betul

Dataset ni takde tarikh permohonan yang boleh dipakai untuk time split, so guna
**stratified split** (kekalkan ratio default dalam train/test):

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
```

### 3.2 Class imbalance: TARGET=1 cuma ~8%

```python
import lightgbm as lgb
model = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),  # handle imbalance
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
          callbacks=[lgb.early_stopping(100)])
```

### 3.3 Metrik industri kredit (JANGAN report accuracy)

```python
from sklearn.metrics import roc_auc_score
proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba)
gini = 2 * auc - 1
print(f"AUC: {auc:.4f}  Gini: {gini:.4f}")
```

Benchmark: AUC 0.75+ dah bagus untuk dataset ni (pemenang Kaggle ~0.80).
Accuracy 92% tu bohong, model yang predict "semua orang bayar" pun dapat 92%.

### 3.4 Baseline dulu

Sebelum LightGBM, run logistic regression. Kalau LightGBM tak beat logistic dengan
margin bermakna, sesuatu tak kena. Report kedua-dua dalam README.

### 3.5 Track dengan MLflow (dah ada dalam SageMaker Studio)

Log AUC/Gini setiap experiment, macam PropIntel dulu.

---

## Phase 4: Skor 300-850 + reason codes (bahagian Experian)

### 4.1 Kalibrasi probability jadi credit score

```python
import numpy as np

def probability_to_score(p_default, base_score=650, base_odds=50, pdo=40):
    """Points to Double the Odds (PDO) scaling, standard scorecard industri."""
    odds = (1 - p_default) / np.clip(p_default, 1e-6, 1)
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return np.clip(offset + factor * np.log(odds), 300, 850).round().astype(int)
```

### 4.2 Reason codes dengan SHAP (adverse action reasons)

```python
import shap
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_applicant)

def reason_codes(shap_row, feature_names, top_n=4):
    """Pulangkan faktor yang paling menaikkan risiko, macam surat penolakan bank."""
    order = np.argsort(shap_row)[::-1]  # positive SHAP = naikkan risiko default
    return [(feature_names[i], round(float(shap_row[i]), 4)) for i in order[:top_n]]
```

Output contoh yang awak nak tunjuk dalam demo:

```
Score: 512 (Declined, cutoff 580)
Kenapa:
  1. debt_to_income tinggi (0.68 vs median 0.31)
  2. bureau_max_overdue_days: 47 hari
  3. employed_years rendah: 0.8 tahun
  4. previous_refused_ratio: 2 dari 3 permohonan lepas ditolak
```

Ini ayat penting untuk README dan interview: "Equal Credit Opportunity Act dan
garis panduan BNM sama-sama menuntut sebab penolakan yang boleh diterangkan.
Skor tanpa reason codes tak boleh dipakai dalam produksi kredit."

---

## Phase 5: Deploy endpoint (gap #1 awak)

Bila quota approved. Guna Serverless Inference kalau real-time quota masih 0:

```python
from sagemaker.sklearn import SKLearnModel  # atau simpan LightGBM sebagai model.tar.gz

# Cara paling mudah: simpan model + kod inference, deploy serverless
from sagemaker.serverless import ServerlessInferenceConfig
serverless_config = ServerlessInferenceConfig(memory_size_in_mb=2048, max_concurrency=5)
predictor = sm_model.deploy(serverless_inference_config=serverless_config)
```

Test endpoint dengan satu applicant, dan screenshot response tu, itu bukti "deployed
on SageMaker" untuk post dan interview.

Alternatif kalau quota semua buntu: FastAPI + Docker macam PropIntel, tapi cuba habis-
habisan dapatkan endpoint SageMaker dulu sebab itu keyword yang awak belum ada.

**Kos:** serverless inference bayar per request (sen je untuk demo). Kalau guna
real-time endpoint, `ml.t2.medium` ~USD 0.06/jam, **DELETE endpoint lepas demo**:

```python
predictor.delete_endpoint()
```

---

## Phase 6: Monitoring (gap #2 awak)

Pilihan ringan yang cukup untuk portfolio: **Evidently** (open source) untuk data drift
report, banding distribusi features antara training set dan requests baru:

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=X_train.sample(5000), current_data=X_new)
report.save_html("drift_report.html")
```

Pilihan penuh AWS: SageMaker Model Monitor (perlu real-time endpoint + data capture).
Buat kalau quota ada; kalau tak, Evidently dah melayakkan awak sebut "model monitoring".

---

## Phase 7: Kemas dan publish

Checklist sebelum umum:

- [ ] Repo public `creditlens` di GitHub: README tulis TANGAN SENDIRI
      (problem, data, architecture diagram, metrics AUC/Gini, reason codes demo, cara run)
- [ ] Jangan commit data Kaggle (licensing), letak arahan download je. `.gitignore` data/
- [ ] Screenshot: Glue job graph, MLflow experiments, endpoint response, drift report
- [ ] Update resume: bullet baru bawah Own Business + projek dalam HIGHLIGHTED PROJECT
      (keyword baru yang jadi halal: SageMaker endpoint, model monitoring, credit risk,
      class imbalance, AUC/Gini)
- [ ] Update hazimdev.com projects + GitHub profile README
- [ ] Kandungan (satu projek = 4-5 post value-first):
      1. "Accuracy 92% pada credit data ialah penipuan" (imbalance + AUC/Gini)
      2. "Reason codes: kenapa credit scoring wajib explainable" (SHAP + regulasi)
      3. "Join 7 tables, 27 juta rows: kenapa feature engineering duduk dalam ETL"
      4. "Deploy model pertama saya ke SageMaker endpoint, ini kosnya"
      5. YouTube ep 2: full walkthrough

---

## Anggaran kos AWS keseluruhan

| Item | Anggaran |
|---|---|
| S3 (3GB) | < USD 0.10/bulan |
| Glue job runs (development, ~10 runs) | USD 3-8 |
| SageMaker Studio notebook (ml.t3.medium) | ~USD 0.05/jam, tutup bila tak guna |
| Serverless inference demo | < USD 1 |
| **Total projek** | **~USD 5-15** |

Set billing alert USD 20 kat AWS Budgets sebelum mula. Selamat membina.
