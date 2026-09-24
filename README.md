# 💧 SmartPure AI
ML-Based Water Quality & Filter Life Prediction System

## What it does
| Part | Algorithm | Inputs | Output |
|---|---|---|---|
| Water Quality | Random Forest **Classifier** | pH, TDS, turbidity | Safe / Moderate / Unsafe |
| Filter Life | Random Forest **Regressor** | TDS, turbidity, daily usage, users, filter age | Remaining days |

A Streamlit dashboard lets anyone enter values and see both predictions.

## Run it
```bash
pip install -r requirements.txt
python train_models.py      # creates dataset + trains + saves models
streamlit run app.py        # opens the dashboard in the browser
```

## Files
- `train_models.py` - dataset creation, training, evaluation, saving
- `app.py` - Streamlit dashboard
- `water_data.csv` - generated dataset (3000 rows)
- `*.pkl` - saved trained models

## Deploy free (get a public link)
1. Upload this folder to a GitHub repo.
2. Go to share.streamlit.io, sign in with GitHub, pick the repo, main file `app.py`, click Deploy.
3. You get a link like `https://yourname-smartpure.streamlit.app`.

## Simple explanation for presentation
- **Random Forest** = many decision trees (100) each vote; the majority (classifier) or average (regressor) is the answer. It resists overfitting and works well on tabular data.
- **Classifier vs Regressor**: classifier predicts a category (Safe/Unsafe); regressor predicts a number (days left).
- **Train/Test split (80/20)**: model learns on 80%, we check it on unseen 20%.
- **Metrics**: Accuracy for classifier; MAE (average error in days) and R² (0-1, closer to 1 is better) for regressor.
- **Feature importance**: shows which input affects the result most.

## Likely viva questions
1. *Why Random Forest?* Accurate, needs little tuning, handles non-linear data, gives feature importance.
2. *Where is the data from?* Synthetic data built from WHO/BIS-style limits (pH 6.5-8.5, TDS < 500 mg/L, turbidity < 5 NTU). With real company sensor data, just replace `water_data.csv` and retrain.
3. *Why is accuracy so high?* Quality labels follow clear rules, so trees learn them easily. Real data would score lower; be upfront about this.
4. *How to improve?* Real IoT sensor data, more features (hardness, chlorine), hyperparameter tuning, cross-validation.
5. *What is overfitting?* Memorising training data and failing on new data; the test split detects it.

## Limitations (say this honestly)
Data is simulated; the filter-life formula is an assumption, not measured from real filters.
