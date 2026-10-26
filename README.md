# Rossmann Sales Forecasting: SARIMA vs. LSTM vs. GRU

A comparative study of a classical seasonal time-series model (**SARIMA**) and two recurrent neural networks (**LSTM** and **GRU**) for retail sales forecasting on the Rossmann Store Sales dataset.

> Graduate research project, *A Comparative Study of SARIMA and Recurrent Neural Networks (RNN) for Enhanced Sales Forecasting Accuracy*.
> Cairo University, Faculty of Graduate Studies for Statistical Research, Department of Data Science, 2025/2026.
> Prepared by Naira Magdy Abd Elhamied. Supervised by Prof. Dr. Mohammed Abdel-Hamied Sabry.

---

## Key finding

The classical model beat both deep-learning models on the held-out test set. SARIMA's lead is large: its test RMSE is about 5.5× lower than GRU's and about 6.6× lower than LSTM's.

| Model | Type | Test RMSE (log + Min-Max scaled) |
|---|---|---|
| **SARIMA** | Classical statistical | **0.13** |
| GRU | Deep learning (RNN) | 0.72 |
| LSTM | Deep learning (RNN) | 0.86 |

The weekly Rossmann series has strong, regular seasonality and only about 130 weekly observations. That is the setting SARIMA was designed for, and it is short for training recurrent networks with many parameters. GRU beat LSTM, which fits its lighter gating mechanism on a series of this length. Both RNNs under-predict the highest-sales days (Christmas and major promotions). The project report also gives MAE and MAPE (SARIMA 4.2% MAPE vs. about 20% for GRU and 23% for LSTM); see [Notes on the results](#notes-on-the-results).

---

## Dataset

[Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) (Kaggle): daily sales for **1,115 German drug stores** from January 2013 to July 2015 (1,017,209 rows).

| File | Contents |
|---|---|
| `train.csv` | Daily `Sales`, `Customers`, `Open`, `Promo`, `StateHoliday`, `SchoolHoliday`, `DayOfWeek`, per store |
| `store.csv` | Store metadata: `StoreType` (a–d), `Assortment`, `CompetitionDistance`, competition-open dates, `Promo2` and `PromoInterval` |

The dataset is not included in this repository. Download it from Kaggle and place the files like this:

```
dataset/
├── train.csv
└── store.csv
```

---

## Pipeline

### 1. Data cleaning and feature engineering
- Merge `train.csv` and `store.csv` on `Store` (left join).
- Extract `Year`, `Month`, `Day` and `WeekofYear` from the date, and add a `SalesPerCustomer` feature.
- Parse `PromoInterval` into 12 monthly binary flags (`Promo_Jan` … `Promo_Dec`).
- Fill a missing `CompetitionDistance` with 200,000 m (no nearby competitor). Impute the remaining competition and Promo2 date gaps with scikit-learn's `IterativeImputer`.
- Standardise `StateHoliday` and one-hot encode `StoreType`, `Assortment` and `StateHoliday`.
- Detect outliers with the IQR rule (2.62% of rows). They are **kept**, because the high-sales days (Christmas, big promotions) are real demand spikes the models need to learn.
- About 172,871 rows have zero sales, mostly Sundays when stores are closed. Closed days are excluded from the neural-network error metrics.

### 2. Exploratory data analysis
Sales by year and month, day-of-week effects, promotion impact, store-type comparisons, holiday effects, open vs. closed days, and a correlation heatmap. Key observations:
- `Customers` and `Sales` are strongly correlated (r ≈ 0.82).
- Promotions raise both footfall and revenue per visit.
- Store type `b` has the highest average sales and the closest competitors, which motivates the store embeddings in the neural models.

### 3. Stationarity and seasonality
Rolling mean/std plots and the Augmented Dickey-Fuller test on four representative stores (all p < 0.001). Additive seasonal decomposition (period = 365) and ACF/PACF plots show strong, regular seasonality.

### 4. Models

**SARIMA** (statsmodels `SARIMAX`)
- Fitted on the **weekly-resampled** mean sales series, with a log1p transform followed by Min-Max scaling.
- Chronological **60 / 20 / 20** train / validation / test split (81 / 27 / 27 weeks).
- Grid search over `(p,d,q)` and `(P,D,Q)` ∈ {0,1} with seasonal periods S ∈ {7, 14, 21, 29, 52}. The final model is the one with the lowest validation RMSE, which is **SARIMA(0,1,1)(1,0,0)[52]**. It is refit on train + validation and evaluated on the test set.

**LSTM and GRU** (PyTorch, global models trained across all stores)
- Daily data, 60-step sliding windows, 128 hidden units, 2 recurrent layers.
- A 10-dimensional store embedding is concatenated with the continuous features.
- Adam optimiser with `ReduceLROnPlateau`, gradient clipping, mixed precision (AMP) on GPU, and early stopping on validation MSE (patience 5 for LSTM, 8 for GRU).
- The same chronological 60 / 20 / 20 split by date. Predictions are inverse-transformed (`MinMaxScaler`, then `expm1`) back to euros for reporting.

### 5. Evaluation
Test-set RMSE for all three models, plus residual analysis (histograms and residuals over time) and training/validation loss curves for the RNNs.

---

## Repository contents

| File | Description |
|---|---|
| `Rossman_Sales_Forecasting_DL_1.ipynb` | End-to-end notebook: cleaning, EDA, SARIMA, LSTM, GRU, comparison |
| `SARIMA_RNN_Final_Proposal.pdf` | Full written report (literature review, methodology, results) |
| `dataset/` | Rossmann `train.csv` and `store.csv` (**you add these**) |

---

## Getting started

### Requirements
Python 3.10+ (the notebook was run on 3.12) and a GPU for the LSTM/GRU training (the notebook was run on a CUDA GPU; it falls back to CPU, but much more slowly).

```bash
pip install pandas numpy matplotlib seaborn plotly dash scikit-learn statsmodels torch torchmetrics
```

### Run
1. Download the dataset and place `train.csv` and `store.csv` in `dataset/` (see above).
2. Open the notebook in Jupyter, VS Code, or Google Colab (GPU runtime recommended).
3. Run all cells top to bottom. The SARIMA grid search and the RNN training are the slowest steps.

---

## Notes on the results

A few things to know when comparing the notebook with the written report:

- **Metric scale.** The RMSE values above (0.13 / 0.72 / 0.86) are on the log + Min-Max scaled scale. The SARIMA figure is for the aggregated *weekly* series. The LSTM/GRU figures are for *daily* store-level predictions. The two sets of numbers are on different granularities, which the report lists as a limitation. On the euro scale, the notebook's LSTM and GRU test RMSE are both about €7,065.
- **MAE and MAPE.** The report gives MAE and MAPE for each model. The notebook computes only RMSE, so those two metrics are not reproduced by running it as-is.
- **Details that differ between the report and the notebook.** The notebook's code uses SARIMA(0,1,1)(1,0,0)[52] (selected by validation RMSE), a 60-day window, a learning rate of 0.01, a batch size of 512 and 47 input features (37 continuous plus the 10-dimensional store embedding). The report describes SARIMA(1,1,1)(1,1,1)[52] (selected by AIC), a 30-day window, a learning rate of 1e-3, a batch size of 256 and 19 input features. If you publish this repo, update one of the two so they agree.
- **Reproducibility.** Results can vary between runs because of GPU non-determinism and the random seed. Only the `IterativeImputer` is seeded in the notebook.

---

## Limitations and future work

**Limitations**
- A single dataset (German drugstores, 2013–2015), so results may not transfer to other retail formats.
- The SARIMA and RNN models are trained at different temporal granularities (weekly vs. daily).
- No exhaustive hyperparameter search for the neural networks.
- No external macroeconomic variables, and only one-step-ahead evaluation (multi-step accuracy is not assessed).

**Future work**
- Extend SARIMA to **SARIMAX** using `Promo` and holiday indicators as exogenous regressors.
- Train store-specific neural models.
- Build **hybrid SARIMA–GRU/LSTM** models where a neural network corrects SARIMA's residuals.

---

## Acknowledgements

Dataset: Rossmann Store Sales, Kaggle. Methods and motivation follow the studies cited in the project report (Makridakis et al. 2022, Hewamalage et al. 2023, Ma et al. 2023, and others).
