# 📈 Time Series Forecasting for Retail Sales Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-SARIMA-green?style=for-the-badge)
![Prophet](https://img.shields.io/badge/Facebook-Prophet-blue?style=for-the-badge&logo=meta&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

**CV Project #3 — by [@umairrana99](https://github.com/umairrana99)**

*Comparing SARIMA · Facebook Prophet · LSTM on 3 years of retail sales data*

</div>

---

## Why I Built This

Every retail business lives and dies by one question — *what will sales look like next month?* Get it wrong and you're either sitting on dead inventory or watching empty shelves cost you customers. I wanted to build something that actually tackles this problem end-to-end, so I put together a full forecasting pipeline that pits three very different approaches against each other on the same dataset and lets the numbers do the talking.

This isn't just a copy-paste tutorial. I generated realistic synthetic data with proper trend, seasonality, holiday spikes, and noise — the kind of messiness you'd actually see in real retail data — and then ran SARIMA, Prophet, and an LSTM through their paces on it.

---

## Project Structure

```
📦 time-series-forecasting-retail-sales
 ┣ Time_Series_Forecasting_Retail_Sales.ipynb   ← main notebook
 ┣ plot_daily_sales.png
 ┣ plot_weekly_sales.png
 ┣ plot_yoy_comparison.png
 ┣ plot_decomposition.png
 ┣ plot_sarima_diagnostics.png
 ┣ plot_sarima_forecast.png
 ┣ plot_prophet_components.png
 ┣ plot_prophet_forecast.png
 ┣ plot_lstm_training.png
 ┣ plot_lstm_forecast.png
 ┣ plot_forecast_comparison.png
 ┣ plot_metrics_comparison.png
 ┗ README.md
```

---

## What's Actually Going On

The notebook walks through a complete end-to-end pipeline across 9 steps:

| Step | What Happens |
|------|-------------|
| **0** | Install dependencies (with a protobuf fix for Streamlit compatibility) |
| **1** | Generate 3 years of daily retail sales — trend + seasonality + holiday events + noise |
| **2** | Visualise: daily sales, weekly aggregates, year-over-year overlay |
| **3** | Decompose the series into trend/seasonal/residual + ADF stationarity test |
| **4** | Split into train (145 weeks) and test (12 weeks) |
| **5** | Fit SARIMA(1,1,1)(1,1,1,52) — classical statistical forecasting |
| **6** | Fit Facebook Prophet with custom Black Friday & Christmas holidays |
| **7** | Train an LSTM network with 12-week sliding windows |
| **8** | Compare all three models on RMSE, MAE, and MAPE |
| **9** | Summarise findings and flag where each model shines |

---

## The Data

I simulated **3 years of daily retail sales (2021–2023)** by combining four realistic signals:

| Component | Description | Size |
|-----------|-------------|------|
| **Trend** | Steady upward growth | $10,000 → $15,000 over 3 years |
| **Yearly seasonality** | Peaks in December, dips in February | ±$2,500 |
| **Weekly seasonality** | Weekends drive higher sales | ±$800 |
| **Random noise** | Normal day-to-day variation | σ = $600 |
| **Event spikes** | Black Friday · Christmas rush · New Year | +$2,000 to +$6,000 |

After generation, the daily data gets **aggregated to weekly** frequency. This smooths out daily noise and makes the seasonal patterns much cleaner for modelling — 1,095 daily rows become 157 weekly rows.

---

## Models Used

### SARIMA — The Classic
SARIMA stands for Seasonal AutoRegressive Integrated Moving Average. It's been the backbone of time series forecasting for decades and for good reason — it's interpretable, well-understood, and works well on data that's roughly linear.

I used **SARIMA(1,1,1)(1,1,1,52)** — one lag each for the AR, I, and MA terms, with full seasonal differencing at a 52-week period (yearly). Getting this to converge on weekly data takes a bit of coaxing, so I set `enforce_stationarity=False`, `enforce_invertibility=False`, and bumped `maxiter` to 200.

### Prophet — The Business-Friendly One
Facebook Prophet treats forecasting as a curve-fitting problem rather than a statistical one. It decomposes the series into trend, seasonality, and holidays, and it fits incredibly fast — we're talking seconds, not minutes. The killer feature here is the **custom holiday support**, which lets you tell the model exactly when Black Friday and Christmas happen and how long the effect lasts before and after.

### LSTM — The Deep Learner
Long Short-Term Memory networks are the go-to architecture for sequential data in deep learning. The model takes the past 12 weeks as input and predicts the next week. I used a two-layer LSTM (64 units → 32 units) with dropout to prevent overfitting, trained with early stopping so it stops automatically when validation loss stops improving.

---

## Results

All three models were evaluated on the same held-out **12-week test window**:

| Model | RMSE ($) | MAE ($) | MAPE (%) | Fit Time | Interpretable | Handles Holidays |
|-------|----------|---------|----------|----------|---------------|-----------------|
| **SARIMA** | ~8,800 | ~5,100 | ~4.6% | 2–3 min | ✅ Yes | ❌ No |
| **Prophet** | ~10,200 | ~6,400 | ~5.8% | ~10 sec | ✅ Yes | ✅ Yes |
| **LSTM** | ~12,800 | ~8,700 | ~8.0% | 1–2 min | ❌ No | Partially |

> **SARIMA wins on this dataset.** That's not surprising — the data follows a relatively linear seasonal pattern, which is exactly where SARIMA is strongest. On messier real-world data with non-linear patterns and structural breaks, LSTM would likely close the gap.

---

## Key Takeaways

**What worked:**
- All three models kept MAPE under 10%, which is generally considered solid for retail forecasting
- Prophet did a great job capturing the holiday spike shape even though its overall error was higher
- SARIMA's diagnostic plots showed clean residuals — the model genuinely captured the structure in the data
- LSTM converged quickly with early stopping (best epoch was well before 100)

**What I learned:**
- Seasonal period matters enormously for SARIMA — using `s=52` on weekly data is correct but needs a large enough training window and higher `maxiter` to converge reliably
- Prophet's `changepoint_prior_scale` has a big impact on how aggressively it tracks trend shifts
- For LSTM on time series, MinMax scaling before training and inverse-transforming predictions is essential to get sensible numbers out

---

## Getting Started

### Prerequisites

```bash
pip install "protobuf>=3.20,<7"
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels prophet tensorflow
```

### Run the Notebook

```bash
git clone https://github.com/umairrana99/time-series-forecasting-retail-sales.git
cd time-series-forecasting-retail-sales
jupyter notebook Time_Series_Forecasting_Retail_Sales.ipynb
```

Then run all cells top to bottom with **Kernel → Restart & Run All**.

---

## What's Next

A few things I want to add or try when I revisit this:

- **Hyperparameter search** — grid search over SARIMA (p, d, q) orders instead of hand-picking them
- **Ensemble forecasting** — average SARIMA + Prophet predictions and see if that beats either alone
- **Real dataset** — swap synthetic data for the [M5 Forecasting competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy) or Walmart sales data
- **External regressors** — add weather, promotion flags, or competitor pricing as features for Prophet/LSTM
- **Prediction intervals** — output confidence bands, not just point estimates

---

##Tech Stack

| Tool | Purpose |
|------|---------|
| `pandas` / `numpy` | Data manipulation and numerical computing |
| `matplotlib` / `seaborn` | Visualisation |
| `statsmodels` | SARIMA, seasonal decomposition, ADF test |
| `prophet` | Facebook Prophet model |
| `tensorflow` / `keras` | LSTM neural network |
| `scikit-learn` | Evaluation metrics, MinMax scaling |

---

## Connect

Built by **Umair Ali** — if you found this useful or have suggestions, feel free to open an issue or reach out.

[![GitHub](https://img.shields.io/badge/GitHub-umairrana99-181717?style=flat-square&logo=github)](https://github.com/umairrana99)

---

<div align="center">
<sub>⭐ If this helped you, a star would mean a lot!</sub>
</div>
