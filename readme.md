# GDP & CPI Nowcast System for All Weather Strategy

A lightweight, engineering-first Nowcasting system designed to generate robust macroeconomic growth and inflation signals. This project is built specifically to support **All Weather / Risk Parity** investment strategies by providing timely, point-in-time estimates of US Real GDP and CPI (Headline & Components).

## 🚀 Key Features

* **Strict Point-in-Time Simulation (Vintage Replay):**
    * Prevents look-ahead bias by rigorously simulating data availability at every historical time step.
    * Handles the **"Ragged Edge"** problem where high-frequency data (e.g., Oil, Claims) is available before low-frequency official releases (e.g., GDP, CPI).
* **Structured Inflation Signals:**
    * Does not just predict "Inflation"; it breaks it down into **Headline, Core, Energy, Food, and Shelter** to identify the *source* of the shock (Supply vs. Demand).
* **Anti-Fragile Data Engineering:**
    * **Robust Scaling & Clipping:** Automatically clips extreme outliers (±4σ) to prevent black swan events (like COVID-19 2020) from breaking linear models.
    * **Smoothing:** Applies 3-month rolling averages to feature inputs to filter out monthly noise.
* **Meta-Data for Strategy Execution:**
    * Outputs **Data Completeness** scores and **Attribution** (Hard vs. Soft data Z-scores) to tell the downstream strategy *how much to trust* the current signal.

## 📂 System Architecture

```text
nowcast/
├── config/
│   └── series.yaml        # Single Source of Truth: Definitions for all macro indicators.
├── data/
│   └── fred.py            # Data Provider with local caching (Offline-First).
├── features/
│   ├── panel_builder.py   # ETL: Resampling, Alignment, Smoothing, and Clipping.
│   ├── asof_dataset.py    # The Brain: Vintage logic, Lag masking, and Imputation.
│   └── targets.py         # Target construction (GDP QoQ, CPI MoM).
├── models/
│   └── ridge.py           # Ridge Regression with CV (Primary Model).
├── pipeline/
│   ├── run_gdp_nowcast.py # Pipeline for Quarterly GDP.
│   └── run_cpi_nowcast.py # Pipeline for Monthly CPI & Sub-components.
└── export/
    └── macro_features.py  # Orchestrator: Merges GDP & CPI into a unified signal table.
````

## 🛠️ Installation & Setup

### 1\. Prerequisites

Ensure you have Python 3.10+ installed. Install dependencies:

```bash
pip install pandas numpy scikit-learn scipy fredapi python-dotenv tqdm matplotlib
```

### 2\. API Configuration

This project uses the St. Louis Fed (FRED) API.

1.  Request an API key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html).
2.  Create a `.env` file in the root directory:
    ```env
    FRED_API_KEY=your_32_character_api_key_here
    ```

## ⚡ Usage

### Daily Routine (Production)

To generate the latest signals for your trading strategy, simply run the daily entry point. This will update data, re-train models, and generate the CSV.

```bash
python run_daily.py
```

  * **Output:** `data/output/gdp_nowcast_latest.csv`

### Updating Data

To force a download of the latest data from FRED (bypassing local cache):

```bash
python scripts/data_update.py
```

### Running Backtests (Research)

To visualize the historical performance of the models:

```bash
# Run GDP Nowcast Backtest
python -m nowcast.pipeline.run_gdp_nowcast

# Run CPI Nowcast Backtest
python -m nowcast.pipeline.run_cpi_nowcast
```

## 🧠 Methodology

### 1\. The "Ragged Edge" & Publication Lags

The system hard-codes publication delays to ensure realistic backtests (`nowcast/features/asof_dataset.py`).

  * **Real-time (Lag=0):** WTI Oil, Gasoline, Breakeven Rates, Consumer Sentiment. *These update the model immediately within the month.*
  * **Lagged (Lag=15+):** CPI, PPI, Industrial Production. *These only update the model mid-next-month.*

### 2\. Modeling Approach

We utilize **Ridge Regression (L2 Regularization)** as the primary learner.

  * **Why Ridge?** Macroeconomic data has high multicollinearity (e.g., Nonfarm Payrolls and Industrial Production move together). Ridge handles this gracefully without zeroing out useful signals like Lasso.
  * **Why not complex ML?** For low-signal-to-noise macro data, linear models with strong regularization often outperform complex non-linear models (Random Forest/LSTM) out-of-sample.

### 3\. Signal Construction

  * **Regime Logic:**
      * **Growth Regime:** Based on GDP Nowcast Z-Score (\>0.5 or \<-0.5).
      * **Inflation Regime:** Based on CPI Headline Nowcast Z-Score.
  * **Smoothing:** The final output signal uses a **5-day moving average** to prevent single-day data releases (and subsequent revisions) from triggering "whipsaw" trades.

## 📊 Output Data Dictionary

The generated CSV contains more than just predictions. It enables **Tactical Asset Allocation**.

| Column Name | Description | Strategy Use Case |
| :--- | :--- | :--- |
| `gdp_nowcast` | Forecasted Real GDP QoQ (Annualized). | Primary Growth Signal. |
| `cpi_headline_nowcast` | Forecasted CPI MoM (Annualized). | Primary Inflation Signal (TIPS proxy). |
| `data_completeness` | % of indicators actually released for the current period (0.0 - 1.0). | **Position Sizing**: Low completeness (early month/quarter) = Lower position size. |
| `hard_data_z` | Combined Z-Score of "Hard" Data (Payrolls, IndPro). | **Confirmation**: If Soft Data crashes but Hard Data is stable, do not short risk assets aggressively. |
| `soft_data_z` | Combined Z-Score of "Soft" Data (Sentiment, PMIs). | **Sentiment**: Leading indicator for tactical equity adjustments. |
| `cpi_energy_nowcast` | Forecasted Energy component. | **Commodity Allocation**: High Energy + Low Core = Supply Shock (Long Commodities). |
| `cpi_shelter_nowcast` | Forecasted Shelter component. | **Duration Risk**: Rising Shelter = Sticky Inflation (Short Nominal Bonds). |

## 📝 Configuration (`series.yaml`)

To add new economic indicators, edit `nowcast/config/series.yaml`.

  * **Target:** The variable to predict.
  * **Transform:** `pct_mom` (Month-over-Month), `diff` (Difference), etc.
  * **Frequency:** Monthly (`monthly`), Weekly (`weekly`), or Daily (`daily`).

*Note: High-frequency data (Daily/Weekly) is automatically aggregated to monthly means within the pipeline.*

```