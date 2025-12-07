Below is a complete, formal, and detailed Bridge Nowcasting Model Configuration & Design Document.
It is written in English, suitable for inclusion in your repository (e.g., docs/bridge_model_design.md).
It includes:
	•	model target specification
	•	data source descriptions
	•	dataset construction rules
	•	rolling training design
	•	evaluation framework
	•	output schema
	•	configuration parameters

⸻

Bridge GDP Nowcasting Model – Configuration & Design Document (v1)

1. Overview

This document defines the formal configuration and methodological design of the GDP Nowcasting Pipeline (Version 1).
It establishes all rules and conventions that govern:
	•	the target variable
	•	the macroeconomic data sources
	•	the construction of daily panels
	•	the bridge dataset
	•	the rolling training scheme
	•	out-of-sample evaluation
	•	required output formats

The goal of this document is to ensure that every model—baseline or advanced—operates under the same, transparent specifications. No model-specific logic appears here; this document defines the global framework only.

⸻

2. Target Variable Specification

2.1 Economic Target

The model predicts quarterly U.S. Real GDP growth (SAAR, % q/q) using advance (first) release values.

Variable name

gdp_growth

Economic meaning

Quarter-over-quarter percentage change in real GDP, annualized, as published by the Bureau of Economic Analysis (BEA).

Data source
	•	Vendor: FRED
	•	Code: A191RL1Q225SBEA
	•	Source: ALFRED vintage database
	•	Vintage mode: first_release
Only the first published value for each quarter is used as the target.

Units

Percent change at an annualized rate.

Transformation

None applied:
	•	Raw target is kept in percentage units (e.g., 3.5 meaning +3.5% SAAR).
	•	Models may internally standardize values during training, but the target stored in the dataset is untransformed.

⸻

3. Data Sources and Real-Time Structure

All macroeconomic indicators are obtained from FRED or ALFRED.

3.1 Why ALFRED

ALFRED provides the full historical vintage archive:
	•	For each reference period, ALFRED records every revision.
	•	This enables true real-time simulation, avoiding look-ahead bias.

3.2 Daily Panel Construction

Each indicator has:
	•	ref_period: the period the observation refers to (month or quarter)
	•	available_date: the date on which the value first becomes observable in real time
	•	value: the observed economic indicator
	•	age_days: number of days since the latest release
	•	is_new_release: whether a new value is released on that date

The panel is constructed daily using actual available_date stamps, not publication calendars from external sources.
This ensures real-time consistency.

3.3 Forward-Fill Policy

Because macroeconomic series are not published daily:
	•	Values remain unchanged between release dates.
	•	The daily panels forward-fill the latest known information.
	•	This is the correct convention for nowcasting models.

No synthetic interpolation is performed.

⸻

4. Bridge Dataset Specification

The bridge dataset is built from:
	1.	Daily macroeconomic panels
	2.	Quarterly GDP release calendar

Each dataset row corresponds to an origin_date, the date at which the nowcast is produced.

4.1 Target Fields

Field	Meaning
origin_date	The date the nowcast is produced
target_ref_period	The quarter being predicted (start date of quarter)
target_release_date	The advance release date for that quarter
y	Observed first-release GDP growth

4.2 Target Mapping Rule

For origin date t, the model predicts the growth rate of the quarter containing t.

Formally:

target_ref_period = quarter_start(t)

This is a pure H=0 nowcast:
	•	When you are inside Q2, you nowcast Q2.
	•	No next-quarter forecasting is attempted in v1.

⸻

5. Features (Predictors)

Each macro series contributes one daily-updated predictor:

x_<series_name>

All predictors reflect real-time availability:
	•	Values are subject to release delays (release_lag_days)
	•	Missing data from early decades remains missing rather than interpolated
	•	ALFRED revision history ensures no future revisions leak into the past

Seven predictors currently included:
	1.	Industrial Production
	2.	Nonfarm Payrolls
	3.	Real Retail Sales
	4.	Housing Starts
	5.	Philly Fed Manufacturing Index
	6.	Consumer Sentiment
	7.	Initial Jobless Claims

The project supports adding or removing series through series.yaml.

⸻

6. Rolling Training Design

Version 1 uses a transparent and simple rolling scheme.

6.1 Training Window
	•	Length: 15 years
	•	For an origin date t, the training sample includes all dataset rows where:

origin_date ∈ [t - 15 years, t - 1 day]

Parameters

Parameter	Default
train_window_years	15
min_train_quarters	40


⸻

6.2 Reestimation Frequency

The model is not retrained daily.

Instead:

reestimate_freq = "M"

Interpretation:
	•	On the first origin date of each month, fit a new model using the rolling sample.
	•	Throughout the rest of that month, reuse the same coefficients while predictors evolve over days.

This greatly stabilizes the model.

⸻

6.3 Test Period

For evaluation purposes:

test_start_date = 2010-01-01
test_end_date = latest available date

Rationale
	•	Earlier real-time data (1990s–2000s) is very sparse for some series (e.g., Retail Sales).
	•	Starting in 2010 ensures a dense and consistent set of predictors.
	•	Still long enough (~15 years) for statistically meaningful results.

⸻

7. Evaluation Framework

The evaluation compares predicted vs. realized GDP using several slicing schemes.

7.1 Time-Segment Slicing

To examine robustness across macro regimes, performance is evaluated within multiple subperiods.

Suggested segments:
	1.	2010–2014
	2.	2015–2019
	3.	2020–2022 (pandemic and early recovery)
	4.	2023–present (high inflation and normalization)

Actual slicing is configurable.

⸻

7.2 Distance-to-Release Slicing

Define:

distance_to_release = (target_release_date - origin_date).days

Four buckets illustrate model performance at different stages of the quarter:

Bucket	Rule	Interpretation
Early	≥ 60 days	Start of quarter; only limited indicators available
Mid	30–59 days	Moderate information environment
Late	10–29 days	Near-complete information
Final	< 10 days	Just before GDP release

This slicing is crucial for understanding the value-added of the bridge model relative to AR baselines.

⸻

7.3 Metrics

Primary error metrics:
	1.	RMSE
	2.	MAE
	3.	Mean Bias (y_hat - y)
	4.	Hit Rate
	•	Directional correctness (sign agreement)

Optionally:
	•	R-squared (for completeness, though less meaningful in macro forecasting)

⸻

8. Output Schema

All rolling models must output a table with the following minimal schema:

Column	Meaning
model_name	Identifier of model or configuration
origin_date	Forecast date
target_ref_period	Quarter being predicted
target_release_date	Official release date
distance_to_release	Days prior to release
y	Actual GDP first release
y_hat	Model’s prediction
train_start, train_end	Boundaries of training window

This table is the universal interface consumed by the evaluation engine.

⸻

9. Configuration Summary (rolling.yaml)

Example configuration file corresponding to this document:

# rolling.yaml — Version 1 (H=0 Nowcast)

target:
  name: gdp_growth
  vintage_mode: first_release
  scale: percent

training:
  train_window_years: 15
  min_train_quarters: 40
  reestimate_freq: "M"

evaluation:
  test_start_date: "2010-01-01"
  test_end_date: null   # auto = latest
  time_segments:
    - {start: "2010-01-01", end: "2014-12-31", name: "post_gfc"}
    - {start: "2015-01-01", end: "2019-12-31", name: "expansion_pre_covid"}
    - {start: "2020-01-01", end: "2022-12-31", name: "pandemic"}
    - {start: "2023-01-01", end: null,        name: "high_inflation"}

distance_buckets:
  - {name: early, min_days: 60, max_days: null}
  - {name: mid,   min_days: 30, max_days: 59}
  - {name: late,  min_days: 10, max_days: 29}
  - {name: final, min_days: 0,  max_days: 9}


