# Bridge Dataset Design (`src/training/bridge_dataset.py`)

## 1. Purpose

The **bridge dataset** is the link between:

- The **data layer** (daily macro panels with `value / ref_period / age_days / is_new`), and  
- The **model layer** (bridge / AR models that nowcast quarterly GDP growth).

It converts "world snapshots" into **training samples** for regression models:

> Each row = one `origin_date` (prediction time)  
> Target `y` = GDP growth for the quarter containing `origin_date`  
> Features `X` = macro indicators observed as of `origin_date`

This dataset is used for:

- Model estimation (bridge regressions, AR baselines, etc.)
- Out-of-sample evaluation (walk-forward, rolling windows)
- Strategy signal generation for the All Weather framework

---

## 2. Inputs

### 2.1 Daily macro panels (`panels`)

Output of `panel.build_macro_panels()`:

```python
panels = {
  "value":      value_panel,      # DataFrame: as_of_date × series_name
  "ref_period": ref_period_panel, # DataFrame: as_of_date × series_name
  "age_days":   age_panel,        # DataFrame: as_of_date × series_name
  "is_new":     is_new_panel,     # DataFrame: as_of_date × series_name
}

	•	Index: as_of_date (daily)
	•	Columns: macro series names, e.g.:
	•	gdp_growth
	•	industrial_production
	•	payrolls
	•	retail_sales_real
	•	housing_starts
	•	philly_fed_mfg
	•	consumer_sentiment
	•	initial_claims

Baseline implementation only uses panels["value"] as features.
age_days / is_new / ref_period are reserved for future richer feature sets.

⸻

2.2 GDP calendar (gdp_cal)

Calendar-augmented DataFrame for the target series gdp_growth, obtained from:
	•	calendar.add_calendar_columns_for_series(raw_df, meta)
	•	or build_calendar_for_all_series(...)[ "gdp_growth" ]

Required columns:
	•	ref_period
	•	Quarter start date (e.g., 2014-04-01 for 2014Q2)
	•	release_date
	•	Theoretical economic release date (advance estimate) for that quarter
	•	In our baseline, available_date = release_date
	•	value
	•	GDP growth (% SAAR) for that quarter

Other columns (e.g. period_end, available_date, vintage_date) may be present but are not required by the baseline.

⸻

3. Target Construction

3.1 Collapsing vintages to one target per quarter

Because gdp_cal may contain multiple rows per ref_period (due to vintages/revisions), we define the target table as:
	•	Index: ref_period (quarter start)
	•	Columns:
	•	target_release_date = earliest release_date for that ref_period
	•	y = GDP growth value at that first release

Implementation (conceptual):
	1.	Filter out rows missing ref_period, release_date, or value
	2.	Sort by (ref_period, release_date)
	3.	For each ref_period, take the first row:
	•	This row is treated as the advance / first official release
	4.	Save to:

target_table.index  = ref_period
target_table.columns = ["target_release_date", "y"]



This corresponds to the version of GDP that markets would try to nowcast.

⸻

4. Mapping origin_date to target quarters

4.1 Defining origin_date
	•	origin_date = as_of_date from the daily macro panel:
	•	I.e., the date on which we simulate forming a nowcast.

We restrict origin_date to a time window [start_date, end_date].
	•	If start_date / end_date are None, use full range of panels["value"].index.

4.2 Mapping origin_date to target_ref_period

For each origin_date:

target_ref_period = quarter_start(origin_date)

	•	Using calendar convention:
	•	Convert origin_date to quarterly period (to_period("Q"))
	•	Take .start_time as quarter start (e.g. 2010-02-15 → 2010-01-01)

This means:

On any date inside a quarter, the model is trying to nowcast that quarter’s GDP.

⸻

5. Filtering to true “nowcast” observations

For each origin_date:
	1.	Join with target_table on target_ref_period:
	•	Get target_release_date and y
	2.	Drop rows where y is missing:
	•	Either GDP for that quarter is not in sample yet
	•	Or the quarter is beyond data availability
	3.	Apply the nowcast condition:

origin_date < target_release_date

Interpretation:
	•	We only keep dates before the target quarter’s GDP is released.
	•	Once origin_date >= target_release_date, that quarter’s GDP is known and no longer a nowcast problem.

This naturally yields:
	•	Multiple nowcast observations per quarter (one per valid origin_date)
	•	For each quarter, nowcasts become more informed as more data arrive within the quarter

⸻

6. Feature Construction (X)

6.1 Predictor series selection

From panels["value"]:
	•	Columns = macro series names
	•	Baseline rules:
	•	If predictor_series is None:
	•	Use all columns except those listed in exclude_targets
	•	Default exclude_targets = ("gdp_growth",) so that the target series is not fed back as a predictor
	•	If predictor_series is provided:
	•	Use exactly the specified series

This yields a predictor set:

predictors = [series_name_1, series_name_2, ...]

6.2 Extracting feature values at origin_date

For each origin_date, we take:

x_<series_name> = panels["value"].loc[origin_date, series_name]

This guarantees:
	•	Only information available as of origin_date is used
	•	No look-ahead beyond that day

The resulting features are stored as columns:
	•	x_industrial_production
	•	x_payrolls
	•	x_retail_sales_real
	•	x_housing_starts
	•	x_philly_fed_mfg
	•	x_consumer_sentiment
	•	x_initial_claims
	•	…

6.3 Handling missing features

For each row, we count non-missing X values:

non_missing = number of non-NaN entries among all x_<series>

If non_missing < min_non_missing_features, the row is dropped.
	•	Baseline: min_non_missing_features = 1
	•	i.e., keep any row with at least one usable macro signal

In future, this threshold can be increased to enforce stronger data density.

⸻

7. Output Schema

The final bridge dataset is a pandas.DataFrame with columns:
	•	Identification & target:
	•	origin_date
	•	Date when the nowcast is formed (as_of_date)
	•	target_ref_period
	•	Quarter start date for the target GDP (e.g. 2014-04-01)
	•	target_release_date
	•	First official release date of that quarter’s GDP
	•	y
	•	GDP growth (% SAAR) for the target quarter
	•	Features:
	•	x_<series_name> for each predictor series

Example (conceptual):

origin_date  target_ref_period  target_release_date      y    x_industrial_production  x_payrolls  ...
2010-01-05   2010-01-01         2010-04-30             3.2    ...                      ...
2010-01-06   2010-01-01         2010-04-30             3.2    ...                      ...
...
2010-04-28   2010-01-01         2010-04-30             3.2    ...                      ...
2010-04-29   2010-01-01         2010-04-30             3.2    ...                      ...
# 2010-04-30 and later: dropped for this quarter (GDP already released)

Each row is a snapshot of the information set used to nowcast a given quarter’s GDP before its release.

⸻

8. Extensions and Future Enhancements

The current design intentionally keeps the feature set minimal (only value_panel).
Future extensions can add:
	•	Lags and transformations:
	•	Include previous-day / previous-week values
	•	Log-differences, growth rates, standardized versions
	•	Age-based features:
	•	Use age_days_panel to indicate how “old” each observation is
	•	Decay weights or interaction terms value × (1 / (1 + age_days))
	•	News indicators:
	•	Use is_new_panel to flag macro release days
	•	Separate “news shock” effects from level effects
	•	Ragged-edge aware design:
	•	Explicitly encode which series are available / missing at each origin_date
	•	Use missingness patterns as features

The current schema and function boundaries are designed so that:
	•	The interface of build_bridge_dataset remains stable,
	•	Internals (how X is constructed from panels) can be enriched without breaking downstream code.
