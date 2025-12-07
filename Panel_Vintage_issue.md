📄 GDPNowcast — Vintage Handling & Availability Logic

1. Background: Why ALFRED Creates “Late Appearance” Artifacts

Our raw macroeconomic data come from two sources:
	•	FRED → latest-revised time series
	•	ALFRED → historical “vintage” snapshots of data as they were known on past dates

In theory:
	•	ALFRED stores the value of each macro series for each vintage_date
	•	This allows us to reconstruct exactly what information was available on any as_of_date, ideal for real-time nowcasting research

However, ALFRED does NOT store true real-time vintages for very old periods.

For many macro series:

ref_period	vintage_date stored in ALFRED
1980Q1	2014-09-26 (first ALFRED snapshot)
1980Q2	2014-09-26
…	2014-09-26

Meaning:

ALFRED first began recording historical GDP / IP / Payrolls vintages around 2014, so old historical observations receive a vintage_date decades later than the actual release.

If we enforce:

available_date = max(release_date, vintage_date)

then:
	•	A value with ref_period = 1980Q1 and vintage_date = 2014-09-26
	•	Will be considered unavailable before 2014

→ This breaks any real-time backtest before 2014.

This is expected behavior from ALFRED but not usable for our modeling goal.

⸻

2. Engineering Decision: Baseline Assumption for This Project

❗ What we want for nowcast modeling and backtesting

We want:
	•	A usable panel back to 1980
	•	No look-ahead bias
	•	A clean daily time series for bridge / AR / DFM models

To achieve that, we adopt:

✅ Baseline Rule

available_date = release_date

We do NOT delay availability when vintage_date is far later.

Rationale:
	•	The GDP data was in fact publicly available in 1980 (first release)
	•	ALFRED’s late-vintage snapshots do NOT reflect economic reality
	•	Our modeling goal needs 40+ years of data; otherwise the sample collapses
	•	This baseline matches most academic and practitioner nowcasting systems that do not attempt deep revision modeling

What we still keep

We still store:
	•	vintage_date
	•	Full vintage panel
	•	Revision histories

These remain available for future extensions where stricter real-time reconstruction is required.

⸻

3. What Would Strict Real-Time Mode Require?

Once the baseline system is stable, we can upgrade to true real-time vintage logic.

In strict mode, availability should be:

available_date = max(release_date, vintage_date)

BUT this requires:

✔ 1. Availability of true first-release vintages for old data

ALFRED currently does not provide:
	•	GDP advance estimates from 1980
	•	Payrolls first releases from 1970s
	•	Industrial Production first releases, etc.

This must be sourced from:
	•	Philadelphia Fed Real-Time Data Research Center (RTDRC)
	•	BEA historical PDFs / archives
	•	BLS “First Friday” historical releases
	•	Manually digitized datasets

This is a different level of complexity.

⸻

4. Practical Upgrade Path for Future Versions

When you want to turn on “Super Strict Real-Time Mode”, the project can evolve this way:

⸻

Step 1 — Switch calendar logic

Modify in calendar.py:

available_date = release_date    # baseline

→

available_date = max(release_date, vintage_date)  # strict mode

Add environment flag or config option:

strict_realtime: true


⸻

Step 2 — Import true first-release vintages

Add new loaders:

src/data/rtdrc_loader.py      # Philadelphia Fed RTDSM datasets
src/data/bea_first_release.py # scraped or archived BEA advance GDP

Use these datasets to replace ALFRED-imputed vintage dates for early years.

⸻

Step 3 — Construct a proper real-time panel

Implement Giannone-style jagged-edge reconstruction:
	•	Each as_of_date sees only data with vintage_date ≤ as_of_date
	•	Missing data handled via ragged-edge interpolation or bridge design
	•	Allows fully rigorous real-time nowcasting experiment

⸻

Step 4 — Add tests verifying no leakage

Strict tests:
	•	Ensure no value enters panel before its real-world availability
	•	Validate ref_period, release_date, vintage_date alignment
	•	Mark structural breaks in vintage history

⸻

5. Summary of Our Decision (for documentation)

### 🟩 Baseline mode (current implementation)
	•	available_date = release_date
	•	Full history (1980+) remains usable
	•	Zero look-ahead bias with respect to economic release
	•	Does not attempt to replicate exact historical vintage availability
	•	Suitable for nowcasting model development, strategy signal building, economic regime detection

🟥 Strict real-time mode (future optional)
	•	available_date = max(release_date, vintage_date)
	•	Requires accurate historical first-release vintages
	•	Data becomes unusable before ~2014 unless external datasets added
	•	Suitable for academic-grade real-time evaluation research

