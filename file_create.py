import os

project_structure = [
    "gdp_nowcast/config/series.yaml",
    "gdp_nowcast/config/bridge.yaml",
    "gdp_nowcast/config/rolling.yaml",
    "gdp_nowcast/config/signal.yaml",

    "gdp_nowcast/data/raw/",
    "gdp_nowcast/data/processed/",
    "gdp_nowcast/data/results/",

    "gdp_nowcast/src/data/series_config.py",
    "gdp_nowcast/src/data/loaders.py",
    "gdp_nowcast/src/data/calendar.py",
    "gdp_nowcast/src/data/panel.py",
    "gdp_nowcast/src/data/validators.py",

    "gdp_nowcast/src/features/bridge_base.py",
    "gdp_nowcast/src/features/bridge_stage.py",

    "gdp_nowcast/src/models/base.py",
    "gdp_nowcast/src/models/bridge.py",
    "gdp_nowcast/src/models/ar_baseline.py",

    "gdp_nowcast/src/training/bridge_dataset.py",
    "gdp_nowcast/src/training/rolling_trainer.py",

    "gdp_nowcast/src/engine/pipeline.py",
    "gdp_nowcast/src/engine/io.py",

    "gdp_nowcast/src/signal/mapper.py",
    "gdp_nowcast/src/signal/history.py",

    "gdp_nowcast/test/test_data_layer.py",
    "gdp_nowcast/test/test_feature_layer.py",
    "gdp_nowcast/test/test_model_ar.py",
    "gdp_nowcast/test/test_model_bridge.py",
    "gdp_nowcast/test/test_pipeline.py",
]

for path in project_structure:
    if path.endswith("/"):
        os.makedirs(path, exist_ok=True)
    else:
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        with open(path, "w") as f:
            f.write("")  # create empty file

print("Project structure created.")
