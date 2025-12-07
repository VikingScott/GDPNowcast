# run_bridge_step3.py

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.rolling_trainer import RollingBridgeTrainer  # noqa: E402


def main():
    trainer = RollingBridgeTrainer(
        config_path="config/series.yaml",
        raw_data_dir="data/raw",
    )
    pred_df = trainer.run()
    trainer.save_outputs(pred_df)

    print("[OK] Step 3 Simple Bridge completed.")
    print("Predictions -> data/results/bridge/simple_bridge_predictions.csv")
    print("Eval time   -> eval/bridge_time_slice.csv")
    print("Eval dist   -> eval/bridge_distance_bucket.csv")


if __name__ == "__main__":
    main()