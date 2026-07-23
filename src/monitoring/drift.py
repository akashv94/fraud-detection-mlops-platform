from pathlib import Path
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

REFERENCE_DATA = Path("data/reference/reference.csv")
CURRENT_DATA = Path("data/production/current.csv")
REPORT_DIR = Path("reports/drift")


def generate_drift_report():
    if not REFERENCE_DATA.exists() or not CURRENT_DATA.exists():
        raise FileNotFoundError(
            "Create data/reference/reference.csv and data/production/current.csv first."
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(REFERENCE_DATA)
    current = pd.read_csv(CURRENT_DATA)

    report = Report([DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=current)

    output = REPORT_DIR / "drift_report.html"
    result.save_html(str(output))
    print(f"Drift report saved: {output}")


if __name__ == "__main__":
    generate_drift_report()
