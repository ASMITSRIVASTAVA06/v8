"""
TerraVision AI — model card.

A small JSON file next to the classifier that says what the model was trained on and how well it
did, so the app can tell the user honestly whether the classifier has seen real data.

  provenance "synthetic"  -> trained on generated examples (the default first-run model)
  provenance "real"       -> trained on real, independently labelled satellite clusters
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

CARD_PATH = Path(__file__).parent / "model_card.json"


def read(path: Path = CARD_PATH) -> Optional[Dict]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write(card: Dict, path: Path = CARD_PATH) -> Dict:
    import sklearn
    card = {"trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "sklearn_version": sklearn.__version__, **card}
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(card, indent=2), encoding="utf-8")
    tmp.replace(path)
    return card


def summary(card: Optional[Dict]) -> str:
    """One honest line for the sidebar."""
    if not card:
        return "Classifier: no model card (trained on synthetic examples unless you ran train_real.py)."
    if card.get("provenance") == "real":
        m = card.get("metrics", {})
        return (f"Classifier: trained on {card.get('n_samples', '?')} real, independently labelled clusters; "
                f"spatial-CV macro-F1 {m.get('macro_f1', float('nan')):.2f}.")
    return "Classifier: trained on SYNTHETIC examples only. Build real data (training_data.py) and run train_real.py."
