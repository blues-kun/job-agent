from typing import Dict, Any
from pathlib import Path

from models import ResumeProfile
from features.extractor import extract

def is_ready(model_path: str) -> bool:
    p = Path(model_path)
    return p.exists()

def score(resume: ResumeProfile, job: Dict[str, Any], model_path: str) -> float:
    try:
        import xgboost as xgb
    except Exception:
        return 0.0
    feats = extract(resume, job)
    order = sorted(feats.keys())
    X = [[feats[k] for k in order]]
    dm = xgb.DMatrix(X, feature_names=order)
    bst = xgb.Booster()
    bst.load_model(model_path)
    pred = float(bst.predict(dm)[0])
    return pred

