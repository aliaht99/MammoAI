"""Train the Gradient Boosting model and save it to disk for the Streamlit app."""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

DATA_DIR = Path("/Users/alihamza/Desktop/AICD/manifest-ZkhPvrLo5216730872708713142")
OUT_DIR  = Path("/Users/alihamza/Desktop/AICD")

CALC_TYPE_RISK = {
    "PLEOMORPHIC": 3, "AMORPHOUS": 2, "HETEROGENEOUS": 2,
    "FINE_LINEAR_BRANCHING": 3, "PUNCTATE": 1, "LUCENT_CENTERED": 0,
    "ROUND_AND_REGULAR": 0, "EGGSHELL": 0, "MILK_OF_CALCIUM": 0,
    "COARSE": 0, "LARGE_RODLIKE": 0, "DYSTROPHIC": 0, "N/A": 1,
}
CALC_DIST_RISK = {
    "LINEAR": 3, "SEGMENTAL": 2, "REGIONAL": 1,
    "DIFFUSELY_SCATTERED": 0, "CLUSTERED": 2,
}
MASS_SHAPE_RISK = {
    "IRREGULAR": 3, "IRREGULAR-ARCHITECTURAL_DISTORTION": 3,
    "LOBULATED": 2, "OVAL": 1, "ROUND": 1, "ARCHITECTURAL_DISTORTION": 2,
    "LYMPH_NODE": 0,
}
MASS_MARGIN_RISK = {
    "SPICULATED": 3, "ILL_DEFINED": 2, "OBSCURED": 1,
    "MICROLOBULATED": 2, "CIRCUMSCRIBED": 0,
}
TARGET_MAP = {"MALIGNANT": 1, "BENIGN": 0, "BENIGN_WITHOUT_CALLBACK": 0}

FEATURE_COLS = [
    "assessment", "subtlety", "breast_density", "is_mass",
    "calc_type_risk", "calc_dist_risk", "mass_shape_risk", "mass_margin_risk",
    "morph_risk", "view_mlo", "is_right",
]


def risk_score(value, mapping, default=1):
    if pd.isna(value):
        return default
    key = str(value).upper().strip()
    parts = key.replace("-", "_").split("_AND_")
    scores = [mapping.get(p.strip(), default) for p in parts if p.strip()]
    return max(scores) if scores else default


def load_and_prepare():
    frames = []
    for fname in [
        "calc_case_description_train_set.csv",
        "mass_case_description_train_set.csv",
    ]:
        df = pd.read_csv(DATA_DIR / fname, skipinitialspace=True)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        for col in df.select_dtypes("object"):
            df[col] = df[col].str.strip()
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df["label"] = df["pathology"].map(TARGET_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    df["assessment"]    = pd.to_numeric(df["assessment"], errors="coerce")
    df["subtlety"]      = pd.to_numeric(df["subtlety"], errors="coerce")
    df["breast_density"]= pd.to_numeric(df["breast_density"], errors="coerce")
    df["is_mass"]       = (df["abnormality_type"].str.lower() == "mass").astype(int)

    df["calc_type_risk"]  = df.get("calc_type", pd.Series(dtype=str)).apply(lambda x: risk_score(x, CALC_TYPE_RISK))
    df["calc_dist_risk"]  = df.get("calc_distribution", pd.Series(dtype=str)).apply(lambda x: risk_score(x, CALC_DIST_RISK))
    df["mass_shape_risk"] = df.get("mass_shape", pd.Series(dtype=str)).apply(lambda x: risk_score(x, MASS_SHAPE_RISK))
    df["mass_margin_risk"]= df.get("mass_margins", pd.Series(dtype=str)).apply(lambda x: risk_score(x, MASS_MARGIN_RISK))
    df["morph_risk"]      = df["calc_type_risk"] + df["calc_dist_risk"] + df["mass_shape_risk"] + df["mass_margin_risk"]
    df["view_mlo"]        = (df["image_view"].str.upper() == "MLO").astype(int)
    df["is_right"]        = (df["left_or_right_breast"].str.upper() == "RIGHT").astype(int)

    X = df[FEATURE_COLS]
    y = df["label"].values
    return X, y


X, y = load_and_prepare()

model = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("clf", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                       max_depth=4, random_state=42)),
])
model.fit(X, y)

with open(OUT_DIR / "gb_model.pkl", "wb") as f:
    pickle.dump(model, f)

print(f"Model saved → {OUT_DIR / 'gb_model.pkl'}")
print(f"Training samples: {len(y)} | Malignant: {y.sum()} | Benign: {(y==0).sum()}")
