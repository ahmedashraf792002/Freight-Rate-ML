"""Train, validate (forward-chaining by time), predict validation.csv and the December chart inputs.

Usage: python src/train.py --data-dir data --out-dir outputs
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import FEATURES, build_features

PARAMS = dict(objective="l1", n_estimators=500, learning_rate=0.03, num_leaves=15,
              min_child_samples=80, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              reg_lambda=5, verbose=-1, n_jobs=1)
# expanding-window folds: train on everything BEFORE the fold, test on the fold
FOLDS = [("2025-07-01", "2025-08-31"), ("2025-09-01", "2025-09-30"), ("2025-10-01", "2025-10-31")]


def metrics(y, p):
    return dict(MAE=float(np.mean(abs(y - p))), RMSE=float(np.sqrt(np.mean((y - p) ** 2))),
                MAPE=float(np.mean(abs(y - p) / y) * 100), MedAPE=float(np.median(abs(y - p) / y) * 100))


def fit(df, seed=0):
    # target = log(rate per mile); prediction = exp(pred) * distance.
    # L1 loss is robust to the ~1.5% heavy label outliers.
    return lgb.LGBMRegressor(**PARAMS, random_state=seed).fit(df[FEATURES], np.log(df.posted_rate / df.distance))


def predict(models, df):
    return np.mean([np.exp(m.predict(df[FEATURES])) for m in models], axis=0) * df.distance.values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="outputs")
    a = ap.parse_args()
    dd, out = Path(a.data_dir), Path(a.out_dir)
    out.mkdir(exist_ok=True, parents=True)
    tr = build_features(pd.read_csv(dd / "train_test.csv"))


    rows = []
    for s, e in FOLDS:
        a_, b_ = tr[tr.date < s], tr[(tr.date >= s) & (tr.date <= e)]
        p = predict([fit(a_)], b_)
        med = (a_.posted_rate / a_.distance).groupby(a_.equipment, observed=True).median()
        naive = b_.distance.values * med.reindex(b_.equipment).values
        rows.append(dict(fold=f"{s}..{e}", n_train=len(a_), n_test=len(b_),
                         **{f"model_{k}": v for k, v in metrics(b_.posted_rate, p).items()},
                         **{f"naive_{k}": v for k, v in metrics(b_.posted_rate, naive).items()}))
    cv = pd.DataFrame(rows)
    cv.to_csv(out / "cv_results.csv", index=False)
    print(cv.round(2).to_string(index=False))
    print("mean over folds:\n", cv.drop(columns="fold").mean().round(2).to_string())


    models = [fit(tr, s) for s in range(3)]


    val = build_features(pd.read_csv(dd / "validation.csv"))
    tmpl = pd.read_csv(dd / "validation_predictions_template.csv")[["load_id"]]
    pred = pd.DataFrame({"load_id": val.load_id, "predicted_rate": predict(models, val).round(2)})
    tmpl.merge(pred, on="load_id", how="left").to_csv(out / "validation_predictions.csv", index=False)


    dec = pd.read_csv(dd / "december_chart_inputs.csv")
    pick = tr.drop_duplicates("pickup").set_index("pickup")[["pickup_lat", "pickup_lon"]]
    deliv = tr.drop_duplicates("delivery").set_index("delivery")[["delivery_lat", "delivery_lon"]]
    x = dec.copy()
    x = x.join(pick, on="pickup").join(deliv, on="delivery")
    x = build_features(x)
    dec["predicted_rate"] = predict(models, x).round(2)
    dec.to_csv(out / "december_chart_inputs.csv", index=False)
    print("December range:", dec.predicted_rate.min(), dec.predicted_rate.max())


if __name__ == "__main__":
    main()
