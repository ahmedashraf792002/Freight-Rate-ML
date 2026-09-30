"""Cleaning + feature engineering (shared by training, validation and December chart)."""
import numpy as np
import pandas as pd

EQUIPMENT = ["Dry Van", "Reefer", "Flatbed"]


FEATURES = ["distance", "log_distance", "great_circle", "weight_c", "weight_missing",
            "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "equipment", "dow"]


def great_circle(lat1, lon1, lat2, lon2):
    a, b, c, d = map(np.radians, [lat1, lon1, lat2, lon2])
    x = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(x))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["great_circle"] = great_circle(d.pickup_lat, d.pickup_lon, d.delivery_lat, d.delivery_lon)
    d["log_distance"] = np.log(d["distance"])
 
    d["weight_missing"] = d["weight"].isna().astype(int)
    d["weight_c"] = d["weight"].abs()
    d["dow"] = d["date"].dt.dayofweek
    d["equipment"] = pd.Categorical(d["equipment"].astype(str), categories=EQUIPMENT)
    return d
