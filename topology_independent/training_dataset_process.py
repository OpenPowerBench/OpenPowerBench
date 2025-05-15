import glob
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import os
DATA_DIR   = "data/lmp"     # folder for specific task
X_rows, y_rows = [], []

for fp in glob.glob(f"{DATA_DIR}/*.csv"):
    df = pd.read_csv(fp, parse_dates=[0])
    time_col  = df.columns[0]
    end_cols = df.columns[1]
    df.rename(columns={time_col: "ts"}, inplace=True)
    skip_set = {"ts", *end_cols,*time_col}

    zone_cols = [c for c in df.columns if c not in skip_set and pd.api.types.is_numeric_dtype(df[c])]

    df["date"] = df["ts"].dt.normalize()

    for zone in zone_cols:
        #hour-ahead forecasting
        s = (
            df[["ts", zone]]
            .dropna(subset=[zone])
            .sort_values("ts")
            .assign(hour=lambda x: x["ts"].dt.floor("h"))  # <‑‑ NEW
        )

        hours = s["hour"].unique()
        for h in hours[:-1]:  # last hour has no target
            this_hr = s[s["hour"] == h]
            next_hr = s[s["hour"] == h + timedelta(hours=1)]

            if len(this_hr) != 12 or len(next_hr) != 12:
                continue

            x_vals = this_hr[zone].to_numpy()
            y_vals = next_hr[zone].to_numpy()
            X_rows.append(x_vals)
            y_rows.append(y_vals)


        #day-ahead forecasting
        # s = df[["ts", "date", zone]].dropna(subset=[zone]).sort_values("ts")
        #
        # days = s["date"].unique()
        # for d in days[:-1]:
        #     day_df     = s[s["date"] == d]
        #     nextday_df = s[s["date"] == d + timedelta(days=1)]
        #
        #     if len(day_df) != 24 or len(nextday_df) != 24:
        #         continue
        #
        #     x_vals = day_df[zone].to_numpy()
        #     y_vals = nextday_df[zone].to_numpy()
        #     X_rows.append(x_vals)
        #     y_rows.append(y_vals)

X    = np.stack(X_rows)
y    = np.stack(y_rows)
np.save("/data/lmp/X_lmp_hourly.npy", X);  np.save("/data/lmp/y_lmp_hourly.npy", y)
