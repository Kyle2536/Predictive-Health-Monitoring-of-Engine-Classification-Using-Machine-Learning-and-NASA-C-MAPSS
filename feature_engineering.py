from __future__ import annotations
import pandas as pd

def add_normalized_cycles(df, unit_col="engine_id", cycle_col="cycle"):
    g = df.groupby(unit_col)[cycle_col].transform("max")
    df = df.copy(); df["normalized_cycles"] = df[cycle_col] / g
    return df
