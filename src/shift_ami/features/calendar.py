"""
Deterministic calendar features known in advance at forecast origin.
"""
from typing import Optional, Set
import numpy as np
import pandas as pd


# Documented UK Bank Holidays for London smart meter trial period (2011-2014)
UK_BANK_HOLIDAYS: Set[str] = {
    # 2011
    "2011-12-26", "2011-12-27",
    # 2012
    "2012-01-02", "2012-04-06", "2012-04-09", "2012-05-07", "2012-06-04",
    "2012-06-05", "2012-08-27", "2012-12-25", "2012-12-26",
    # 2013
    "2013-01-01", "2013-03-29", "2013-04-01", "2013-05-06", "2013-05-27",
    "2013-08-26", "2013-12-25", "2013-12-26",
    # 2014
    "2014-01-01", "2014-04-18", "2014-04-21", "2014-05-05", "2014-05-26"
}


def extract_calendar_features(timestamps: pd.Series, include_holidays: bool = True) -> pd.DataFrame:
    """
    Extract calendar features from a pandas Series of timestamps.
    These features represent deterministic time properties known in advance for any target horizon.

    Returns
    -------
    pd.DataFrame with columns:
        - half_hour_idx (0..47)
        - hour (0..23)
        - day_of_week (0..6)
        - is_weekend (0/1)
        - month (1..12)
        - day_of_year (1..366)
        - is_bank_holiday (0/1)
        - sin_hh, cos_hh (cyclical half-hour)
        - sin_dow, cos_dow (cyclical day-of-week)
        - sin_doy, cos_doy (cyclical day-of-year)
    """
    ts = pd.to_datetime(timestamps)

    hh_idx = ts.dt.hour * 2 + ts.dt.minute // 30
    hour = ts.dt.hour
    dow = ts.dt.dayofweek
    is_weekend = (dow >= 5).astype(float)
    month = ts.dt.month
    doy = ts.dt.dayofyear

    # Cyclical sine/cosine encodings
    sin_hh = np.sin(2 * np.pi * hh_idx / 48.0)
    cos_hh = np.cos(2 * np.pi * hh_idx / 48.0)
    sin_dow = np.sin(2 * np.pi * dow / 7.0)
    cos_dow = np.cos(2 * np.pi * dow / 7.0)
    sin_doy = np.sin(2 * np.pi * doy / 365.25)
    cos_doy = np.cos(2 * np.pi * doy / 365.25)

    df_cal = pd.DataFrame({
        "half_hour_idx": hh_idx.values,
        "hour": hour.values,
        "day_of_week": dow.values,
        "is_weekend": is_weekend.values,
        "month": month.values,
        "day_of_year": doy.values,
        "sin_hh": sin_hh.values,
        "cos_hh": cos_hh.values,
        "sin_dow": sin_dow.values,
        "cos_dow": cos_dow.values,
        "sin_doy": sin_doy.values,
        "cos_doy": cos_doy.values,
    }, index=timestamps.index)

    if include_holidays:
        date_str = ts.dt.strftime("%Y-%m-%d")
        df_cal["is_bank_holiday"] = date_str.isin(UK_BANK_HOLIDAYS).astype(float).values
    else:
        df_cal["is_bank_holiday"] = 0.0

    return df_cal
