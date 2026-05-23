#!/usr/bin/env python3
"""Preprocess BTC data for time series forecasting."""

import os

import numpy as np
import pandas as pd


CSV_FILES = (
    "coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv",
    "bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv",
)

FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume_(BTC)",
    "Volume_(Currency)",
    "Weighted_Price",
]

VOLUME_COLUMNS = [
    "Volume_(BTC)",
    "Volume_(Currency)",
]

TARGET_COLUMN = "Close"

WINDOW_SIZE = 24
FORECAST_HORIZON = 1

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

OUTPUT_FILE = "btc_preprocessed.npz"
HOURLY_FILE = "btc_hourly.csv"


def load_csv(path):
    """Load one raw BTC csv file and remove unusable minute rows."""
    columns = ["Timestamp"] + FEATURE_COLUMNS
    data = pd.read_csv(path, usecols=columns)

    data = data.dropna(subset=FEATURE_COLUMNS, how="any")
    data = data.drop_duplicates(subset=["Timestamp"])
    data["Datetime"] = pd.to_datetime(
        data["Timestamp"],
        unit="s",
        utc=True
    )

    return data


def load_all_data(paths):
    """Load and combine all available BTC datasets."""
    frames = []

    for path in paths:
        if os.path.exists(path):
            frames.append(load_csv(path))

    if not frames:
        raise FileNotFoundError("No BTC csv files were found")

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values("Datetime")

    return data


def resample_to_hourly(data):
    """Convert minute-level BTC data into hourly OHLCV data."""
    data = data.set_index("Datetime")
    data = data.sort_index()

    hourly = pd.DataFrame()
    hourly["Open"] = data["Open"].resample("1h").first()
    hourly["High"] = data["High"].resample("1h").max()
    hourly["Low"] = data["Low"].resample("1h").min()
    hourly["Close"] = data["Close"].resample("1h").last()
    hourly["Volume_(BTC)"] = data["Volume_(BTC)"].resample("1h").sum()
    hourly["Volume_(Currency)"] = (
        data["Volume_(Currency)"].resample("1h").sum()
    )

    btc_volume = hourly["Volume_(BTC)"].replace(0, np.nan)
    hourly["Weighted_Price"] = hourly["Volume_(Currency)"] / btc_volume

    hourly = hourly.dropna(subset=FEATURE_COLUMNS, how="any")
    hourly[VOLUME_COLUMNS] = np.log1p(hourly[VOLUME_COLUMNS])

    return hourly


def split_limits(length):
    """Return chronological train and validation split limits."""
    train_end = int(length * TRAIN_RATIO)
    val_end = int(length * (TRAIN_RATIO + VAL_RATIO))

    return train_end, val_end


def normalize_features(values, train_end):
    """Normalize features using only the training portion statistics."""
    train_values = values[:train_end]
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)

    std[std == 0] = 1.0
    normalized = (values - mean) / std

    return normalized, mean, std


def build_windows(values, target_index):
    """Create 24-hour input windows and next-hour close targets."""
    x_values = []
    y_values = []
    target_indices = []
    total = values.shape[0]

    last_start = total - WINDOW_SIZE - FORECAST_HORIZON + 1

    for start in range(last_start):
        end = start + WINDOW_SIZE
        forecast_index = end + FORECAST_HORIZON - 1

        x_values.append(values[start:end])
        y_values.append(values[forecast_index, target_index])
        target_indices.append(forecast_index)

    x_values = np.asarray(x_values, dtype=np.float32)
    y_values = np.asarray(y_values, dtype=np.float32)
    target_indices = np.asarray(target_indices)

    return x_values, y_values, target_indices


def save_preprocessed_data(hourly):
    """Normalize, window, split, and save the preprocessed BTC data."""
    if hourly.shape[0] <= WINDOW_SIZE + FORECAST_HORIZON:
        raise ValueError("Not enough hourly data to build time windows")

    values = hourly[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    train_end, val_end = split_limits(values.shape[0])

    if train_end <= WINDOW_SIZE:
        raise ValueError("Training split is too small for 24-hour windows")

    values, mean, std = normalize_features(values, train_end)

    target_index = FEATURE_COLUMNS.index(TARGET_COLUMN)
    x_values, y_values, target_indices = build_windows(values, target_index)

    train_mask = target_indices < train_end
    val_mask = (target_indices >= train_end) & (target_indices < val_end)
    test_mask = target_indices >= val_end

    np.savez_compressed(
        OUTPUT_FILE,
        X_train=x_values[train_mask],
        y_train=y_values[train_mask],
        X_val=x_values[val_mask],
        y_val=y_values[val_mask],
        X_test=x_values[test_mask],
        y_test=y_values[test_mask],
        feature_mean=mean,
        feature_std=std,
        feature_columns=np.asarray(FEATURE_COLUMNS),
        target_column=TARGET_COLUMN,
        window_size=WINDOW_SIZE,
        forecast_horizon=FORECAST_HORIZON,
    )


def main():
    """Run the complete BTC preprocessing pipeline."""
    data = load_all_data(CSV_FILES)
    hourly = resample_to_hourly(data)

    hourly.to_csv(HOURLY_FILE, index=True)
    save_preprocessed_data(hourly)

    print("Saved {}".format(HOURLY_FILE))
    print("Saved {}".format(OUTPUT_FILE))
    print("Hourly rows: {}".format(hourly.shape[0]))


if __name__ == "__main__":
    main()
