import talib
import numpy as np
import os
import pandas as pd

from datetime import datetime
from typing import List

# def get_backtest_bounds(target_date: str, last_n_days: int):
#     target_dt = pd.to_datetime(target_date)
#     end_date_dt = target_dt - pd.Timedelta(days=1)
#     start_date_dt = end_date_dt - pd.Timedelta(days=last_n_days)
#     return start_date_dt.strftime('%Y-%m-%d'), end_date_dt.strftime('%Y-%m-%d')

def get_backtest_bounds(target_date: str, last_n_days: int):
    target_dt = pd.to_datetime(target_date)
    end_date_dt = target_dt - pd.offsets.BusinessDay(1)
    start_date_dt = end_date_dt - pd.offsets.BusinessDay(last_n_days)
    return start_date_dt.strftime('%Y-%m-%d'), end_date_dt.strftime('%Y-%m-%d')


def generate_csv_filepaths(start_date, end_date, save_dir):
    date_series = pd.date_range(start=start_date, end=end_date)    
    keys = []
    base_path = save_dir
    
    for dt in date_series:
        full_date = dt.strftime('%Y-%m-%d')
        key = os.path.join(base_path, f"{full_date}.csv.gz")
        keys.append(key)        
    return keys

def generate_csv_filepath(date_str, save_dir):
    dt = pd.to_datetime(date_str)
    base_path = save_dir
    full_date = dt.strftime('%Y-%m-%d')
    key = os.path.join(base_path, f"{full_date}.csv.gz")
    return key


def convert_timestamp_to_us_datetime(timestamp: int, unit="ms"):
    """
    Converts a millisecond timestamp to US/Eastern time.

    NYSE/NASDAQ are located at US/Eastern
    """
    return pd.to_datetime(
        timestamp, unit=unit, utc=True
    ).tz_convert('US/Eastern')      
    
def load_csv_by_chunking(
    filepath: str, ticker_list: List[str],
    chunk_size = 100000
):

    chunks = []
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        filtered_chunk = chunk[chunk["ticker"].isin(ticker_list)]
        chunks.append(filtered_chunk)

    df = pd.concat(chunks)
    return df

def aggregate_to_hour(df):
    df["us_datetime"] = df["window_start"].apply(
        lambda x: convert_timestamp_to_us_datetime(x, unit="ns"))
    df["us_date"] = df["us_datetime"].dt.date
    df["hour"] = df["us_datetime"].dt.hour
    agg_logic = {
        "open": "first",    # The first price of the hour
        "high": "max",      # The highest price in that hour
        "low": "min",       # The lowest price in that hour
        "close": "last",    # The final price of the hour
        "volume": "sum",    # Total volume traded in that hour
        "transactions": "sum" # Total count of trades in that hour
    }
    
    hour_df = df.groupby(
        ["ticker", "us_date", "hour"]
    ).agg(agg_logic).reset_index()
    return hour_df


def get_target_and_historical_df(target_date, last_n_days, save_dir, ticker_list):
    start_date, end_date = get_backtest_bounds(target_date, last_n_days)
    csv_filepaths = generate_csv_filepaths(start_date, end_date, save_dir)

    df_ls = []
    
    for csv_filepath in csv_filepaths:
        try:
            # TO CHANGE
            # load_csv_by_chunking - use pd.read_csv()
            df_ls.append(
                load_csv_by_chunking(csv_filepath, ticker_list)
            )
        except FileNotFoundError:
            print(f"Error: The file at {csv_filepath} was not found. Skipping...")
        except Exception as e:
            print(f"An unexpected error occurred with {os.path.basename(csv_filepath)}: {e}")
            
    hist_df = pd.concat(df_ls).reset_index(drop=True)
    
    # TO CHANGE
    # load_csv_by_chunking - use pd.read_csv()
    target_df = load_csv_by_chunking(
        generate_csv_filepath(target_date, save_dir), ticker_list
    )

    hour_df = aggregate_to_hour(hist_df)
    target_hour_df = aggregate_to_hour(target_df)    
    return target_hour_df, hour_df


def get_indicators_with_lookback(target_df: pd.DataFrame, hist_df: pd.DataFrame) -> pd.DataFrame:
    # 1. Capture original index to guarantee perfect structural sorting alignment later
    target_copy = target_df.copy()
    hist_copy = hist_df.copy()
    
    target_copy['_orig_idx'] = target_copy.index
    target_copy['_is_target'] = True
    hist_copy['_orig_idx'] = -1
    hist_copy['_is_target'] = False
    
    # 2. Combine and sort
    combined = pd.concat([hist_copy, target_copy], ignore_index=True)
    combined = combined.sort_values(by=["ticker", "us_date", "hour"]).reset_index(drop=True)
    
    # 3. Apply TA-Lib inside safe ticker boundaries
    def _calc_metrics(group):
        upper, middle, lower = talib.BBANDS(
            group["close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        group['bb_upper'] = upper
        group['bb_middle'] = middle
        group['bb_lower'] = lower
        group['ema_20'] = talib.EMA(group["close"], timeperiod=20)
        group['sma_20'] = talib.SMA(group["close"], timeperiod=20)
        return group

    processed = combined.groupby("ticker", group_keys=False).apply(_calc_metrics)
    
    # 4. Extract target rows and restore original dataframe order
    target_rows = processed[processed['_is_target'] == True].copy()
    target_rows = target_rows.sort_values('_orig_idx').set_index('_orig_idx')
    target_rows.index.name = target_df.index.name  # Restore original index name if any
    
    # Return ONLY the newly created feature columns
    feature_cols = ['bb_upper', 'bb_middle', 'bb_lower', 'ema_20', 'sma_20']
    return target_rows[feature_cols]    


def volume_check(target_hour_df, hour_df, apply_log_norm: bool=False):    
    if apply_log_norm:
        hour_df["volume"] = np.log1p(hour_df["volume"])
        target_hour_df["volume"] = np.log1p(target_hour_df["volume"])
        
    volume_df = hour_df.groupby(["ticker", "hour"]).agg({
        "volume": ["mean", "std"]
    })
    volume_df.columns = [f"{col}_{stat}" for col, stat in volume_df.columns]
    volume_df = volume_df.reset_index()
    
    merge_df = pd.merge(target_hour_df, volume_df, on=["ticker", "hour"])
    merge_df["z_score"] = (merge_df["volume"] - merge_df["volume_mean"]) / merge_df["volume_std"]
    merge_df["IS_RED"] = (merge_df["close"] - merge_df["open"]) < 0
    merge_df["BODY_RATIO"] = abs((merge_df["open"] - merge_df["close"]) / (merge_df["high"] - merge_df["low"]))
    merge_df["IS_BODY_LARGE"] = merge_df["BODY_RATIO"] > 0.7

    indicator_cols = ['bb_upper', 'bb_middle', 'bb_lower', 'ema_20', 'sma_20']
    merge_df[indicator_cols] = get_indicators_with_lookback(
        target_df=merge_df, hist_df=hour_df)
    return merge_df
