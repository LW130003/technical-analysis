import pandas as pd


def look_backward_per_ticker(df: pd.DataFrame, column_to_shift: str, ticker_col: str = 'ticker', periods: int = 1) -> pd.Series:
    return df.groupby(ticker_col)[column_to_shift].shift(periods)


def apply_mean_reversion_rule(df, params_dict):
    z_score_threshold = params_dict.get("z_score_threshold", 2)
    body_ratio_threshold = params_dict.get("body_ratio_threshold", 0.7)
    condition_panic_selling = (df['IS_RED'] == True) & (df['BODY_RATIO'] >= body_ratio_threshold)
    condition_stat_extreme  = (df['close'] <= df['bb_lower']) & (df['z_score'].abs() > z_score_threshold)
        
    df['panic_alert'] = condition_panic_selling & condition_stat_extreme
    df["buy_signal"] = df["panic_alert"]
    return df

def apply_strategy_shift_v1(df, params_dict):
    z_score_threshold = params_dict.get("z_score_threshold", 2)
    body_ratio_threshold = params_dict.get("body_ratio_threshold", 0.7)

    condition_panic_selling = (df['IS_RED'] == True) & (df['BODY_RATIO'] >= body_ratio_threshold)
    condition_stat_extreme  = (df['close'] <= df['bb_lower']) & (df['z_score'].abs() > z_score_threshold)
        
    df['panic_alert'] = condition_panic_selling & condition_stat_extreme
    df['was_previous_hour_panic'] = look_backward_per_ticker(df, column_to_shift='panic_alert', periods=1)
    df['buy_signal'] = (df['was_previous_hour_panic'] == True) & (df['IS_RED'] == False)
    return df

def apply_strategy_shift_v2(df, params_dict):
    # Extract thresholds
    volume_z_threshold = params_dict.get("volume_z_threshold", 2.0) # For panic
    body_ratio_threshold = params_dict.get("body_ratio_threshold", 0.7)
    
    # 1. Capture the Panic Hour (Using your Volume Z-Score)
    condition_panic_selling = (df['IS_RED'] == True) & (df['BODY_RATIO'] >= body_ratio_threshold)
    # Assuming your 'z_score' column represents your normalized volume metric:
    condition_volume_exhaustion = df['z_score'] >= volume_z_threshold
    condition_price_extreme = df['close'] <= df['bb_lower']
        
    df['panic_alert'] = condition_panic_selling & condition_volume_exhaustion & condition_price_extreme
    
    # Shift it forward: Did the *previous* hour check out as a valid panic?
    df['was_previous_hour_panic'] = look_backward_per_ticker(df, column_to_shift='panic_alert', periods=1)    
    
    # 2. Verify the Rebound Hour (The current bar)
    # Instead of just checking if it's "not red", let's ensure buyers actually pushed the price up
    df['bar_range'] = (df['high'] - df['low']).replace(0, 1e-5)
    df['close_location'] = (df['close'] - df['low']) / df['bar_range']
    
    # Condition: Must be green, and must close in the upper 50% of the hour's price range
    condition_strong_rebound = (df['IS_RED'] == False) & (df['close_location'] >= 0.5)
    
    # 3. Generate Signal
    df['buy_signal'] = (df['was_previous_hour_panic'] == True) & condition_strong_rebound
    
    return df


def apply_kotegawa_rule(df, params_dict):
    """
    Buys panic candles only when the asset is severely disconnected/stretched
    below its moving average (Moving Average Deviation Rate).
    """
    body_ratio_threshold = params_dict.get("body_ratio_threshold", 0.7)
    # ma_period = params_dict.get("kotegawa_ma_period", 25)
    # The percentage distance below the MA (e.g., 0.10 means 10% below MA)
    deviation_threshold = params_dict.get("kotegawa_deviation", 0.10) 
    
    # 1. Calculate the core moving average
    # ma = df['close'].rolling(window=ma_period).mean()
    
    # 2. Calculate how far below the MA the current price is
    # Formula: (MA - Close) / MA
    # ma_deviation = (ma - df['close']) / ma
    ma_deviation = (df["ema_20"] - df["close"]) / df["ema_20"]
    
    # 3. Core Conditions
    condition_panic_selling = (df['IS_RED'] == True) & (df['BODY_RATIO'] >= body_ratio_threshold)
    condition_kotegawa_extreme = ma_deviation >= deviation_threshold
    
    # 4. Assign Output
    df['panic_alert'] = condition_panic_selling & condition_kotegawa_extreme
    df["buy_signal"] = df["panic_alert"]
    
    return df

def apply_momentum_confirmed_rule(df, params_dict):
    """
    Waits for a panic hour, requires a green/flat hour, AND demands 
    that short-term upward momentum has actively initiated.
    """
    z_score_threshold = params_dict.get("z_score_threshold", 2)
    body_ratio_threshold = params_dict.get("body_ratio_threshold", 0.7)
    ema_period = params_dict.get("fast_ema_period", 5)

    # 1. Standard panic alerts
    condition_panic_selling = (df['IS_RED'] == True) & (df['BODY_RATIO'] >= body_ratio_threshold)
    condition_stat_extreme  = (df['close'] <= df['bb_lower']) & (df['z_score'].abs() > z_score_threshold)
    df['panic_alert'] = condition_panic_selling & condition_stat_extreme
    
    # 2. Track past panic
    df['was_previous_hour_panic'] = look_backward_per_ticker(df, column_to_shift='panic_alert', periods=1)
    
    # 3. Momentum Filter: Calculate a fast EMA and ensure current price is reclaiming it
    # fast_ema = df['close'].ewm(span=ema_period, adjust=False).mean()
    # condition_momentum_reversal = df['close'] > fast_ema
    condition_momentum_reversal = df["close"] > df["ema_20"]
    
    # 4. Execution Logic: Previous was panic AND present is green AND price has claimed the EMA trend
    df['buy_signal'] = (
        (df['was_previous_hour_panic'] == True) & 
        (df['IS_RED'] == False) & 
        (condition_momentum_reversal == True)
    )    
    return df

# Create a strategy registry dictionary
STRATEGY_REGISTRY = {
    "baseline": apply_mean_reversion_rule,
    "shift_v1": apply_strategy_shift_v1,
    "shift_v2": apply_strategy_shift_v2,
    "kotegawa": apply_kotegawa_rule,
    "momentum_filter": apply_momentum_confirmed_rule
}



z_score_threshold = 0.2
body_ratio_threshold = 0.7
profit_target = 0.03
stop_loss_target = 0.03
max_held_hours = 48

DEFAULT_DICT = {
    "z_score_threshold": z_score_threshold,
    "body_ratio_threshold": body_ratio_threshold,
    "profit_target": profit_target,
    "stop_loss_target": stop_loss_target,
    "max_held_hours": max_held_hours,    
}

def get_all_strategies():
    return STRATEGY_REGISTRY

def get_default_params():
    return DEFAULT_DICT