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


STRATEGY_DICT = {
    "default": apply_mean_reversion_rule,
    "shift_v1": apply_strategy_shift_v1,
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
    return STRATEGY_DICT

def get_default_params():
    return DEFAULT_DICT