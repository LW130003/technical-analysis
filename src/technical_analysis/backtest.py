import pandas as pd
import numpy as np


def run_backtest(df: pd.DataFrame, profit_target: float = 0.03, stop_loss: float = 0.03, max_hours: int = 48) -> pd.DataFrame:
    """
    Executes trades based on buy_signal and tracks performance over a forward window.
    Handles multi-ticker boundaries and logs data shortages.
    """
    # Sort data chronologically per ticker to ensure forward indexing is flawless
    df = df.sort_values(by=['ticker', 'us_date', 'hour']).reset_index(drop=True)
    
    # 1. Find the exact indices where a buy signal triggered
    trade_indices = df[df['buy_signal'] == True].index
    
    results = []
    
    # 2. Loop ONLY through the active trades, not the whole dataframe
    for start_idx in trade_indices:
        buy_row = df.loc[start_idx]
        ticker = buy_row['ticker']
        buy_price = buy_row['close']
        buy_time = f"{buy_row['us_date']} {buy_row['hour']}:00"
        
        # 3. Grab the next 48 hours safely for THIS ticker
        forward_window = df.loc[start_idx + 1 : start_idx + max_hours]
        forward_window = forward_window[forward_window['ticker'] == ticker]
        
        # EFFICIENT EDGE CASE CHECK: 
        # If the forward window is empty, it means there is absolutely zero trailing data 
        # left for this ticker in the dataset.
        if len(forward_window) == 0:
            results.append({
                'ticker': ticker, 
                'buy_time': buy_time, 
                'buy_price': buy_price,
                'exit_price': np.nan, 
                'return': 0.0, 
                'exit_reason': 'ERROR_NO_FORWARD_DATA', 
                'hours_held': 0,
                'buy_idx': start_idx,
            })
            continue

        # 4. If code reaches here, forward_window has rows! Run your evaluations...
        max_hourly_profit = (forward_window['high'] - buy_price) / buy_price
        max_hourly_loss = (forward_window['low'] - buy_price) / buy_price
        
        
        # 5. Check targets based on High and Low boundaries
        hit_tp = max_hourly_profit >= profit_target
        hit_sl = max_hourly_loss <= -stop_loss
        
        # Find the row dataframe indices where conditions were met
        tp_indices = max_hourly_profit[hit_tp].index
        sl_indices = max_hourly_loss[hit_sl].index
        
        first_tp = tp_indices[0] if len(tp_indices) > 0 else float('inf')
        first_sl = sl_indices[0] if len(sl_indices) > 0 else float('inf')
        
        # 6. Determine exit reason and price based on what happened first
        if first_tp == float('inf') and first_sl == float('inf'):
            # Neither hit: Exit at the end of the 48-hour window using CLOSE price
            final_row = forward_window.iloc[-1]
            exit_price = final_row['close']
            hours_held = len(forward_window)
            exit_reason = 'Time_Limit_48h' if hours_held == max_hours else 'Data_Cutoff_End_of_File'
            
        elif first_tp < first_sl:
            # Take Profit hit first. Your exit price is your exact target value.
            exit_price = buy_price * (1 + profit_target)
            hours_held = int(first_tp - start_idx)
            exit_reason = 'Take_Profit'
            
        elif first_sl < first_tp:
            # Stop Loss hit first. Your exit price is your exact stop loss value.
            exit_price = buy_price * (1 - stop_loss)
            hours_held = int(first_sl - start_idx)
            exit_reason = 'Stop_Loss'
            
        else:
            # EDGE CASE: Both TP and SL were hit in the exact same hour block.
            # To remain strictly conservative, we assume the worst-case scenario: we hit the Stop Loss first.
            exit_price = buy_price * (1 - stop_loss)
            hours_held = int(first_sl - start_idx)
            exit_reason = 'Stop_Loss_Simultaneous_Hit'
            
        final_return = (exit_price - buy_price) / buy_price
        
        results.append({
            'ticker': ticker,
            'buy_time': buy_time,
            'buy_price': buy_price,
            'exit_price': exit_price,
            'return': final_return,
            'exit_reason': exit_reason,
            'hours_held': hours_held,
            'buy_idx': start_idx,
        })
    return pd.DataFrame(results)
