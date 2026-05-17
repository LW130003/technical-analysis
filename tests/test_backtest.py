import pandas as pd
import numpy as np

from technical_analysis.backtest import run_backtest


def test_take_profit_hit():
    # Setup: Buy at 100. Price hits 104 (High) on row index 2. Close is only 102.
    # The code should capture the exit price at exactly 103 (the 3% target) on index 2.
    data = {
        'ticker': ['AAPL'] * 5,
        'us_date': ['2026-05-08'] * 5,
        'hour': [4, 5, 6, 7, 8],
        'close': [100, 101, 102, 101, 100],
        'high':  [100, 101, 104, 102, 101], # +4% high triggers our +3% TP
        'low':   [100, 99, 101, 99, 98],
        'buy_signal': [True, False, False, False, False]
    }
    df = pd.DataFrame(data)
    res = run_backtest(df, profit_target=0.03, stop_loss=0.03, max_hours=4)
    
    assert len(res) == 1, "Should capture exactly 1 trade"
    assert res.iloc[0]['exit_reason'] == 'Take_Profit'
    assert res.iloc[0]['exit_price'] == 103.0, "Exit price must be exactly the 3% target"
    assert res.iloc[0]['hours_held'] == 2, "Should have held for exactly 2 hours (index 2 - index 0)"
    print("✅ Test 1 Passed: Take Profit tracking is correct.")


def test_stop_loss_hit():
    # Setup: Buy at 100. Price drops to 96 (Low) on row index 1.
    # The code should trigger SL at exactly 97 (the -3% target).
    data = {
        'ticker': ['AAPL'] * 4,
        'us_date': ['2026-05-08'] * 4,
        'hour': [4, 5, 6, 7],
        'close': [100, 98, 97, 98],
        'high':  [100, 101, 98, 99],
        'low':   [100, 96, 95, 96], # -4% low triggers our -3% SL
        'buy_signal': [True, False, False, False]
    }
    df = pd.DataFrame(data)
    res = run_backtest(df, profit_target=0.03, stop_loss=0.03, max_hours=3)
    
    assert res.iloc[0]['exit_reason'] == 'Stop_Loss'
    assert res.iloc[0]['exit_price'] == 97.0, "Exit price must be exactly the -3% target"
    assert res.iloc[0]['hours_held'] == 1, "Should have been stopped out after 1 hour"
    print("✅ Test 2 Passed: Stop Loss tracking is correct.")


def test_time_limit_exit():
    # Setup: Buy at 100. Max hold is 2 hours. Price stays within +/- 1% bounds.
    # Neither High nor Low triggers targets. Exits at index 2 close price (99.5).
    data = {
        'ticker': ['AAPL'] * 4,
        'us_date': ['2026-05-08'] * 4,
        'hour': [4, 5, 6, 7],
        'close': [100, 101, 99.5, 100.5], 
        'high':  [100, 101, 101.5, 101], # Max profit reached was only 1.5%
        'low':   [100, 99.5, 98.5, 99.5],  # Max loss reached was only -1.5%
        'buy_signal': [True, False, False, False]
    }
    df = pd.DataFrame(data)
    res = run_backtest(df, profit_target=0.03, stop_loss=0.03, max_hours=2)
    
    assert res.iloc[0]['exit_reason'] == 'Time_Limit_48h'
    assert res.iloc[0]['exit_price'] == 99.5, "Should exit at the close price of the max_hours row"
    assert res.iloc[0]['hours_held'] == 2
    print("✅ Test 3 Passed: Time Limit constraint is correct.")


def test_ticker_leakage():
    # Setup: AAPL triggers buy, but has no subsequent rows. MSFT follows immediately.
    data = {
        'ticker': ['AAPL', 'MSFT', 'MSFT'],
        'us_date': ['2026-05-08'] * 3,
        'hour': [4, 4, 5],
        'close': [100, 50, 60], 
        'high':  [100, 52, 62],
        'low':   [100, 49, 59],
        'buy_signal': [True, False, False]
    }
    df = pd.DataFrame(data)
    res = run_backtest(df, profit_target=0.03, stop_loss=0.03)
    
    assert res.iloc[0]['exit_reason'] == 'ERROR_NO_FORWARD_DATA'
    print("✅ Test 4 Passed: Ticker leakage protection is working perfectly.")


def test_simultaneous_hit():
    # Setup: Buy at 100. Next hour goes completely haywire: High is 105 (+5%), Low is 95 (-5%).
    # The code must execute the worst-case scenario: Stop_Loss_Simultaneous_Hit at 97.
    data = {
        'ticker': ['AAPL'] * 3,
        'us_date': ['2026-05-08'] * 3,
        'hour': [4, 5, 6],
        'close': [100, 101, 102],
        'high':  [100, 105, 102], 
        'low':   [100, 95, 101],  
        'buy_signal': [True, False, False]
    }
    df = pd.DataFrame(data)
    res = run_backtest(df, profit_target=0.03, stop_loss=0.03, max_hours=2)
    
    assert res.iloc[0]['exit_reason'] == 'Stop_Loss_Simultaneous_Hit'
    assert res.iloc[0]['exit_price'] == 97.0, "Should register the losing price for safety"
    assert res.iloc[0]['hours_held'] == 1
    print("✅ Test 5 Passed: Simultaneous hit edge case resolved conservatively.")