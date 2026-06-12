# technical-analysis

# Download Flat Files

Notebook: 
- ./technical-analysis/notebooks/download_flat_files_automation.ipynb

# Preprocess Flat Files to CSV FIles

In this step, we preprocess the one-minute data of Flat files into hourly data.

Notebook:
- ./technical-analysis/notebooks/volume_check_loop.ipynb


# Simulation

Notebook:
- ./technical-analysis/notebooks/simulation.ipynb
- ./technical-analysis/notebooks/simulation_loop.ipynb

Note:
- The strategy related code can be found under: ./technical-analysis/src/technical_analysis/strategy.py
- The backtest related code can be found under: ./technical-analysis/src/technical_analysis/backtest.py

# Execution Order
1. Download Flat Files
2. Preprocess Flat Files to CSV Files
3. Execute Simulation


You can Add and modify the strategy and repeat the step 3. 

Step 2 only repeated if you need to redo the hourly aggregation process, computing additional metrics.

