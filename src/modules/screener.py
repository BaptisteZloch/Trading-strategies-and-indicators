# def calculate_indicators(df):
#     #  Long-term
#     df['EMA_200'] = ta.ema(df['Close'], length=200)
#     df['EMA_200_CLOSE_PC'] = (df['Close'] / df['EMA_200']) * 100
#     df['ROC_125'] = ta.momentum.roc(close=df['Close'], window=125)
#     #  Mid-term
#     df['EMA_50'] = ta.ema(df['Close'], length=50)
#     df['EMA_50_CLOSE_PC'] = (df['Close'] / df['EMA_50']) * 100
#     df['ROC_20'] = ta.momentum.roc(close=df['Close'], window=20)
#     # Short-term
#     ppo_ind = PercentagePriceOscillator(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
#     df['PPO'] = ppo_ind.ppo()
#     df['PPO_EMA_9'] = ta.ema(df['PPO'], length=9)
#     df['PPO_HIST'] = df['Close'] - df['PPO_EMA_9']
#     #  Calculate PPO histogram slope
#     df = calculate_ppo_hist_slope(df)
#     df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
#     return df

# def calculate_weights(df):
#     #  Long-term
#     df['EMA_200_CLOSE_PC_WEIGHTED'] = df['EMA_200_CLOSE_PC'] * 0.3
#     df['ROC_125_WEIGHTED'] = df['ROC_125'] * 0.3
#     #  Mid-term
#     df['EMA_50_CLOSE_PC_WEIGHTED'] = df['EMA_50_CLOSE_PC'] * 0.15
#     df['ROC_20_WEIGHTED'] = df['ROC_20'] * 0.15
#     #  Short-term
#     df['RSI_WEIGHTED'] = df['RSI'] * 0.05
#     df['PPO_HIST_SLOPE_WEIGHTED'] = 0
#     df.loc[df['PPO_HIST_SLOPE'] < -1, 'PPO_HIST_SLOPE_WEIGHTED'] = 0
#     df.loc[df['PPO_HIST_SLOPE'] >= -1, 'PPO_HIST_SLOPE_WEIGHTED'] = (df['PPO_HIST_SLOPE'] + 1) * 50 * 0.05
#     df.loc[df['PPO_HIST_SLOPE'] > 1, 'PPO_HIST_SLOPE_WEIGHTED'] = 5
#     return df