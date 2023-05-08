import pytz
from datetime import datetime, timedelta
import numpy as np
from gym_mtsim import MtEnv, MtSimulator, SymbolInfo, Timeframe, STOCKS_DATA_PATH
from stable_baselines3 import A2C

import yfinance as yf
import pandas as pd

import pickle as pkl


df = yf.download('AAPL', period='7d', interval='1m')
# info = ({"AAPL": SymbolInfo()}, {"AAPL": df})

#with open("aapl.pkl", "wb") as f:
#    pkl.dump(info, f)

# with open("aapl.pkl", "rb") as f:
#     df = pkl.load(f)

sim = MtSimulator(
    unit='USD',
    balance=200.,
    leverage=1.,
    stop_out_level=0.2,
    hedge=False,
    symbols_filename=STOCKS_DATA_PATH
)

# if not sim.load_symbols("aapl.pkl"):
#     sim.download_data(
#         symbols=['AAPL'],
#         time_range=(
#             datetime(2021, 5, 5, tzinfo=pytz.UTC),
#             datetime(2021, 9, 5, tzinfo=pytz.UTC)
#         ),
#         timeframe=Timeframe.D1
#     )
#     sim.save_symbols("aapl.pkl")

env = MtEnv(
    original_simulator=sim,
    trading_symbols=['AAPL', 'TSLA', 'MSFT', 'GOGL'],
    window_size=10,
    # time_points=[desired time points ...],
    hold_threshold=0.5,
    close_threshold=0.5,
    fee=lambda symbol: {
        'AAPL': max(1.0, np.random.normal(1, 0.5)),
        'TSLA': max(1.0, np.random.normal(1, 0.5)),
        'MSFT': max(1.0, np.random.normal(1, 0.5)),
        'GOGL': max(1.0, np.random.normal(1, 0.5)),
    }[symbol],
    symbol_max_orders=2,
    multiprocessing_processes=2
)

# env = gym.make('forex-hedge-v0')

train = False
model_file = 'a2d_200k_trained.model'
if train:
    model = A2C('MultiInputPolicy', env, verbose=0, tensorboard_log="./a2c_cartpole_tensorboard/")
    model.learn(total_timesteps=500000)
    #model.learn(total_timesteps=100000, reset_num_timesteps=False)
else:
    model = A2C.load(model_file)

observation = env.reset()
while True:
    action, _states = model.predict(observation)
    #action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        break

# env.render('advanced_figure', time_format="%Y-%m-%d")
env.render('advanced_figure')
state = env.render()

print(
    f"balance: {state['balance']}, equity: {state['equity']}, margin: {state['margin']}\n"
    f"free_margin: {state['free_margin']}, margin_level: {state['margin_level']}\n"
)

print(state['orders'])

model.save(model_file)
