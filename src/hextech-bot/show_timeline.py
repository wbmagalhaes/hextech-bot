import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

paths = [
    'data/exec_1-1 (0).csv',
    'data/exec_1-1 (1).csv',
    'data/exec_1-1 (2).csv',
    'data/exec_1-1 (3).csv',
    'data/exec_1-1 (4).csv',
    'data/exec_1-1 (5).csv',
]


def plot_interval(interval):
    x = np.arange(0, 90, interval)
    plt.vlines(x, -500, 500, colors='k', linestyles='dashed')


def plot_df(df, offset, cmap='brg'):
    x = df['Time']  # np.round(df['Time'] / 0.25) * 0.25
    y = [offset] * len(x)
    c = df['Command'].map({'JUMP': 0, 'BOMB': 1, 'DOWN': 2})
    plt.scatter(x, y, c=c, s=30, cmap=cmap)


def plot_diff(df, cmap='brg'):
    x = df['Time']
    y = np.diff(x)
    x = x[:-1]
    plt.scatter(x, y, s=20, cmap=cmap)
    return y


plot_interval(0.25)
for i, path in enumerate(paths):
    df = pd.read_csv(path)
    plot_df(df, i)


diff = []
for i, path in enumerate(paths):
    d = plot_diff(pd.read_csv(path))
    diff.extend(d)

# mean = np.mean(diff)
# median = np.median(diff)
# print(mean)
# print(median)

plt.xlim(-2, 90)
plt.ylim(-20, 20)
plt.show()