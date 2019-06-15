import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

# Alas, cyclic import
# from src.main import Stats
from .stats import Stats


def build_stats_plot(stats: Stats, smoothing_window=10):

    fig = plt.figure(figsize=(10, 15))
    plt.figure(figsize=(15, 10), facecolor="w")
    # noinspection PyTypeChecker
    ax1: Axes = plt.subplot(2, 1, 1)
    # noinspection PyTypeChecker
    ax2: Axes = plt.subplot(2, 1, 2, sharex=ax1)

    # plot the episode length over time
    ax1.plot(stats.episode_lengths)

    # plot the rewards value over time (smoothed)
    rewards_smoothed = pd.Series(stats.episode_rewards) \
        .rolling(smoothing_window, min_periods=smoothing_window).mean()
    ax2.plot(rewards_smoothed)

    ax2.set_xlabel("Episodes")
    ax1.set_ylabel("Episode length")
    ax2.set_ylabel("Episode Reward (Smoothed)")
    plt.setp(ax1.get_xticklabels(), visible=False)

    return fig
