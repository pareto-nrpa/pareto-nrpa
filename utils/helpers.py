import copy
import os
import random
from datetime import datetime

import numpy as np
import networkx as nx
from decimal import Decimal
import pydot
import matplotlib.pyplot as plt
import seaborn as sns


def create_now_folder(path):
    folder = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    os.mkdir(os.path.join(path, folder))
    return folder


def ewma(data, window):

    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out

def running_avg(data, window_size):
    res = np.zeros(len(data)-window_size)
    sum = 0
    for i in range(window_size):
        sum += data[i]
    for i in range(len(data)-window_size):
        res[i] = sum / window_size
        sum -= data[i]
        sum += data[i + window_size]
    return res

def configure_seaborn(**kwargs):
    sns.set_context("notebook")
    sns.set_theme(sns.plotting_context("notebook", font_scale=1), style="whitegrid")
    palette = ["#3D405B", "#E08042", "#54AB69", "#CE2F49", "#A26EBF", "#7D4948", "#D12AA2", "#E0D06F", "#6F9AA7", "#3359C4",
               "#76455B"]
    sns.set_palette(palette)

def softmax_temp(x, temp=1.0):
    x = x / temp
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def z_normalize(data):
    return (data - np.mean(data)) / np.std(data)

def normalize(data, log=False):
    if log is True:
        data = np.log(data)
    return (data - np.min(data)) / (np.max(data) - np.min(data))