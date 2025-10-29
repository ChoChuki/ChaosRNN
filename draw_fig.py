import matplotlib.pyplot as plt
import numpy as np
import os 
import pandas as pd
import torch.nn as nn
import matplotlib
import torch as tc
import utils
from glob import glob
from evaluation import klx
from evaluation import mse
from evaluation.pse import power_spectrum_error, power_spectrum_error_per_dim
from bptt import models
import math
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import pearsonr
from torch import optim
from bptt import dataset
import main_eval
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import sem
import pickle

def load_model(model_id):
    model = models.Model()
    model.init_from_model_path(model_id)
    model.eval()
    return model

def is_model_id(path):
    """Check if path ends with a three digit, e.g. save/test/001 """
    run_nr = path.split('/')[-1]
    three_digit_numbers = {str(digit).zfill(3) for digit in set(range(1000))}
    return run_nr in three_digit_numbers

def get_model_ids(path):
    """
    Get model ids from a directory by recursively searching all subdirectories for files ending with a number
    """
    assert os.path.exists(path)
    if is_model_id(path):
        model_ids = [path]
    else:
        all_subfolders = glob(os.path.join(path, '**/*'), recursive=True)
        model_ids = [file for file in all_subfolders if is_model_id(file)]
    assert model_ids, 'could not load from path: {}'.format(path)
    return model_ids

font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 27}
import matplotlib
matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)
data_path= 'datasets/Roessler_chaos_rosenstein.npy'
model_path = 'results/RNNTests/Roessler/007'

data = tc.tensor(utils.read_data(data_path))

model_ids = get_model_ids(model_path)
model = load_model(*model_ids)
N = 30000

ts,z = model.generate_free_trajectory(data,30000)
ts = ts.detach()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(data[:N,0],data[:N,1],data[:N,2],linewidth=0.2)
ax.plot(ts[:,0],ts[:,1],ts[:,2],linewidth=0.1)
value = 1
ax.w_xaxis.set_pane_color((value, value, value, value))
ax.w_yaxis.set_pane_color((value, value, value, value))
ax.w_zaxis.set_pane_color((value, value, value, value))
plt.show()
# plt.savefig("PLRNN_duffing_16.jpg",dpi=600,format="jpg")
