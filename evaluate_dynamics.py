import torch as tc
import numpy as np
import nolds.measures
import warnings
import os
from glob import glob
from bptt import models
import utils

MODEL_PATH = 'results/RNNTests/Lorenz/001' 

DATA_FILE_ORIGINAL = 'datasets/Lorenz/lorenz_data_chaos.npy'
TIMESTEP = 0.01
N_GENERATE = 30000
N_SUBSET = 20000

def load_model(model_id):
    model = models.Model()
    model.init_from_model_path(model_id)
    model.eval()
    return model

def is_model_id(path):
    run_nr = path.split('/')[-1]
    three_digit_numbers = {str(digit).zfill(3) for digit in set(range(1000))}
    return run_nr in three_digit_numbers

def get_model_ids(path):
    assert os.path.exists(path)
    if is_model_id(path):
        model_ids = [path]
    else:
        all_subfolders = glob(os.path.join(path, '**/*'), recursive=True)
        model_ids = [file for file in all_subfolders if is_model_id(file)]
    assert model_ids, 'could not load from path: {}'.format(path)
    return model_ids

def calculate_lambda(data_1d, ts, emb_dim, lag, min_tsep):
    warnings.filterwarnings("ignore", category=np.ComplexWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    try:
        lambda_val = nolds.measures.lyap_r(
            data_1d,
            emb_dim=emb_dim,
            lag=lag,
            min_tsep=min_tsep,
            tau=ts
        )
        return lambda_val
    except Exception as e:
        print(f"  [Error] : {e}")
        return 0

print(f"Loading model from: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print("Error: Model path not found. Please check MODEL_PATH variable.")
    exit()

data_original_tc = tc.tensor(utils.read_data(DATA_FILE_ORIGINAL))
model_ids = get_model_ids(MODEL_PATH)
model = load_model(*model_ids)

print(f"Generating free trajectory (N={N_GENERATE})...")
ts_generated_tc, _ = model.generate_free_trajectory(data_original_tc, N_GENERATE)
ts_generated_np = ts_generated_tc.detach().numpy()
print("Generation complete.")

data_original_1d = data_original_tc.numpy()[:N_SUBSET, 0]
data_generated_1d = ts_generated_np[:N_SUBSET, 0]

nolds_params = {
    'emb_dim': 10,
    'lag': 50,
    'min_tsep': 50,
    'ts': TIMESTEP
}

print(f"\nCalculating lambda_max for ORIGINAL data (N={N_SUBSET})...")
lambda_original = calculate_lambda(data_original_1d, **nolds_params)

print(f"\nCalculating lambda_max for GENERATED data (N={N_SUBSET})...")
lambda_generated = calculate_lambda(data_generated_1d, **nolds_params)

print("\n--- Dynamics Evaluation Result ---")
print(f"  Original Data lambda_max (1/s): {lambda_original:.4f}")
print(f"  Generated Data lambda_max (1/s): {lambda_generated:.4f}")

if lambda_generated > 0:
    difference = (np.abs(lambda_original - lambda_generated) / lambda_original) * 100
    print(f"\nSUCCESS: Generated data is chaotic and lambda_max is {difference:.2f}% different from the original.")
else:
    print(f"\nFAILURE: Generated data is NOT chaotic (lambda_max = {lambda_generated:.4f}).")
