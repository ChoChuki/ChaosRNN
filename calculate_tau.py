import numpy as np
import nolds.measures
import warnings

# --- 1. Parameters ---
# This script is for the lorenz_data_chaos.npy file
DATA_FILE = 'datasets/Roessler_chaos_rosenstein.npy'

# CRITICAL: This TIMESTEP (dt) must match how the data was generated.
# From paper Appendix A.4, the Lorenz data was integrated with dt=0.01
TIMESTEP = 0.01

# --- 2. Load Data ---
print(f"Loading data from {DATA_FILE}...")
try:
    data = np.load(DATA_FILE)
except Exception as e:
    print(f"Error loading data: {e}")
    print("Please ensure the file path is correct.")
    exit()

# Using a subset for calculation is much faster and usually sufficient
# Paper's plots (e.g., Fig 4a) also show divergence calculated over a limited time
data_subset = data[:20000]
# Use only the first dimension (x-coordinate) for the calculation
data_subset_1d = data_subset[:, 0]
print(f"Using a subset of {len(data_subset)} points for calculation.")

# --- 3. Calculate Maximal Lyapunov Exponent (lambda_max) ---
print("Calculating Maximal Lyapunov Exponent (lambda_max)...")
print("(This may take a minute)")

# Suppress warnings from nolds (it's often noisy during fitting)
warnings.filterwarnings("ignore", category=np.ComplexWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    # nolds.lyap_r calculates the slope of the log-divergence vs. time
    # This slope is lambda_max (in units of 1/seconds if tau is in seconds)
    # We use the Rosenstein method (fit='poly', dimension=1)
    # We must provide the data's sampling interval (tau=TIMESTEP)
    
    # For multi-dimensional data (N, D), nolds.lyap_r handles it directly.
    # We set min_tsep (mean period) to avoid correlated neighbors. 
    # For Lorenz, mean period is ~0.5-1s. At dt=0.01, that's 50-100 steps.
    lambda_max_per_sec = nolds.measures.lyap_r(
        data_subset_1d,
        lag=50,         # Lag used for finding neighbors
        min_tsep=50,    # Mean period (in steps)
        tau=TIMESTEP    # Timestep between data points (in seconds)
    )

except Exception as e:
    print(f"\nError during calculation: {e}")
    print("Please ensure 'nolds' is installed (`pip install nolds`).")
    exit()

if lambda_max_per_sec <= 0:
    print(f"Calculation failed or system is not chaotic (lambda_max = {lambda_max_per_sec:.4f})")
else:
    # --- 4. Calculate Tau (n_interleave) ---
    # Paper (Eq 17 & A.4) uses the formula: tau_pred = ln(2) / (lambda_max * dt)
    # nolds returns lambda_max in (1/seconds)
    # We want tau_pred in (steps)
    
    # tau_pred (in seconds) = ln(2) / lambda_max_per_sec
    tau_pred_seconds = np.log(2) / lambda_max_per_sec
    
    # tau_pred (in steps) = tau_pred (in seconds) / TIMESTEP (in seconds/step)
    tau_pred_steps = tau_pred_seconds / TIMESTEP
    
    # This is mathematically equivalent to the paper's formula in Appendix A.4
    
    print(f"\n--- Calculation Complete ---")
    print(f"Calculated lambda_max (1/s): {lambda_max_per_sec:.4f}")
    print(f"Calculated tau_pred (seconds): {tau_pred_seconds:.4f} s")
    print(f"Calculated tau_pred (steps): {tau_pred_steps:.2f} steps")
    print(f"\nRecommended --n_interleave parameter: {int(round(tau_pred_steps))}")
