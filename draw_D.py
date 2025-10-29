import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the file path
# file_path = 'lorenz.xlsx'
file_path = 'duffing_data_chaos.xlsx'

try:
    # Load the CSV file
    df = pd.read_excel(file_path)

    # Sort by 'tau' to ensure correct line plotting order
    df = df.sort_values(by='tau')

    # --- Plot 1: D_stsp ---
    plt.figure(figsize=(10, 6))
    
    # Plot each model's D_stsp vs. tau
    # Using different markers for better accessibility, similar to the paper
    # Note: matplotlib will automatically handle NaNs (like in 'RNN-D_stsp')
    # by creating a break in the line.
    plt.plot(df['tau'], df['RNN-D_stsp'], label='RNN', marker='o')
    plt.plot(df['tau'], df['LSTM-D_stsp'], label='LSTM', marker='s')
    plt.plot(df['tau'], df['PLRNN-D_stsp'], label='PLRNN', marker='^')
    
    # Add labels and title (using LaTeX for math symbols)
    plt.xlabel('learning interval tau')
    plt.ylabel('$D_{stsp}$')
    plt.title('Attractor Geometry Comparison ($D_{stsp}$) vs. Learning Interval')
    
    # Add legend and grid
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Ensure layout is clean
    plt.tight_layout()
    
    # Save the figure
    # plt.show()
    # plt.savefig('lorenz_dstsp_plot.png')
    plt.savefig('duffing_dstsp_plot.png')
    print("Saved 'lorenz_dstsp_plot.png'")

    # --- Plot 2: D_H ---
    plt.figure(figsize=(10, 6))
    
    # Plot each model's D_H vs. tau
    plt.plot(df['tau'], df['RNN-D_H'], label='RNN', marker='o')
    plt.plot(df['tau'], df['LSTM-D_H'], label='LSTM', marker='s')
    plt.plot(df['tau'], df['PLRNN-D_H'], label='PLRNN', marker='^')
    
    # Add labels and title (using LaTeX for math symbols)
    plt.xlabel('learning interval tau')
    plt.ylabel('$D_{H}$')
    plt.title('Power Spectra Comparison ($D_{H}$) vs. Learning Interval')
    
    # Add legend and grid
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Ensure layout is clean
    plt.tight_layout()
    
    # Save the figure
    # plt.show()
    # plt.savefig('lorenz_dh_plot.png')
    plt.savefig('duffing_dh_plot.png')
    print("Saved 'lorenz_dh_plot.png'")

except Exception as e:
    print(f"An error occurred during plotting: {e}")
