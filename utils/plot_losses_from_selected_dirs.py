import os
import pickle
import matplotlib.pyplot as plt

# === CONFIG: List of directories containing results.pkl ===
dir_names = ['./out/graphs_seq_1_epochs_100_length_comparison', 
             './out/graphs_seq_2_epochs_100_length_comparison',
             './out/graphs_seq_4_epochs_100_length_comparison',
             './out/graphs_seq_8_epochs_100_length_comparison',
             './out/graphs_seq_12_epochs_100_length_comparison',
             './out/graphs_seq_16_epochs_100_length_comparison',
             './out/graphs_seq_20_epochs_100_length_comparison']
# Replace with your actual list of directory names

# === Storage for results ===
all_rmse_data = []

# === Load RMSE errors from each directory ===
for dir_name in dir_names:
    file_path = os.path.join(dir_name, 'results.pkl')
    
    if not os.path.exists(file_path):
        print(f"⚠️  File not found: {file_path}")
        continue
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if 'rmse_errors' not in data:
            print(f"❌ Key 'rmse_errors' not found in {file_path}")
            continue
        
        rmse_errors = data['rmse_errors']
        
        # Ensure rmse_errors is a list or 1D array
        if isinstance(rmse_errors, (list, tuple)) or (hasattr(rmse_errors, '__iter__') and hasattr(rmse_errors, 'shape')):
            all_rmse_data.append((dir_name, rmse_errors))
        else:
            # If it's a scalar, convert to list for consistency
            all_rmse_data.append((dir_name, [rmse_errors]))
            print(f"💡 '{dir_name}': rmse_errors was scalar, treating as single-point series.")
            
    except Exception as e:
        print(f"💥 Error loading {file_path}: {e}")

# === Plot all RMSE errors together for comparison ===
plt.figure(figsize=(10, 6))

for dir_name, rmse_errors in all_rmse_data[1:]:
    x = list(range(len(rmse_errors)))  # X-axis: step, epoch, etc.
    sequence_length = dir_name.replace("_length_comparison", "")
    sequence_length = sequence_length.replace("./out/graphs_", "")
    plt.plot(x, rmse_errors, marker='o', label=sequence_length, linewidth=2, markersize=4)

# === Styling ===
plt.xlabel('Step / Epoch', fontsize=12)
plt.ylabel('RMSE Error', fontsize=12)
plt.title('Comparison of RMSE Errors Across Experiments', fontsize=14)
plt.legend(title='Experiment', fontsize=10)
plt.grid(True, alpha=0.4)
plt.tight_layout()

plt.savefig('losses_comparison_starting_with_2.png')