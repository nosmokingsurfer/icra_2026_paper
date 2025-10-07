import os
import re
import random



def natural_sort_key(filename):
    # Extracts numbers and sorts them numerically
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]



def generate_data_split(path_to_splines, val_ratio=0.2, is_random=False):
    assert 0 <= val_ratio < 1.0, RuntimeError("val_ration should be between 0 and 1")

    all_spline_files = os.listdir(f"{path_to_splines}/splines")
    amount_of_splines = len(all_spline_files)

    amount_of_train = int(amount_of_splines * (1 - val_ratio))
    # amount_of_val = amount_of_splines - amount_of_train

    train_list, val_list = [], []
    if is_random:
        random.shuffle(all_spline_files)
        paths = all_spline_files
    else:
        paths = sorted(all_spline_files, key=natural_sort_key)

    train_list = paths[:amount_of_train]
    val_list = paths[amount_of_train:]

    with open(f"{path_to_splines}/train_split.txt", "w") as file:
        for item in train_list:
            file.write(item + "\n")

    with open(f"{path_to_splines}/val_split.txt", "w") as file:
        for item in val_list:
            file.write(item + "\n")



    