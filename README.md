# fennec-ml

A suite of data science and machine learning tools for use by the LeTourneau University FENNEC senior design team.

![Fennec Logo](https://raw.githubusercontent.com/afdavisLETU/fennec-ml/main/assets/Fennec%2025-26.png)
[![PyPI version](https://img.shields.io/pypi/v/fennec-ml.svg)](https://pypi.org/project/fennec-ml/)

## Instalation

```bash
pip install fennec-ml
```

## Quick Start

### Data Utils

```python
import os
import fennec_ml as fn

# setup
root_dir = os.getcwd()
excel_dir = os.path.join(root_dir, "Raw_Data")
csv_dir = os.path.join(root_dir, "Proccessed_Data")
timesteps = 60

# excel to csv
fn.folder_cleaner(excel_dir)

# normalize and get labels
norm_data = fn.normalize(csv_dir)
labels = fn.get_CG_labels(csv_dir)

# segment and sort into train, validate, and test datasets
dataset_dict = fn.segment_and_split(norm_data, labels, timesteps)

# use dataset_dict
training_sets = dataset_dict['Training_Set']['sets']
training_labels = dataset_dict['Training_Set']['labels']
```

## Example Project Structure

```txt
myproject/
├── data/
│   ├── raw_data/
│   │   ├── train_val/
│   │   │   └── flight123_AA_L.csv
│   │   └── test/
│   │       └── flight456_BB_R.csv
│   └── processed_data/
│   │   ├── train_val/
│   │   │   └── flight123_AA_L.csv
│   │   └── test/
│   │       └── flight456_BB_R.csv
├── saved_models/
│   └── ...
├── vars_of_interest.json
└── project_dev.ipynb
```

## Features

### data_cleaner()

Preprocesses .xlsx files into fennec question-usefull .csv files.

**Args:**

- filepath (string): The .xlsx file to process.
- savepath (string): The folder to save the .csv file.
- overwrite (bool): Skips the overwrite checker if true.
- skip (bool): Skips duplicate files instead of checking or overwriting if true.
- varspath (string): The vars-of-interest.json path. Defaults to same folder as THIS script.
- downsample (bool): If true, it will downsample to the lowest sample rate, if false it will upsample to the highest

Relies on the vars_of_interest.json file to determine what data is wanted

```python
fn.data_cleaner(excel_filename, overwrite= True)
```

---

### folder_cleaner()

Preprocesses a *folder* of .xlsx files into fennec question-usefull .csv files.

**Args:**

- excel_dir (string): The folder of .xlsx files to process.
- savepath (string): The folder to save the .csv file.
- overwrite (bool): Skips the overwrite checker if true.
- skip (bool): Skips duplicate files instead of checking or overwriting if true.
- varspath (string): The vars-of-interest.json path. Defaults to same folder as *this* script.

Relies on the vars_of_interest.json file to determine what data is wanted

```python
fn.folder_cleaner(excel_dir, skip= True)
```

---

### csv_to_numpy()

Converts csv's into list of numpy arrays, 1 per flight

**Args:**

- csv_dir (string): The path (including the folder name) of cleaned data

**Returns:**

- data (list): A list of numpy arrays holding data for each flight

```python
data = fn.csv_to_numpy(csv_dir)
```

---

### scale()

Return 3 2D arrays of `SCALED` data from split data. Normalizes train, validate, and test data according to train data to avoid data leakage

**Args:**

- train_val_data_input (list): A list of numpy arrays, 1 per file, all for training
- test_data_input (list): A list of numpy arrays, 1 per file, all for testing

**Returns:**

- norm_data (list): A list of numpy arrays holding normalized data

```python
scaled_data = fn.scale(train_val_data, test_data, scaler= "standard")
```

---

### get_1D_CG_labels()

Reads all filenames in a folder and returns 1d CG characterization labels

**Args:**

- csv_dir (string): Directory of .csv files from which to get labels

**Returns:**

- labels (list): A list of all the characterization labels

```python
cg_labels = fn.get_1D_CG_labels(csv_dir)
```

---

### get_2D_CG_labels()

Reads all filenames in a folder and returns 2d CG characterization labels

**Args:**

- csv_dir (string): Directory of .csv files from which to get labels

**Returns:**

- labels (list): A list of all the characterization labels

```python
cg_labels = fn.get_2D_CG_labels(csv_dir)
```

## Contributers

- Luke Fagg  (Team Lead)
- Micah Yarbrough (Pilot ID)
- Wills Kookogey (Fault ID)
- Justin Hawk (3D CG)
