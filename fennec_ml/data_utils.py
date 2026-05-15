# Micah Yarbrough
# 10/9/24

import os
import pandas as pd
import numpy as np
import re
import json
import glob
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Micah Yarbrough
# 11/7/25
# This function will extract and clean up data from .xlsx files, saving them as .csv's
def data_cleaner(filepath, savepath, overwrite= False, skip= False, varspath= "vars_of_interest.json", downsample= True):
    """
    Preprocesses .xlsx files into fennec question-usefull .csv files.

    Args:
        filepath (string): The .xlsx file to process.
        savepath (string): The folder to save the .csv file.
        overwrite (bool): Skips the overwrite checker if true.
        skip (bool): Skips duplicate files instead of checking or overwriting if true.
        varspath (string): The vars-of-interest.json path. Defaults to same folder as THIS script.
        downsample (bool): If true, it will downsample to the lowest sample rate, if false it will upsample to the highest

    Relies on the vars_of_interest.json file to determine what data is wanted
    """

    # --- FILE & FOLDER CHECKS ---
    if not os.path.isfile(filepath): # does .xlsx file exist?
        raise FileNotFoundError(
            f"Error: Input file '{filepath}' does not exist."
        )

    if not filepath.lower().endswith(".xlsx"): # is the file an .xlsx file?
        raise ValueError(
            f"Error: Input file '{filepath}' is not an .xlsx file."
        )

    if not os.path.isdir(savepath): # does savepath exist?
        raise FileNotFoundError(
            f"Error: Save path '{savepath}' not found. "
            f"Please create the directory before running the function."
        )

    if not os.path.isfile(varspath): # does vars_of_interest.json exist?
        raise FileNotFoundError(
            f"Error: Vars-of-interest file '{varspath}' not found. "
            f"Ensure the JSON file is in the same folder as this script OR pass the filepath via arg: varspath =\" \"."
        )


    inputfile = os.path.basename(filepath) #get the name of the xlsx file
    filename = inputfile[:-5] #remove the ".xlsx" from the end
    
    # --- OVERWRITE CHECKER ---
    if (overwrite == False): #skip is overwrite was set to True
        #check savepath to see if the .xlsx file has already been processed
        for csvfile in os.listdir(savepath):
            if os.path.basename(csvfile) == f"{filename}.csv":
                if(skip == True):
                    print(f"{inputfile} skipped due to existing duplicate.")
                    return False
                
                #if a match is found, prompt the user before overwriting the file
                user_input = ""
                while (user_input != "y") and (user_input != "n"):
                    user_input = input("ARE YOU SURE YOU WANT TO OVERWRITE THIS FILE? (y,n)-->")
                if user_input == "n":
                    print(f"{inputfile} not processed due to user input.")
                    return False
    
    # --- PREPROCESSING ---
    """
    For each sheet, we want to take the relevant data at each timestamp
       and package it together in an 2D array[x][y] where x is each timestamp and y is each datatype

        [[GyrX0, GyrY0, ..., AccZ0],
         [GyrX1, GyrY1, ..., AccZ1],
         [GyrX2, GyrY2, ..., AccZ2], ...]

        Then the arrays for each sheet get combined so EVERY datatype is stored at each timestamp.
        That combined array gets saved as a .csv file.
    """

    xl = pd.ExcelFile(filepath) #load the .xlsx into a pandas array (takes the longest)

    #read the vars_of_interest file
    with open(varspath, "r") as f:
        vars_of_interest = json.load(f) #convert json file to dict

    extracted_data = {key: None for key in vars_of_interest} #stores only the designated data from each xl sheet

    sheets = vars_of_interest.keys()

    #get the correct data from each sheet in the pandas array
    for sheet, variables in vars_of_interest.items():
        #make sure sheet exist in .xlsx
        if sheet not in xl.sheet_names:
            raise ValueError(
                f"Error: The sheet '{sheet}' was not found in {inputfile}. "
                f"Available sheets: {xl.sheet_names}"
            )
        
        df = xl.parse(sheet) #parse the correct sheet

        #make sure vars exist in sheet
        missing_cols = [col for col in variables if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Error: In sheet '{sheet}', the following columns are missing: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )        

        extracted_data[sheet] = df[variables].to_numpy(dtype=float) #save the designated data to extracted_data as a numpy array
        
    #FREQUENCY CORRECTION    
    lengths = [len(arr) for arr in extracted_data.values()] # get total timesteps for each sheet
    min_len = min(lengths) # find shortest sheet
    ratios = [round(l / min_len) for l in lengths]

    # Scale each sheet according to its scaling ratio
    for i, (sheet_name, sheet) in enumerate(extracted_data.items()):
        ratio = ratios[i]
        sheet = sheet[::ratio]
        # Downsample
        if downsample == True and ratio != 1:
            extracted_data[sheet_name] = sheet  # reassign if you want to keep the change
            print(f"Sheet {sheet_name}, downsampled by {ratio}")
        # Upsample
        elif downsample == False:
            ratio = max(ratios)/ratio
            if ratio != 1:
                extracted_data[sheet_name] = np.repeat(extracted_data[sheet_name], max(ratios)/ratio, axis=0).astype(float)
                print(f"Sheet {sheet_name}, upsampled by {ratio}")

    # Recalculate the lengths
    lengths = [len(arr) for arr in extracted_data.values()]
    min_len = min(lengths)
 
    # --- LENGTH CORRECTION ---
    #Truncate all arrays to the minimum length
    for sheet in extracted_data: 
        extracted_data[sheet] = extracted_data[sheet][:min_len] 
    lengths = [len(arr) for arr in extracted_data.values()]

    #stack all the data from each sheet into one single 2D array
    csv_data = np.hstack(list(extracted_data.values()))

    # --- SAVE AS .CSV ---
    df = pd.DataFrame(csv_data)
    new_path = os.path.join(savepath, inputfile.replace('xlsx', 'csv')) # Create new path
    df.to_csv(new_path, index=False, encoding='utf_8') # Save to new path

    print(f"{inputfile} processed and saved to {savepath} as {filename}.csv")
    xl.close()
    return True

# Micah Yarbrough and Wills Kookogey
# 10/21/25
# This function will calls the data cleaner for every .xlsx file in a given directory
def folder_cleaner(excel_dir, savepath, overwrite = False, skip = False, varspath = "vars_of_interest.json", downsample= True):
    """
    Preprocesses a folder of .xlsx files into fennec question-usefull .csv files.

    Args:
        excel_dir (string): The folder of .xlsx files to process.
        savepath (string): The folder to save the .csv file.
        overwrite (bool): Skips the overwrite checker if true.
        skip (bool): Skips duplicate files instead of checking or overwriting if true.
        varspath (string): The vars-of-interest.json path. Defaults to same folder as THIS script.

    Relies on the vars_of_interest.json file to determine what data is wanted
    """

    # --- FILE & FOLDER CHECKS ---
    if not os.path.isdir(excel_dir): # does savepath exist?
        raise FileNotFoundError(
            f"Error: Save path '{excel_dir}' not found. "
            f"Please create the directory before running the function."
        )
    
    # --- CLEAN FOLDER ---
    # for each file in the folder, run data_cleaner
    for file in os.listdir(excel_dir):
        filepath = os.path.join(excel_dir, file)
        if filepath.lower().endswith(".xlsx"):
            data_cleaner(filepath, savepath, overwrite, skip, varspath, downsample)




# Wills Kookogey, Micah Yarbrough & Luke Fagg
# 10/09/25
# This function segments the data and splits it into train/validate/test.
def csv_to_numpy(csv_dir, start_index=0, end_index=None):
    """
    Convert csv's into list of numpy arrays, 1 per flight
    
    Args:
        csv_dir: The path (including the folder name) of cleaned data
    
    Returns:
        data: A list of numpy arrays holding data for each flight

    """
    
    output_data = []
    
    if not os.path.isdir(csv_dir): # does savepath exist?
        raise FileNotFoundError(
            f"Error: Save path '{csv_dir}' not found. "
            f"Please create the directory before running the function."
        )
        
    if end_index is not None and end_index < start_index:
        raise ValueError(
            f"Error: end_index ({end_index}) must be greater than or equal to start_index ({start_index})."
        )
    
    # Paths
    clean_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    # SORTED() IS ESSENTIAL TO ENSURE FILES MATCH get_labels() LABELS

    # Import and convert data to list of numpy arrays
    for file in clean_files:
        data = pd.read_csv(file).to_numpy()
        
        # print file name
        print(f"Importing {os.path.basename(file)} with shape {data.shape}...")
        
        # add deltas to data (so each timestep has the change in value from the previous timestep for each feature)
        # deltas = np.diff(data, axis=0, prepend=data[[0]])  # prepend first row so shape stays the same
        # data = np.concatenate([data, deltas], axis=1)      # [timesteps, features*2]
        
        # cut data to desired range if start/end index is specified
        if start_index > 0:
            data = data[start_index:]
        if end_index is not None:
            data = data[:end_index]
        
        output_data.append(data)
    
    return output_data




# Wills Kookogey, Luke Fagg & Micah Yarbrough
# 02/17/26
# This function will normalize cleaned data before it goes into the dataset dictionary
# NORMALIZING means scaling the data between 
def scale(train_val_data_input, test_data_input, scaler="standard"):
    """
    Return 3 2D arrays of `SCALED` data from split data
    Normalizes train, validate, and test data according to train data to avoid data leakage
    
    Args:
        train_val_data_input: A list of numpy arrays, 1 per file, all for training
        test_data_input: A list of numpy arrays, 1 per file, all for testing
    
    Returns:
        train_val_data_norm: A list of numpy arrays holding scaled train data
        test_data_norm: A list of numpy arrays holding scaled test data

    """
    
    if scaler == "minmax":
        scaler = MinMaxScaler(feature_range=(-1, 1))
    elif scaler == "standard":
        scaler = StandardScaler()
    
    
    train_data = []
    train_data_norm = []
    test_data_norm = []

    # load train data so the scaler fits to the train data without being affected by
    # unseen validation and test data (to avoid data leakage)
    train_val_data = np.vstack(train_val_data_input)

    # fit scaler to train data
    scaler.fit(train_val_data)

    # Scale all the data
    for arr in train_val_data_input:
        train_data_norm.append(scaler.transform(arr))
    
    for arr in test_data_input:
        test_data_norm.append(scaler.transform(arr))
    
    return train_data_norm, test_data_norm




# Micah Yarbrough 
# 10/9/25
# Reads all filenames in a folder and returns 1D CG characterization labels
def get_1D_CG_labels(csv_dir):
    """
    Reads all filenames in a folder and returns 1D CG characterization labels

    Args:
        csv_dir (string): Directory of .csv files from which to get labels
    
    Returns:
        labels (list): A list of all the characterization labels

    """
    # the output labels list
    labels = []

    # Regex patterns for reading 2024-2025 1D and 2D CG flight data files
    patternAA = r'clip\d+B\d+_(AA)_(L|S)_\d\.csv$'
    patternBB = r'clip\d+B\d+_(BB)_(L|S)_\d\.csv$'
    patternCC = r'clip\d+B\d+_(CC)_(L|S)_\d\.csv$'
    patternDD = r'clip\d+B\d+_(DD)_(L|S)_\d\.csv$'
    patternEE = r'clip\d+B\d+_(EE)_(L|S)_\d\.csv$'

    # --- CHECK FILEPATH ---
    if not os.path.isdir(csv_dir): # does savepath exist?
        raise FileNotFoundError(
            f"Error: Save path '{csv_dir}' not found. "
            f"Please create the directory before running the function."
        )

    # --- GET FILEPATHS ---
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    # SORTED() IS ESSENTIAL TO ENSURE FILES MATCH normalize() data

    # append label of each file to labels list
    for file in csv_files:
        filename = os.path.basename(file)
        if re.search(patternAA, filename):
            labels.append("AA")
        if re.search(patternBB, filename):
            labels.append("BB")
        if re.search(patternCC, filename):
            labels.append("CC")
        if re.search(patternDD, filename):
            labels.append("DD")
        if re.search(patternEE, filename):
            labels.append("EE")
    
    return labels


# Micah Yarbrough 
# 10/17/25
# Reads all filenames in a folder and returns 2D CG characterization labels
def get_2D_CG_labels(csv_dir):
    """
    Reads all filenames in a folder and returns 2D CG characterization labels

    Args:
        csv_dir (string): Directory of .csv files from which to get labels
    
    Returns:
        labels (list): A list of all the characterization labels

    """
    # the output labels list
    labels = []

    # Regex patterns for reading 2024-2025 1D and 2D CG flight data files
    patternAAP = r'^\d+G_(AAP)_(L|H|S)_\d+\.csv$'
    patternAAC = r'^\d+G_(AAC)_(L|H|S)_\d+\.csv$'
    patternAAS = r'^\d+G_(AAS)_(L|H|S)_\d+\.csv$'

    patternBBP = r'^\d+G_(BBP)_(L|H|S)_\d+\.csv$'
    patternBBC = r'^\d+G_(BBC)_(L|H|S)_\d+\.csv$'
    patternBBS = r'^\d+G_(BBS)_(L|H|S)_\d+\.csv$'

    patternCCP = r'^\d+G_(CCP)_(L|H|S)_\d+\.csv$'
    patternCCC = r'^\d+G_(CCC)_(L|H|S)_\d+\.csv$'
    patternCCS = r'^\d+G_(CCS)_(L|H|S)_\d+\.csv$'

    # --- CHECK FILEPATH ---
    if not os.path.isdir(csv_dir): # does savepath exist?
        raise FileNotFoundError(
            f"Error: Save path '{csv_dir}' not found. "
            f"Please create the directory before running the function."
        )

    # --- GET FILEPATHS ---
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    # SORTED() IS ESSENTIAL TO ENSURE FILES MATCH normalize() data

    # append label of each file to labels list
    for file in csv_files:
        filename = os.path.basename(file)
        if re.search(patternAAP, filename):
            labels.append("AAP")
        if re.search(patternAAC, filename):
            labels.append("AAC")
        if re.search(patternAAS, filename):
            labels.append("AAS")

        if re.search(patternBBP, filename):
            labels.append("BBP")
        if re.search(patternBBC, filename):
            labels.append("BBC")
        if re.search(patternBBS, filename):
            labels.append("BBS")

        if re.search(patternCCP, filename):
            labels.append("CCP")
        if re.search(patternCCC, filename):
            labels.append("CCC")
        if re.search(patternCCS, filename):
            labels.append("CCS")
    
    return labels


# Glory to the Father, and to the Son, and to the Holy Spirit: as
# it was in the beginning, is now, and will be for ever. Amen. 