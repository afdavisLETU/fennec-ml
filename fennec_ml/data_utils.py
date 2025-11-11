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




# Luke Fagg & Micah Yarbrough
# 10/9/25
# This function will normalize and add weights to cleaned data before it goes into the dataset class
# NORMALIZING means scaling the data between the min and max values
def normalize(csv_dir, weights = [None], offsets = [None], scaler= MinMaxScaler(feature_range=(-1,1))):
    """
    Return a 3D array of NORMALIZED data from cleaned csv's
    NORMALIZING means scaling the data between the min and max values
    
    Args:
        csv_dir: The path (including the folder name) of cleaned data
        weights: An optional array of weights corresponding to each column
        offsets: An optional array of offsets corresponding to column
    
    Returns:
        norm_data: A list of numpy arrays holding NORMALIZED data

    """
    
    if not os.path.isdir(csv_dir): # does savepath exist?
        raise FileNotFoundError(
            f"Error: Save path '{csv_dir}' not found. "
            f"Please create the directory before running the function."
        )
    # Paths
    clean_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    # SORTED() IS ESSENTIAL TO ENSURE FILES MATCH get_labels() LABELS
    all_data = []
    norm_data = []

    #load all data so the scaler fits to the WHOLE data range
    for file in clean_files:
        df = pd.read_csv(file) #get csv data to PANDAS
        arr = df.to_numpy() #make pandas data numpy
        if offsets[0] == None:
            offsets = [0] * df.shape[1]
        df = df - offsets
        all_data.append(arr)

    all_data = np.vstack(all_data)

    scaler.fit(all_data)

    #Import and scale the data
    for file in clean_files:
        df = pd.read_csv(file)
        if offsets[0] == None:
            offsets = [0] * df.shape[1]
        scaled_data = scaler.transform(df.to_numpy())
        if weights[0] == None:
            weights = [1] * df.shape[1]
        scaled_data = scaled_data * weights
        norm_data.append(scaled_data)
    
    return norm_data


# Luke Fagg & Micah Yarbrough
# 10/9/25
# This function will standardize and add weights to cleaned data before it goes into the dataset class
# STANDARDIZING means scaling the data so the mean = 0 and the std deviation = 1
def standardize(csv_dir, weights = [None], offsets = [None], scaler= StandardScaler()):
    """
    Return a 3D array of STANDARDIZED data from cleaned csv's
    STANDARDIZING means scaling the data so the mean = 0 and the std deviation = 1
    
    Args:
        csv_dir: The path (including the folder name) of cleaned data
        weights: An optional array of weights corresponding to each column
        offsets: An optional array of offsets corresponding to column
    
    Returns:
        stand_data: A list of numpy arrays holding STANDARDIZED data

    """
    
    if not os.path.isdir(csv_dir): # does savepath exist?
        raise FileNotFoundError(
            f"Error: Save path '{csv_dir}' not found. "
            f"Please create the directory before running the function."
        )
    # Paths
    clean_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    # SORTED() IS ESSENTIAL TO ENSURE FILES MATCH get_labels() LABELS
    all_data = []
    stand_data = []

    #load all data so the scaler fits to the WHOLE data range
    for file in clean_files:
        df = pd.read_csv(file) #get csv data to PANDAS
        arr = df.to_numpy() #make pandas data numpy
        if offsets[0] == None:
            offsets = [0] * df.shape[1]
        df = df - offsets
        all_data.append(arr)

    all_data = np.vstack(all_data)

    scaler.fit(all_data)

    #Import and scale the data
    for file in clean_files:
        df = pd.read_csv(file)
        if offsets[0] == None:
            offsets = [0] * df.shape[1]
        scaled_data = scaler.transform(df.to_numpy())
        if weights[0] == None:
            weights = [1] * df.shape[1]
        scaled_data = scaled_data * weights
        stand_data.append(scaled_data)
    
    return stand_data


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


# Wills Kookogey
# 11/4/25
# Reads all filenames in a folder and returns FID characterization labels
def get_FID_labels(csv_dir):
    """
    Reads all filenames in a folder and returns FID characterization labels

    Args:
        csv_dir (string): Directory of .csv files from which to get labels
    
    Returns:
        labels (list): A list of all the characterization labels

    """
    # the output labels list
    labels = []

    # Regex patterns for reading 2024-2025 1D and 2D CG flight data files
    patternL = r'^\d+G_(L).csv$'
    patternR = r'^\d+G_(R).csv$'
    patternLR = r'^\d+G_(LR).csv$'
    patternNONE = r'^\d+G_(NONE).csv$'

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
        if re.search(patternL, filename):
            labels.append("L")
        if re.search(patternR, filename):
            labels.append("R")
        if re.search(patternLR, filename):
            labels.append("LR")
        if re.search(patternNONE, filename):
            labels.append("NONE")
    
    return labels


# Wills Kookogey & Micah Yarbrough
# 10-09-25
# This function segments the data and splits it into train/validate/test.
# It returns a dictionary
def segment_and_split(input_data, input_labels, timesteps, train_split=0.7, validate_split=0.15):
    """
    Segments, labels, and sorts data into dataset dictionary
    output shape is (batch, sequence, feature)

    Args:
        input_data (list): List of numpy arrays, 1 per file
        input_labels (list): List of characterization lables, 1 per file (should correspond to input_data)
        timesteps (int): length of desired segments
        train_split (float): Percentage of segments to save as training segments
        validate_split (float): Percentage of segments to save as validation segments
    
    Returns:
        output (dict): 3 labels: "Training_Set", "Validation_Set", and "Testing_Set"
            each set has the follwing labels: "sets" and "labels" 
                - "sets" : list of sets, corresponds to "labels"
                - "labels" : list of labels, corresponds to "sets"
            ex: output["Training_Set"]["sets"] 

    """
    # --- ERROR CHECKS ---
    if len(input_data) != len(input_labels):
        raise ValueError(
            f"Length mismatch: got {len(input_data)} arrays but {len(input_labels)} labels. "
            "They must be the same length."
        )

    # temp arrays
    all_segments = []
    all_labels = []

    # --- SPLIT ARRAYS ---
    # for each element in data array, split it into segments of length = timestamps
    # add a corresponding label to each respective index in the label list
    for index, array in enumerate(input_data):
        # find the maximum number of timesteps that can be cut
        cutoff = (len(array) // timesteps) * timesteps
        array = array[:cutoff] # trim array to have no leftover after cutting

        #split the array up into segments
        segmented_arrays = np.split(array, len(array) // timesteps)
        all_segments.extend(segmented_arrays) # add the array segments to all_segments
        all_labels.extend([input_labels[index]]*len(segmented_arrays)) # add the correct number of labels to all_labels

    # convert lists to numpy arrays
    all_segments = np.stack(all_segments)
    all_labels = np.array(all_labels)

    num_of_segments = len(all_labels)

    # --- SHUFFLE SEGMENTS ---
    # shuffle the arrays, keeping the segments and labels together
    rand_indices = np.random.permutation(num_of_segments)
    all_segments, all_labels = all_segments[rand_indices], all_labels[rand_indices]

    # --- SORT SEGMENTS ---
    # set indices to split time series data at
    train_end = int(num_of_segments*train_split)
    val_end = int(num_of_segments*(train_split+validate_split))

    # define output dict
    output = {
        "Training_Set": {"sets": all_segments[:train_end], "labels": all_labels[:train_end]},
        "Validation_Set": {"sets": all_segments[train_end:val_end], "labels": all_labels[train_end:val_end]},
        "Testing_Set": {"sets": all_segments[val_end:], "labels": all_labels[val_end:]}
    }

    # Print out completion message and the number of sets in each category
    print("All data segmented and sorted!")
    print(f"Training_Sets: {len(output['Training_Set']['labels'])}")
    print(f"Validation_Sets: {len(output['Validation_Set']['labels'])}")
    print(f"Testing_Sets: {len(output['Testing_Set']['labels'])}")

    #return split data
    return output

# Glory to the Father, and to the Son, and to the Holy Spirit: as
# it was in the beginning, is now, and will be for ever. Amen. 