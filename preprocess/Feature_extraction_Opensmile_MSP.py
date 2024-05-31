"""
This code processes a dataset for emotion recognition from audio files in the MSP (Multimodal Signal Processing) dataset. It extracts statistical features using OpenSMILE and saves them for further machine learning tasks. Here's a detailed breakdown:

Imports
numpy, os, sys, csv: Standard libraries for numerical operations, file and system operations, and CSV file handling.
scipy.stats: Library for statistical functions (kurtosis and skew).
Function Definition
csv_reader(add): Reads a CSV file and returns the data as a NumPy array, excluding the first two columns.
Data and Configuration
emotions_used: Maps emotions to numerical labels.
emotions_used_comp: Alternative emotion label mapping (not used in the code).
data_path: Path to the MSP dataset.
sessions: List of session directories to process.
framerate: Sample rate for audio files (44100 Hz).
Label, Data: Lists to store processed labels and data.
fix_len: Fixed length for splitting MFCC features (120 frames).
exe_opensmile: Path to the OpenSMILE executable for feature extraction.
path_config: Path to the OpenSMILE configuration file.
Main Processing Loop
The main loop iterates over each session and processes each file containing emotion labels:

Path Setup: Constructs paths for emotion labels and audio files.
Read Emotion Labels: Opens the emotion label file and reads line by line.
File Sorting: Sorts and groups files based on a specific criterion.
Feature Extraction:
Constructs the input and output filenames.
Calls OpenSMILE to extract MFCC features and save them to a CSV file.
Read Features: Reads the MFCC features from the CSV file using csv_reader.
Compute Statistics: Calculates statistical features (mean, max, standard deviation, skewness, kurtosis) from the MFCC features.
Graph Construction: Constructs a graph of features and labels.
Label Assignment: Assigns the numerical label to the emotion.
Data Collection: Appends the features and labels to the lists Data and Label.
Save Data
Finally, the processed data and labels are saved as NumPy arrays.

Example Usage
Here's an example of how the code operates:

Iterate through sessions: For each session in the dataset.
Iterate through emotion label files: For each emotion label file in the session.
Sort and group files: Group files based on a specific criterion.
Extract and process features:
Read emotion labels.
If the label is valid, extract features using OpenSMILE.
Read and process the features.
Compute statistical features from the MFCC features.
Append the features and labels to the lists.
Save the results: Store the processed data and labels in .npy files for later use.
Notes
File Sorting: The sorting logic groups files based on a candidate criterion derived from the filename.
Statistical Features: The code extracts mean, max, standard deviation, skewness, and kurtosis of MFCC features.
Label Filtering: Only files with labels present in emotions_used are processed.
Data Storage: Processed features and labels are saved as .npy files for efficient loading and further processing.
This pipeline prepares the data for further use in machine learning models, such as training a neural network for emotion recognition. The use of statistical features provides a compact representation of the audio signal, which can be effective for classification tasks.
"""
import numpy as np
import os
import sys
import csv
from scipy.stats import kurtosis, skew

def csv_reader(add):
    with open(outfilename, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        headers = next(reader)
        data = np.array(list(reader)).astype(float)
        
    return data[:,2:] 


#### Load dat
emotions_used = { 'A':0, 'H':1, 'N':2, 'S':3 }
emotions_used_comp = {'Neutral;':2, 'Anger;':0, 'Sadness;':3, 'Happiness;':1}
data_path = "path_to_the_MSP_directory"
sessions = ['session1', 'session2', 'session3', 'session4', 'session5', 'session6']
framerate = 44100

Label = []
Data = []
fix_len = 120

# openSMILE
exe_opensmile = "path_to_the_opensmile_exe_fileee"
path_config   = "path_to_the_opensmile_config_file"


for ses in sessions:
    emt_label_path = data_path + ses
    for file in os.listdir(emt_label_path):
        for file2 in os.listdir(emt_label_path+'/'+file):
            if file2=='S':
                sorted_list = []
                file_list = os.listdir(emt_label_path + '/' + file + '/' + file2)
                file_list.sort()
                while(len(file_list)!=0):
                    sorted_list.append(file_list[0])
                    cand = file_list[0].split('-')[-1]
                    for fil in file_list[1:]:
                        if cand in fil:
                            sorted_list.append(fil)
                            file_list.remove(fil)
                            break
                    file_list.remove(file_list[0])
                Graph = []
                Graph_label = []
                for file3 in sorted_list:
                    wav_path = emt_label_path + '/' + file + '/' + file2 + '/' + file3
                    ### Reading Emotion labels
                    label = file[-1]

                    if (label in emotions_used):
                        infilename = wav_path
                        outfilename = "MSP.csv"
                        opensmile_call = exe_opensmile + " -C " + path_config + " -I " + infilename + " -O " + outfilename + " -l 0"
                        os.system(opensmile_call)

                        MFCC = csv_reader(outfilename)
                        mean = np.mean(MFCC, axis=0)
                        max = np.max(MFCC, axis=0)
                        std = np.std(MFCC, axis=0)
                        sk = skew(MFCC, axis=0)
                        kurt = kurtosis(MFCC, axis=0)

                        Graph.append(np.concatenate([mean, max, std, sk, kurt]))
                        label = emotions_used[label]
                        Graph_label.append(label)
                Data.extend(Graph)
                Label.extend(Graph_label)




# Save Graph data
np.save('data_MSP_open.npy', np.array(Data))
np.save('label_MSP_open.npy', np.array(Label))


# Compare this code to the Ext_feat, what diffrence are they:
"""
The second code example introduces several key differences and enhancements compared to the first code example. Here's a detailed comparison:

1. Emotion Labels and Sessions
First Code:
Emotion labels: {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3, 'exc': 1}
Sessions: ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
Sample rate: 16000
Second Code:
Emotion labels: {'A': 0, 'H': 1, 'N': 2, 'S': 3}
Sessions: ['session1', 'session2', 'session3', 'session4', 'session5', 'session6']
Sample rate: 44100
2. Data Path
First Code: Uses a fixed path "/home/user/Downloads/iemocap/".
Second Code: Uses a variable path "path_to_the_MSP_directory", indicating it might be more generalized for different environments.
3. Feature Extraction and Statistical Analysis
First Code:
Extracts MFCC features using OpenSMILE.
Concatenates MFCC features with spontaneity features.
Splits MFCC features into fixed-length segments.
Second Code:
Extracts MFCC features using OpenSMILE.
Computes statistical features: mean, max, standard deviation, skewness, and kurtosis from the MFCC features.
Constructs a feature vector by concatenating these statistical features.
4. Spontaneity Feature
First Code: Adds a spontaneity feature based on whether the recording is 'improvised' or 'scripted'.
Second Code: Does not include spontaneity features, focusing instead on statistical analysis of MFCC features.
5. File Sorting
First Code: Processes files without additional sorting logic.
Second Code: Implements a specific sorting and grouping mechanism for files, ensuring related files are processed together.
6. Output Files
First Code:
Saves data and labels as ../dataset/IEMOCAP_data.npy and ../dataset/IEMOCAP_label.npy.
Second Code:
Saves data and labels as data_MSP_open.npy and label_MSP_open.npy.
Summary of Key Differences
Dataset and Session Handling:

The second code is configured for a different dataset (MSP) and includes an extra session compared to the first code.
Feature Computation:

The second code computes additional statistical features (mean, max, std, skewness, kurtosis) from the MFCC features, providing a more compact and potentially more informative representation of the audio data.
The first code directly uses raw MFCC features and adds spontaneity features.
Sorting Mechanism:

The second code includes a detailed sorting and grouping mechanism for processing files, ensuring a specific order, which can be crucial for maintaining temporal relationships or consistency in data processing.
Spontaneity Features:

The first code includes spontaneity features based on the type of recording (improvised or scripted), while the second code does not.
Output Files:

The output file names and paths differ, reflecting the different datasets being processed.
"""