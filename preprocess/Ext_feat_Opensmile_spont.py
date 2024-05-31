"""
Imports
numpy, os, sys, csv: Standard libraries for numerical operations, file and system operations, and CSV file handling.
torch: PyTorch library for tensor operations.
Function Definition
csv_reader(add): Reads a CSV file and returns the data as a NumPy array, excluding the first two columns.
Data and Configuration
emotions_used: Maps emotions to numerical labels.
emotions_used_comp: Alternative emotion label mapping (not used in the code).
data_path: Path to the dataset.
sessions: List of session directories to process.
framerate: Sample rate for audio files (16000 Hz).
Label, Data: Lists to store processed labels and data.
fix_len: Fixed length for splitting MFCC features (120 frames).
exe_opensmile: Path to the OpenSMILE executable for feature extraction.
path_config: Path to the OpenSMILE configuration file.
Main Processing Loop
The main loop iterates over each session and processes each file containing emotion labels:

Path Setup: Constructs paths for emotion labels and audio files.
Read Emotion Labels: Opens the emotion label file and reads line by line.
Label Filtering: Filters lines with valid emotion labels (not 'xxx' and present in emotions_used).
Feature Extraction:
Constructs the input and output filenames.
Calls OpenSMILE to extract MFCC features and save them to a CSV file.
Read Features: Reads the MFCC features from the CSV file using csv_reader.
Label Assignment: Assigns the numerical label to the emotion.
Spontaneity Feature:
Adds a spontaneity feature based on whether the recording is 'improvised' or 'scripted'.
spont_feat is a tensor indicating spontaneity (one-hot encoded).
Feature Concatenation: Concatenates the MFCC features and spontaneity features.
Split Features: Splits the MFCC features into segments of length fix_len and appends to Data and Label.
"""

import numpy as np
import os
import sys
import csv
import torch

def csv_reader(add):
    with open(outfilename, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        headers = next(reader)
        data = np.array(list(reader)).astype(float)
        
    return data[:,2:] 


#### Load data
emotions_used = { 'ang':0, 'hap':1, 'neu':2, 'sad':3 , 'exc':1}
emotions_used_comp = {'Neutral;':2, 'Anger;':0, 'Sadness;':3, 'Happiness;':1}
data_path = "/home/user/Downloads/iemocap/"
sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
framerate = 16000

Label = []
Data = []
fix_len = 120

# openSMILE (Need to be changed)
exe_opensmile = "./home/user/opensmile-2.3.0/SMILExtract"
path_config   = "/demo2.conf"


for ses in sessions:
    emt_label_path = data_path + ses + '/dialog/EmoEvaluation/'
    for file in os.listdir(emt_label_path):
        if file.startswith('Ses'):
            wav_path = data_path + ses + '/sentences/wav/' + file.split('.')[0] + '/'
            ### Reading Emotion labels
            with open(emt_label_path + file, 'r') as f:
                for line in f:
                    if 'Ses' in line:
                        Imp_name = line.split('\t')[1]
                        label = line.split('\t')[2]

                        if not(label.startswith('xxx')) and (label in emotions_used):
                            infilename = wav_path + Imp_name + '.wav'
                            outfilename = "IEMOCAP.csv"
                            opensmile_call = exe_opensmile + " -C " + path_config + " -I " + infilename + " -O " + outfilename
                            os.system(opensmile_call)

                            MFCC = csv_reader(outfilename)
                            label = emotions_used[label]

                            if 'impro' in line:
                                spont_feat = torch.Tensor([1, 0]).view(1, 2).repeat(MFCC.shape[0], 1).detach().cpu().numpy()
                            elif 'script' in line:
                                spont_feat = torch.Tensor([0, 1]).view(1, 2).repeat(MFCC.shape[0], 1).detach().cpu().numpy()

                            MFCC = np.concatenate([MFCC, spont_feat], axis=1)
                            # Splitting MFCC to equal sizes
                            for i in range(MFCC.shape[0] // fix_len):
                                Data.append(MFCC[i * fix_len:(i + 1) * fix_len, :])
                                Label.append(label)



# Save Graph data
np.save('../dataset/IEMOCAP_data.npy', np.array(Data))
np.save('../dataset/IEMOCAP_label.npy', np.array(Label))
