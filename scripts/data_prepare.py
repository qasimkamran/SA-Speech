import os
import argparse
import wave
import shutil
import random
import transcriptor
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Argument Parser')

parser.add_argument('--filepath', dest='filepath', action='store', required=False, type=str)

args = parser.parse_args()

file_list = os.listdir(args.filepath)

if file_list != 2:
    print('Data directory must only contain two files')
    raise Exception

wav_files = [file for file in file_list if file.endswith('.wav')]
csv_files = [file for file in file_list if file.endswith('.csv')]

if len(wav_files) != 1 and len(csv_files) != 1:
    print('There must only be one CSV and only one WAV file in the data directory')
    raise Exception

required_columns = ['StartTime(s)', 'EndTime(s)', 'EXCITED', 'CHEERING', 'DISAPPOINTED', 'BOOING', 'ANGRY', 'APPLAUSE', 'SINGING', 'PA']

with open(csv_files, 'r') as file:
    csv_reader = csv.reader(file)
    csv_header = next(reader) 

if all(column in csv_header for column in required_columns):
    print("CSV file contains the required columns.")
else:
    print("CSV file does not contain all the required columns.")
    raise Exception

df = pd.read_csv(csv_files)
original_audio = wave.open(wav_files, "rb")
original_sample_rate = original_audio.getframerate()

for index, row in df.iterrows():
  start_time_s = int(row['StartTime(s)'])
  end_time_s = int(row['EndTime(s)'])
  transcriptor.clip_audio(original_audio, start_time_s, (end_time_s - start_time_s), '{0}.wav'.format(index))
  with open('SOUNDFIELD_C_{0}.txt'.format(index), "wb") as label_file:
    line = str(float(row['EXCITED'])) + ' ' + str(float(row['CHEERING'])) + ' ' + \
           str(float(row['DISAPPOINTED'])) + ' ' + str(float(row['BOOING'])) + ' ' + \
           str(float(row['ANGRY'])) + ' ' + str(float(row['APPLAUSE'])) + ' ' + \
           str(float(row['SINGING'])) + ' ' + str(float(row['PA']))
    encoded_line = line.encode('utf-8')
    label_file.write(encoded_line)
    print('Written {0}.txt'.format(index))
