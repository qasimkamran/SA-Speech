import os
import argparse
import wave
import csv
import transcriptor
import pandas as pd


REQUIRED_COLUMNS = ['StartTime(s)', 'EndTime(s)', 'EXCITED', 'CHEERING', 'DISAPPOINTED', 'BOOING', 'ANGRY', 'APPLAUSE', 'SINGING', 'PA']


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--datadir', dest='datadir', action='store', required=True, type=str)
    parser.add_argument('--outputdir', dest='outputdir', action='store', required=True, type=str)
    return parser.parse_args()


def validate_files(file_list):
    if len(file_list) != 2:
        print(f'Data directory must only contain two files. Current count: {len(file_list)}')
        raise ValueError('Invalid file count in directory.')

    wav_files = [file for file in file_list if file.endswith('.wav')]
    csv_files = [file for file in file_list if file.endswith('.csv')]

    if len(wav_files) != 1 or len(csv_files) != 1:
        print('There must only be one CSV and one WAV file in the data directory.')
        raise ValueError('Invalid file types in directory.')
    return wav_files[0], csv_files[0]


def validate_csv(csv_header):
    if all(column in csv_header for column in REQUIRED_COLUMNS):
        print("CSV file contains the required columns.")
    else:
        print("CSV file does not contain all the required columns.")
        raise ValueError('CSV file missing required columns.')


def main():
    args = parse_arguments()
    file_list = os.listdir(args.datadir)
    wav_file, csv_file = validate_files(file_list)
    wav_path = os.path.join(args.datadir, wav_file)
    csv_path = os.path.join(args.datadir, csv_file)

    with open(csv_path, 'r') as file:
        csv_reader = csv.reader(file)
        csv_header = next(csv_reader)
    validate_csv(csv_header)

    df = pd.read_csv(csv_path)
    original_audio = wave.open(wav_path, "rb")

    os.chdir(args.outputdir)
    for index, row in df.iterrows():
        start_time_s = int(row['StartTime(s)'])
        end_time_s = int(row['EndTime(s)'])
        transcriptor.clip_audio(original_audio, start_time_s, (end_time_s - start_time_s), f'{index}.wav')
        labels = ' '.join(str(float(row[column])) for column in REQUIRED_COLUMNS[2:])
        with open(f'{index}.txt', "wb") as label_file:
            label_file.write(labels.encode('utf-8'))
            print(f'Written to {index}.txt')


if __name__ == "__main__":
    main()
