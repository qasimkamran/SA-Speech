import os
import resampy
import librosa
import numpy as np
import pandas as pd
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--data_dir', dest='data_dir', action='store', required=True, type=str)
    parser.add_argument('--mfcc_len', dest='mfcc_len', action='store', type=int)
    parser.add_argument('--duration', dest='duration', action='store', type=float)
    parser.add_argument('--mode', dest='mode', action='store', required=True, type=str)
    return parser.parse_args()


def get_mfcc(directory: str, mfcc_len: int = 13, duration: int = 3):
    signal, fs = librosa.load(directory, res_type='kaiser_fast', duration=duration, sr=22050*2, offset=0.5)
    s_len = len(signal)
    if s_len < duration * fs:
        padding = int(duration * fs) - s_len
        signal = librosa.util.pad_center(signal, padding)
    mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=mfcc_len)
    mfcc = mfcc.T
    return mfcc


def get_mean_mfcc_per_frame(directory: str):
    mfcc = get_mfcc(directory)
    mean_mfcc = np.mean(mfcc, axis=1)


def get_n_wav_files(directory: str, n: int = None):
    wav_files = []
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))
                count += 1
                if count == n:
                    return wav_files
    return wav_files


def main():
    args = parse_arguments()
    wav_files = get_n_wav_files(args.data_dir)
    mfcc_list = []
    if args.mode == 'mean':
        mfcc_list.append()

if __name__ == '__main__':
    main()
