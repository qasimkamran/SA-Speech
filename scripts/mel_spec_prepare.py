import os
import argparse
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm


N_MELS = 128
HOP_LENGTH = 512


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--input_dir', dest='input_dir', action='store', required=True, type=str)
    parser.add_argument('--output_dir', dest='output_dir', action='store', required=True, type=str)
    return parser.parse_args()


def process_data(input_dir, output_dir):
    filenames = [filename for filename in os.listdir(input_dir) if filename.endswith('.wav')]

    progress_bar = tqdm(filenames, desc='Writing Mel Spectrograms', unit='filename')

    for filename in progress_bar:
        filenamepath = os.path.join(input_dir, filename)
        output_filenamepath = os.path.join(output_dir, os.path.splitext(filename)[0] + '.png')

        audio_file, sample_rate = librosa.load(filenamepath)
        mel_spec = librosa.feature.melspectrogram(y=audio_file, sr=sample_rate, n_mels=N_MELS, hop_length=HOP_LENGTH)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spec_db, sr=sample_rate, x_axis='time', y_axis='mel', cmap='viridis')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_filenamepath, bbox_inches='tight', pad_inches=0)  # Save the plot without extra white space
        plt.close()

    progress_bar.close()


def main():
    args = parse_arguments()
    process_data(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()