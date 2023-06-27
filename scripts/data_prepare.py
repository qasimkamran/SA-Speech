import os
import argparse
import wave
import csv
import shutil
import tempfile
import transcriptor
import pandas as pd
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = ['StartTime(s)', 'EndTime(s)', 'Excited', 'Cheering', 'Disappointed', 'Booing', 'Angry', 'Applause', 'Singing', 'PA']
THRESHOLD = 0.5


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--input_dir', dest='input_dir', action='store', required=True, type=str)
    parser.add_argument('--output_dir', dest='output_dir', action='store', required=True, type=str)
    parser.add_argument('--already_transformed', action='store_true', required=False, default=False)
    return parser.parse_args()


def get_filename_pairs(data_dir):
    filenames = os.listdir(data_dir)
    filename_pairs = {}

    csv_filenames = [f for f in filenames if f.endswith('.csv')]
    wav_filenames = [f for f in filenames if f.endswith('.wav')]

    # Assume that for every csv file there is a corresponding wav file with the same name
    for csv_filename in csv_filenames:
        csv_filenamepath = os.path.join(data_dir, csv_filename)
        with open(csv_filenamepath, 'r') as file:
            csv_reader = csv.reader(file)
            csv_header = next(csv_reader)
        validate_csv(csv_header)
        
        base_name = os.path.splitext(csv_filename)[0]
        corresponding_wav = base_name + '.wav'

        if corresponding_wav in wav_filenames:
            filename_pairs[csv_filename] = corresponding_wav

    return filename_pairs


def validate_csv(csv_header):
    case_agnostic_required_columns = [column.lower() for column in REQUIRED_COLUMNS]
    case_agnostic_csv_header = [column.lower() for column in csv_header]
    if not all(column in case_agnostic_csv_header for column in case_agnostic_required_columns):
        raise ValueError('CSV file missing required columns.')


def pad_audio(audio, target_duration, sr):
    # TODO change wave object parameter here to generic audio file representation and modify function to match
    tmpfile = tempfile.NamedTemporaryFile(delete=False)
    tmpfile.close()

    with wave.open(tmpfile.name, 'wb') as wavfile:
        wavfile.setparams(audio.getparams())
        wavfile.writeframes(audio.readframes(audio.getnframes()))

    audio_array, sr = sf.read(tmpfile.name)
    audio_duration = len(audio_array) / sr

    if audio_duration < target_duration:
        total_samples_required = int(target_duration * sr)
        audio_full_repeats = np.tile(audio_array, int(total_samples_required / len(audio_array)))
        remaining_samples = total_samples_required - len(audio_full_repeats)
        audio_partial_repeat = audio_array[:remaining_samples]
        padded_audio_array = np.concatenate([audio_full_repeats, audio_partial_repeat])

        os.unlink(tmpfile.name)
        return padded_audio_array.tobytes()
    else:
        os.unlink(tmpfile.name)
        return audio.readframes(audio.getnframes())


def open_audio_file(audio_dir):
    # TODO make function return generic audio file representation
    audio_file = wave.open(audio_dir, "rb")
    audio_name, _ = os.path.splitext(audio_dir)
    audio_params = (audio_file.getnchannels(),
                    audio_file.getsampwidth(),
                    audio_file.getframerate(),
                    audio_name)
    n_frames = audio_file.getnframes()
    audio_duration = n_frames / float(audio_params[2])
    return audio_file, audio_params, audio_duration


def create_clips(audio_file, audio_params, audio_duration, clip_duration):
    clip_names = []
    n_splits = int(audio_duration / clip_duration) + 1
    audio_name = audio_params[-1]

    for i in range(n_splits):
        startpos = i * clip_duration
        remainder = audio_duration - startpos
        clip_name = '{0}_{1}.wav'.format(audio_name, i)

        if remainder == 0:
            break

        if remainder < clip_duration:
            process_last_clip(audio_file, audio_params, startpos, clip_duration, clip_name)
            clip_names.append(clip_name)
            break

        transcriptor.clip_audio(audio_file, startpos, clip_duration, clip_name)
        clip_names.append(clip_name)

    return clip_names


def process_last_clip(audio_file, audio_params, startpos, clip_duration, clip_name):
    transcriptor.clip_audio(audio_file, startpos, clip_duration, clip_name)
    with wave.open(clip_name, "rb") as audio_clip:
        audio_bytes = pad_audio(audio_clip, clip_duration , audio_params[2])
    os.unlink(clip_name)
    audio_array = np.frombuffer(audio_bytes, dtype=np.float64)  # Assuming 24-bit PCM audio
    sf.write(clip_name, audio_array, audio_params[2], subtype='PCM_24') # TODO use original audio bitrate without specifying


def copy_text_file(source_filenamepath, target_filenamepath):
    with open(source_filenamepath, 'r') as source_file:
        with open(target_filenamepath, 'w') as target_file:
            content = source_file.read()
            target_file.write(content)


def homogenize_clips(audio_filenamepath, label_filenamepath, clip_duration):
    audio_file, audio_params, duration = open_audio_file(audio_filenamepath)
    clip_names = create_clips(audio_file, audio_params, duration, clip_duration)
    label_name = audio_params[-1]
    for i in range(len(clip_names)):
        copy_text_file(label_filenamepath, f'{label_name}_{i}.txt')
    audio_file.close()
    return clip_names


def relocate_wav_and_txt(base_dir, txt_dir, wav_dir):
    filenames = os.listdir(base_dir)
    for filename in filenames:
        filenamepath = os.path.join(base_dir, filename)

        if filename.endswith('.txt'):
            shutil.move(filenamepath, os.path.join(txt_dir, filename))
            print(f"Moved {filename} to {txt_dir}")

        if filename.endswith('.wav'):
            shutil.move(filenamepath, os.path.join(wav_dir, filename))
            print(f"Moved {filename} to {wav_dir}")


def plot_frequency_distribution(label_dir, stats_dir):
    label_filenames = [filename for filename in os.listdir(label_dir) if filename.endswith('.txt')]

    categories = REQUIRED_COLUMNS[2:]
    frequencies = [0] * len(categories)

    for filename in label_filenames:
        filenamepath = os.path.join(label_dir, filename)

        with open(filenamepath, 'r') as file:
            line = file.readline().strip()
            values = [float(val) for val in line.split()]
            present_categories = [0] *  len(values)
            present_categories = [1 if value > THRESHOLD else 0 for value in values]

        # Update the frequency count for each emotion
        frequencies = [frequencies[i] + present_categories[i] for i in range(len(frequencies))]

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(categories, frequencies)
    plt.title('Thresholded Frequency Distribution')
    plt.xlabel('Category')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_file = os.path.join(stats_dir, 'frequency_dstribution.png')
    plt.savefig(output_file)
    print(f"Figure saved to {output_file}")
    plt.close()


def process_data(filename_pairs, args, base_dir):
    count = 0
    for csv_filename, wav_filename in filename_pairs.items():
        wav_filenamepath = os.path.join(args.input_dir, wav_filename)
        csv_filenamepath = os.path.join(args.input_dir, csv_filename)

        df = pd.read_csv(csv_filenamepath)
        df.columns = df.columns.str.lower()
        original_audio = wave.open(wav_filenamepath, "rb")
        
        os.chdir(args.output_dir)
        for index, row in df.iterrows():
            start_time_s = int(row['starttime(s)'])
            end_time_s = int(row['endtime(s)'])
            output_wav_filenamepath = f'{count}_{index}.wav'
            output_txt_filenamepath = f'{count}_{index}.txt'
            transcriptor.clip_audio(original_audio, start_time_s, (end_time_s - start_time_s), output_wav_filenamepath)
            case_agnostic_required_columns = [column.lower() for column in REQUIRED_COLUMNS]
            labels = ' '.join(str(float(row[column])) for column in case_agnostic_required_columns[2:])
            with open(output_txt_filenamepath, "wb") as label_file:
                label_file.write(labels.encode('utf-8'))
                print(f'Written to {output_txt_filenamepath}')
            homogenize_clips(output_wav_filenamepath, output_txt_filenamepath, 3) # Make all clips 3 seconds in length
            os.unlink(output_wav_filenamepath)
            os.unlink(output_txt_filenamepath)
        os.chdir(base_dir)
        count += 1


def main():
    args = parse_arguments()
    filename_pairs = get_filename_pairs(args.input_dir)
    base_dir = os.getcwd()

    raw_dir = os.path.join(args.output_dir, 'raw')
    label_dir = os.path.join(args.output_dir, 'labels')
    stats_dir = os.path.join(args.output_dir, 'stats')

    if not os.path.exists(raw_dir):
        os.mkdir(raw_dir)

    if not os.path.exists(label_dir):
        os.mkdir(label_dir)

    if not os.path.exists(stats_dir):
        os.mkdir(stats_dir)

    if not args.already_transformed:
        process_data(filename_pairs, args, base_dir)
        # TODO extra step takes much more time and is unoptimal, refactor to just write at different directories initially
        relocate_wav_and_txt(args.output_dir, label_dir, raw_dir)
        
    plot_frequency_distribution(label_dir, stats_dir)


if __name__ == "__main__":
    main()
