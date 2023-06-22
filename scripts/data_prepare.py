import os
import argparse
import wave
import csv
import tempfile
import transcriptor
import pandas as pd
import numpy as np
import soundfile as sf


REQUIRED_COLUMNS = ['StartTime(s)', 'EndTime(s)', 'EXCITED', 'CHEERING', 'DISAPPOINTED', 'BOOING', 'ANGRY', 'APPLAUSE', 'SINGING', 'PA']


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--data_dir', dest='datadir', action='store', required=True, type=str)
    parser.add_argument('--output_dir', dest='outputdir', action='store', required=True, type=str)
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
        homogenize_clips(f'{index}.wav', f'{index}.txt', 3) # Make all clips 3 seconds in length
        os.unlink(f'{index}.wav')
        os.unlink(f'{index}.txt')


if __name__ == "__main__":
    main()
