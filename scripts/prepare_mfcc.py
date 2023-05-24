import os
import librosa


def get_mfcc_features(file_path: str, mfcc_len: int = 13, duration: int = 2.5):
  signal, fs = librosa.load(file_path, res_type='kaiser_fast', duration=duration, sr=22050*2, offset=0.5)
  s_len = len(signal)
  if s_len < duration * fs:
    padding = int(duration * fs) - s_len
    signal = librosa.util.pad_center(signal, padding)
  mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=mfcc_len)
  mfcc = mfcc.T
  features = mfcc
  return features


def get_n_wav_files(directory, n):
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


if __name__ == '__main__':
