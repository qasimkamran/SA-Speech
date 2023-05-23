import os
import librosa


def get_feature(file_path: str, mfcc_len: int = 39, mean_signal_length: int = 100000):
  """
  file_path: Speech signal folder
  mfcc_len: MFCC coefficient length
  mean_signal_length: MFCC feature average length
  """
  signal, fs = librosa.load(file_path)
  s_len = len(signal)

  if s_len < mean_signal_length:
    pad_len = mean_signal_length - s_len
    pad_rem = pad_len % 2
    pad_len //= 2
    signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values = 0)
  else:
    pad_len = s_len - mean_signal_length
    pad_len //= 2
    signal = signal[pad_len:pad_len + mean_signal_length]
  mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=39)
  mfcc = mfcc.T
  feature = mfcc
  return feature


if __name__ == '__main__':