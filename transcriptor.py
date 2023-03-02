import whisper
import re
import os
import wave


model = whisper.load_model("base")


def remove_punctuation(input_string):
    # Define a regular expression pattern to match all punctuation characters
    pattern = r'[^\w\s]'

    # Use the sub function from the re module to replace all matches with an empty string
    output_string = re.sub(pattern, '', input_string)

    return output_string


def calculate_wer(reference, hypothesis):
    ref_raw = remove_punctuation(reference)
    hyp_raw = remove_punctuation(hypothesis)

    ref_words = ref_raw.split()
    hyp_words = hyp_raw.split()

    levenshtein_matrix = [[0] * len(hyp_words) for i in range(len(ref_words))]

    for i in range(len(ref_words)):
        levenshtein_matrix[i][0] = i

    for j in range(len(hyp_words)):
        levenshtein_matrix[0][j] = j

    for i in range(1, len(ref_words)):
        for j in range(1, len(hyp_words)):
            cost = 0
            if ref_words[i - 1] != hyp_words[j - 1]:
                cost = 1
            levenshtein_matrix[i][j] = min(levenshtein_matrix[i - 1][j] + 1, levenshtein_matrix[i][j - 1] + 1, levenshtein_matrix[i - 1][j - 1] + cost)

    return levenshtein_matrix[len(ref_words) - 1][len(hyp_words) - 1]


def transcribe(filename):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(filename)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(result.text)


def clip_audio(wav_obj, startpos, clip_duration, clip_name):
    assert isinstance(wav_obj, wave.Wave_read), f'Not a wave read object'

    # Load audio params
    framerate = wav_obj.getframerate()
    n_frames = wav_obj.getnframes()
    n_channels = wav_obj.getnchannels()
    sample_width = wav_obj.getsampwidth()

    duration = n_frames / float(framerate)

    assert 0 <= startpos <= duration, f'Start position out of bounds'
    wav_obj.setpos(startpos)

    start_frames = startpos * framerate
    clip_frames = clip_duration * framerate
    endpos = start_frames + clip_frames
    if endpos > n_frames:
        clip_frames = (n_frames - start_frames)
        print('Not enough audio left, creating {0} second clip instead'.format(clip_frames/framerate))

    clip_frames_data = wav_obj.readframes(clip_frames)

    with wave.open(clip_name, "wb") as clip_file:
        clip_file.setnchannels(n_channels)
        clip_file.setsampwidth(sample_width)
        clip_file.setframerate(framerate)
        clip_file.writeframes(clip_frames_data)


def full_transcribe(filename):
    with wave.open(filename, "rb") as wav_obj:

        # Load audio params
        framerate = wav_obj.getframerate()
        n_frames = wav_obj.getnframes()
        n_channels = wav_obj.getnchannels()
        sample_width = wav_obj.getsampwidth()

        # Compute number of 30 second n splits
        duration = n_frames / float(framerate)
        n_splits = duration / 30
        if n_splits is not int:
            n_splits = int(n_splits) + 1

        clip_audio(wav_obj, 0, 30, 'test_clip.wav')

        print('Number of n splits:', n_splits)


if __name__ == '__main__':
    full_transcribe('audio.wav')