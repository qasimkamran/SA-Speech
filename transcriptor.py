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
            levenshtein_matrix[i][j] = min(levenshtein_matrix[i - 1][j] + 1, levenshtein_matrix[i][j - 1] + 1,
                                           levenshtein_matrix[i - 1][j - 1] + cost)

    return levenshtein_matrix[len(ref_words) - 1][len(hyp_words) - 1]


def transcribe(audio_filename, transcription_filename):
    # load audio and pad/trim it to fit 30 seconds
    try:
        audio = whisper.load_audio(audio_filename)
    except RuntimeError as e:
        print('Error in loading ', audio_filename)
        return

    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    # print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(result.text)

    with open(transcription_filename, 'a') as transcription:
        transcription.write(result.text)


def clip_audio(wav_obj, startpos, clip_duration, clip_name):
    assert isinstance(wav_obj, wave.Wave_read), f'Not a wave read object'

    # Load audio params
    framerate = wav_obj.getframerate()
    n_frames = wav_obj.getnframes()
    n_channels = wav_obj.getnchannels()
    sample_width = wav_obj.getsampwidth()

    duration = n_frames / float(framerate)

    assert 0 <= startpos <= duration, f'Start position out of bounds'

    start_frames = startpos * framerate
    clip_frames = clip_duration * framerate
    endpos = start_frames + clip_frames
    if endpos > n_frames:
        clip_frames = (n_frames - start_frames)
        print('Not enough audio left, creating {0}s clip instead'.format(clip_frames / framerate))

    # Read clip frames from player head position
    wav_obj.setpos(start_frames)
    clip_frames_data = wav_obj.readframes(clip_frames)

    # Write clipped audio to a new file
    with wave.open(clip_name, "wb") as clip_file:
        clip_file.setnchannels(n_channels)
        clip_file.setsampwidth(sample_width)
        clip_file.setframerate(framerate)
        clip_file.writeframes(clip_frames_data)
        print('Written to {0}'.format(clip_name))


def clean_clips(clip_names):
    assert clip_names, f'Empty clip names list'
    for clip in clip_names:
        os.remove(clip)
        print('Removed {0}'.format(clip))


def full_transcribe(audio_filename, transcription_filename):
    with wave.open(audio_filename, "rb") as wav_obj:

        # Load audio params
        framerate = wav_obj.getframerate()
        n_frames = wav_obj.getnframes()

        # Compute number of 30 second n splits
        duration = n_frames / float(framerate)
        n_splits = duration / 30
        if n_splits is not int:
            n_splits = int(n_splits) + 1

        clip_names = []
        startpos = 0
        remainder = duration
        for i in range(n_splits):
            clip_name = '{0}_clip_{1}.wav'.format(os.path.splitext(audio_filename)[0], i)
            clip_audio(wav_obj, startpos, 30, clip_name)
            clip_names.append(clip_name)
            remainder = duration - startpos
            if remainder < 30:
                break
            startpos += 30

        for clip in clip_names:
            transcribe(clip, transcription_filename)

        clean_clips(clip_names)


if __name__ == '__main__':
    '''
    # Clip large input audio
    with wave.open('audio/commentary.wav', "rb") as wav_obj:
        clip_audio(wav_obj, 0, 600, 'audio/commentary_short.wav')
    '''

    full_transcribe('audio/commentary_short.wav', 'transcription/commentary_short.txt')
