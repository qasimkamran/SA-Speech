import whisper
import re
import os


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


if __name__ == '__main__':
    transcribe('audio.wav')