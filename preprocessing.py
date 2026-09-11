import librosa


TARGET_SR = 16000


def preprocess_audio(file_path):
    """
    Load and preprocess an audio file.

    Steps:
    1. Load audio at 16 kHz
    2. Convert to mono
    3. Remove leading/trailing silence
    """

    y, sr = librosa.load(
        file_path,
        sr=TARGET_SR,
        mono=True
    )

    # Remove leading/trailing silence
    y, _ = librosa.effects.trim(y, top_db=30)

    if len(y) == 0:
        raise ValueError("Audio contains no usable signal.")

    return y, sr