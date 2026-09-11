import librosa
import numpy as np

from preprocessing import preprocess_audio


def extract_features(file_path):

    try:
        y, sr = preprocess_audio(file_path)

        features = []

        # MFCC
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=13
        )

        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))

        # Spectral Centroid
        centroid = librosa.feature.spectral_centroid(
            y=y,
            sr=sr
        )

        features.extend([
            np.mean(centroid),
            np.std(centroid)
        ])

        # Spectral Bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr
        )

        features.extend([
            np.mean(bandwidth),
            np.std(bandwidth)
        ])

        # Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(
            y=y,
            sr=sr
        )

        features.extend([
            np.mean(rolloff),
            np.std(rolloff)
        ])

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)

        features.extend([
            np.mean(zcr),
            np.std(zcr)
        ])

        # RMS Energy
        rms = librosa.feature.rms(y=y)

        features.extend([
            np.mean(rms),
            np.std(rms)
        ])

        # Pitch / F0
        f0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr
        )

        f0_valid = f0[~np.isnan(f0)]

        if len(f0_valid) > 0:

            features.extend([
                np.mean(f0_valid),
                np.std(f0_valid),
                np.min(f0_valid),
                np.max(f0_valid)
            ])

        else:

            features.extend([
                0,
                0,
                0,
                0
            ])

        features = np.array(
            features,
            dtype=np.float32
        )

        if features.shape != (40,):
            raise ValueError(
                f"Expected 40 features, got {features.shape}"
            )

        return features

    except Exception as e:

        print(
            f"Error extracting features from {file_path}: {e}"
        )

        return None


# ============================================================
# CNN MEL SPECTROGRAM
# ============================================================

def extract_mel_spectrogram(
    file_path,
    n_mels=128,
    max_length=256
):

    try:

        # Same preprocessing used during training
        y, sr = preprocess_audio(file_path)

        # Create Mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=n_mels,
            n_fft=1024,
            hop_length=512
        )

        # Convert to decibels
        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        # Make time dimension exactly 256
        if mel_db.shape[1] < max_length:

            pad_width = max_length - mel_db.shape[1]

            mel_db = np.pad(
                mel_db,
                ((0, 0), (0, pad_width)),
                mode="constant",
                constant_values=mel_db.min()
            )

        else:

            mel_db = mel_db[:, :max_length]

        return mel_db.astype(np.float32)

    except Exception as e:

        print(
            f"Error extracting Mel spectrogram from {file_path}: {e}"
        )

        return None


# ============================================================
# CNN MEL NORMALIZATION
# ============================================================

def normalize_mel(mel):

    mel_min = mel.min()
    mel_max = mel.max()

    if mel_max - mel_min == 0:

        return np.zeros_like(mel)

    return (
        (mel - mel_min)
        /
        (mel_max - mel_min)
    )