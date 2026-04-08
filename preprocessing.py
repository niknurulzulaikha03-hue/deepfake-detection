import cv2
import numpy as np
import mediapipe as mp
import librosa
from moviepy.editor import VideoFileClip

mp_face_mesh = mp.solutions.face_mesh


def extract_frames(video_path, max_frames=30):

    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < max_frames:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)

    cap.release()

    return frames


def extract_landmarks(frames):

    features = []

    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:

        for frame in frames:

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:

                landmarks = []

                for lm in results.multi_face_landmarks[0].landmark:

                    landmarks.append(lm.x)
                    landmarks.append(lm.y)

                features.append(landmarks)

    return np.array(features)


def extract_audio(video_path):

    try:
        clip = VideoFileClip(video_path)

        audio_path = "temp.wav"

        clip.audio.write_audiofile(audio_path, verbose=False, logger=None)

        y, sr = librosa.load(audio_path)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        return np.mean(mfcc.T, axis=0)

    except:
        return np.zeros(13)


def preprocess_video(video_path):

    frames = extract_frames(video_path)

    if len(frames) == 0:
        return None

    landmark = extract_landmarks(frames)

    if len(landmark) == 0:
        return None

    audio = extract_audio(video_path)

    combined = np.concatenate([
        np.mean(landmark, axis=0),
        audio
    ])

    return combined
