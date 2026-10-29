import os
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\ffmpeg-9.0.1-essentials_build\bin"

import whisper
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.io import wavfile
import numpy as np
import pyttsx3
import threading

_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")
    return _whisper_model

def record_audio(filename="input.wav", duration=5):
    fs = 16000
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, recording)

def transcribe(filename="input.wav"):
    model = get_whisper_model()
    result = model.transcribe(filename)
    return result["text"]

# ---------------- Continuous recording ----------------
_recording_data = []
_is_recording = False
_stream = None

def start_recording():
    global _recording_data, _is_recording, _stream
    _recording_data = []
    _is_recording = True

    def callback(indata, frames, time_info, status):
        if _is_recording:
            _recording_data.append(indata.copy())

    _stream = sd.InputStream(samplerate=16000, channels=1, callback=callback)
    _stream.start()

def stop_recording(filename="input.wav"):
    global _is_recording, _stream
    _is_recording = False
    _stream.stop()
    _stream.close()
    if _recording_data:
        full_recording = np.concatenate(_recording_data, axis=0)
        write(filename, 16000, full_recording)
        return True
    return False

# ---------------- Text-to-speech: SAVE + ACTUALLY PLAY ----------------
EMOTION_VOICE_SETTINGS = {
    "happy":     {"rate": 180, "volume": 1.0},
    "excited":   {"rate": 195, "volume": 1.0},
    "curious":   {"rate": 165, "volume": 0.9},
    "concerned": {"rate": 145, "volume": 0.85},
    "calm":      {"rate": 150, "volume": 0.8},
    "neutral":   {"rate": 165, "volume": 0.9},
}

def speak_and_save(text, filename="ai_voice.wav", emotion="neutral"):
    import pyttsx3
    settings = EMOTION_VOICE_SETTINGS.get(emotion, EMOTION_VOICE_SETTINGS["neutral"])
    engine = pyttsx3.init()
    engine.setProperty('rate', settings["rate"])
    engine.setProperty('volume', settings["volume"])
    engine.save_to_file(text, filename)
    engine.runAndWait()
    engine.stop()
    del engine

def play_audio_file(filename="ai_voice.wav"):
    """Actually plays the saved audio file out loud through your speakers."""
    rate, data = wavfile.read(filename)
    sd.play(data, rate)

def play_audio_async(filename="ai_voice.wav"):
    """Starts playback in the background so animation can run at the same time."""
    thread = threading.Thread(target=play_audio_file, args=(filename,))
    thread.start()
    return thread

def get_audio_duration(filename="ai_voice.wav"):
    rate, data = wavfile.read(filename)
    return len(data) / rate

def get_amplitude_envelope(filename="ai_voice.wav", n_frames=20):
    rate, data = wavfile.read(filename)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float64)
    chunk_size = max(1, len(data) // n_frames)
    envelope = []
    for i in range(n_frames):
        chunk = data[i*chunk_size : (i+1)*chunk_size]
        loudness = np.sqrt(np.mean(chunk**2)) if len(chunk) > 0 else 0
        envelope.append(loudness)
    max_val = max(envelope) if max(envelope) > 0 else 1
    envelope = [v / max_val for v in envelope]
    return envelope