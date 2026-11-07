import wave

import numpy as np

from rd_audio.audio_utils import chunk_level, iter_chunks, load_wav_mono, prepend_silence
from rd_audio.tts_engines import FakeTTS


def test_fake_tts_writes_speech_length_audio(tmp_path):
    p = tmp_path / "a.wav"
    assert FakeTTS().synthesize("hello there robot disco", str(p))
    audio, rate = load_wav_mono(str(p))
    assert rate == 16000 and abs(audio.size / rate - 23 * 0.06) < 0.05
    assert np.abs(audio).max() > 0.3
    with wave.open(str(p)) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2


def test_chunks_and_levels():
    rate = 16000
    audio = np.sin(2 * np.pi * 220 * np.arange(rate) / rate).astype(np.float32)
    chunks = list(iter_chunks(audio, rate, 0.04))
    assert len(chunks) == 25 and abs(chunks[-1][1] - 1.0) < 1e-6
    data, level = chunk_level(chunks[0][0], 1.0)
    assert 0.5 < level <= 1.0 and data.size > 0
    padded = prepend_silence(audio, rate, 0.5, level=0.0)
    assert padded.size == audio.size + rate // 2 and np.abs(padded[:rate // 2]).max() == 0.0
