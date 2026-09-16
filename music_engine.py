"""
music_engine.py
Core music generation and remix engine using pure Python / NumPy / SciPy.
No external AI API keys required — everything runs locally.
"""

import numpy as np
import io
import wave
import struct
import random
from scipy.io import wavfile
from scipy.signal import butter, sosfilt


# ─────────────────────────────────────────────
# MUSIC THEORY CONSTANTS
# ─────────────────────────────────────────────

SAMPLE_RATE = 44100

SCALES = {
    "Major":       [0, 2, 4, 5, 7, 9, 11],
    "Minor":       [0, 2, 3, 5, 7, 8, 10],
    "Pentatonic":  [0, 2, 4, 7, 9],
    "Blues":       [0, 3, 5, 6, 7, 10],
    "Dorian":      [0, 2, 3, 5, 7, 9, 10],
    "Mixolydian":  [0, 2, 4, 5, 7, 9, 10],
    "Phrygian":    [0, 1, 3, 5, 7, 8, 10],
}

MOOD_PROFILES = {
    "Happy": {
        "scale": "Major",
        "root": "C",
        "tempo": 128,
        "octave": 5,
        "waveform": "sine",
        "reverb": 0.15,
        "description": "Bright, uplifting major-key vibes with a punchy beat.",
    },
    "Sad": {
        "scale": "Minor",
        "root": "A",
        "tempo": 72,
        "octave": 4,
        "waveform": "sine",
        "reverb": 0.45,
        "description": "Melancholic minor-key melody with soft, slow rhythm.",
    },
    "Energetic": {
        "scale": "Pentatonic",
        "root": "E",
        "tempo": 160,
        "octave": 5,
        "waveform": "sawtooth",
        "reverb": 0.10,
        "description": "Fast, aggressive riff using sawtooth waves.",
    },
    "Calm": {
        "scale": "Major",
        "root": "G",
        "tempo": 60,
        "octave": 4,
        "waveform": "sine",
        "reverb": 0.55,
        "description": "Gentle, slow-paced ambient tone with heavy reverb.",
    },
    "Mysterious": {
        "scale": "Phrygian",
        "root": "D",
        "tempo": 90,
        "octave": 4,
        "waveform": "triangle",
        "reverb": 0.35,
        "description": "Dark Phrygian mode with eerie triangle waves.",
    },
    "Romantic": {
        "scale": "Major",
        "root": "F",
        "tempo": 84,
        "octave": 5,
        "waveform": "sine",
        "reverb": 0.40,
        "description": "Warm, flowing melody in F Major with soft dynamics.",
    },
    "Angry": {
        "scale": "Blues",
        "root": "E",
        "tempo": 150,
        "octave": 4,
        "waveform": "square",
        "reverb": 0.08,
        "description": "Raw blues scale driven by harsh square-wave distortion.",
    },
    "Dreamy": {
        "scale": "Dorian",
        "root": "B",
        "tempo": 70,
        "octave": 5,
        "waveform": "triangle",
        "reverb": 0.65,
        "description": "Floating Dorian mode with lush reverb for a dreamy feel.",
    },
}

GENRE_PROFILES = {
    "Pop": {
        "chord_prog": [0, 5, 3, 4],
        "rhythm_pattern": [1, 0, 1, 0, 1, 0, 1, 0],
        "arp_speed": 2,
    },
    "Hip-Hop": {
        "chord_prog": [0, 0, 3, 5],
        "rhythm_pattern": [1, 0, 0, 1, 0, 1, 0, 0],
        "arp_speed": 1,
    },
    "Rock": {
        "chord_prog": [0, 3, 4, 3],
        "rhythm_pattern": [1, 0, 1, 1, 0, 1, 1, 0],
        "arp_speed": 2,
    },
    "Jazz": {
        "chord_prog": [0, 2, 4, 5],
        "rhythm_pattern": [1, 0, 0, 1, 1, 0, 0, 1],
        "arp_speed": 3,
    },
    "Electronic": {
        "chord_prog": [0, 5, 3, 6],
        "rhythm_pattern": [1, 1, 0, 1, 1, 0, 1, 0],
        "arp_speed": 4,
    },
    "Classical": {
        "chord_prog": [0, 4, 5, 3],
        "rhythm_pattern": [1, 0, 0, 0, 1, 0, 0, 0],
        "arp_speed": 1,
    },
    "Reggae": {
        "chord_prog": [0, 3, 4, 3],
        "rhythm_pattern": [0, 1, 0, 1, 0, 1, 0, 1],
        "arp_speed": 1,
    },
    "R&B": {
        "chord_prog": [0, 2, 5, 4],
        "rhythm_pattern": [1, 0, 0, 1, 0, 0, 1, 0],
        "arp_speed": 2,
    },
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ─────────────────────────────────────────────
# CORE SYNTHESIS
# ─────────────────────────────────────────────

def note_to_freq(note_name: str, octave: int) -> float:
    """Convert note name + octave to frequency (Hz), A4 = 440 Hz."""
    idx = NOTE_NAMES.index(note_name)
    midi = (octave + 1) * 12 + idx
    return 440.0 * (2 ** ((midi - 69) / 12))


def generate_waveform(freq: float, duration: float, waveform: str = "sine",
                      amplitude: float = 0.4) -> np.ndarray:
    """Generate a single note waveform."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    if waveform == "sine":
        wave = amplitude * np.sin(2 * np.pi * freq * t)
    elif waveform == "square":
        wave = amplitude * np.sign(np.sin(2 * np.pi * freq * t))
    elif waveform == "sawtooth":
        wave = amplitude * (2 * (t * freq - np.floor(0.5 + t * freq)))
    elif waveform == "triangle":
        wave = amplitude * (2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1)
    else:
        wave = amplitude * np.sin(2 * np.pi * freq * t)

    # ADSR envelope
    attack  = int(0.02 * SAMPLE_RATE)
    decay   = int(0.05 * SAMPLE_RATE)
    sustain_level = 0.7
    release = int(0.08 * SAMPLE_RATE)
    n = len(wave)
    env = np.ones(n)
    env[:attack] = np.linspace(0, 1, attack)
    env[attack:attack + decay] = np.linspace(1, sustain_level, decay)
    env[max(0, n - release):] = np.linspace(sustain_level, 0, min(release, n))
    return (wave * env).astype(np.float32)


def apply_reverb(audio: np.ndarray, amount: float = 0.3) -> np.ndarray:
    """Simple comb-filter reverb."""
    delay_samples = int(0.03 * SAMPLE_RATE)
    reverb_audio = audio.copy()
    for i in range(1, 6):
        delay = delay_samples * i
        decay  = amount * (0.6 ** i)
        if delay < len(audio):
            reverb_audio[delay:] += decay * audio[:len(audio) - delay]
    return np.clip(reverb_audio, -1.0, 1.0)


def apply_lowpass(audio: np.ndarray, cutoff: float = 4000.0) -> np.ndarray:
    """Butterworth low-pass filter."""
    sos = butter(4, cutoff / (SAMPLE_RATE / 2), btype="low", output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def apply_chorus(audio: np.ndarray, depth: float = 0.003) -> np.ndarray:
    """Chorus / vibrato effect using a small pitch-shift sweep."""
    lfo_rate = 2.0
    t = np.arange(len(audio)) / SAMPLE_RATE
    delay_mod = (depth * SAMPLE_RATE * np.sin(2 * np.pi * lfo_rate * t)).astype(int)
    out = audio.copy()
    indices = np.arange(len(audio)) - np.abs(delay_mod)
    valid = indices >= 0
    out[valid] = (audio[valid] + 0.5 * audio[indices[valid]]) / 1.5
    return out.astype(np.float32)


# ─────────────────────────────────────────────
# SCALE / CHORD UTILITIES
# ─────────────────────────────────────────────

def build_scale_freqs(root: str, scale_name: str, octave: int, n_notes: int = 16):
    """Return a list of frequencies for `n_notes` steps of the scale."""
    intervals = SCALES[scale_name]
    root_idx  = NOTE_NAMES.index(root)
    freqs = []
    oct_offset = 0
    for i in range(n_notes):
        step_idx = i % len(intervals)
        if i > 0 and step_idx == 0:
            oct_offset += 1
        semitone = root_idx + intervals[step_idx]
        note_oct = octave + oct_offset + semitone // 12
        note_name = NOTE_NAMES[semitone % 12]
        freqs.append(note_to_freq(note_name, note_oct))
    return freqs


def build_chord(root_freq: float, chord_type: str = "major") -> list:
    """Return [root, third, fifth] frequencies for a chord."""
    if chord_type == "major":
        return [root_freq, root_freq * (2 ** (4 / 12)), root_freq * (2 ** (7 / 12))]
    elif chord_type == "minor":
        return [root_freq, root_freq * (2 ** (3 / 12)), root_freq * (2 ** (7 / 12))]
    elif chord_type == "dominant7":
        return [root_freq, root_freq * (2 ** (4/12)), root_freq * (2 ** (7/12)), root_freq * (2 ** (10/12))]
    return [root_freq]


# ─────────────────────────────────────────────
# BEAT / DRUM SYNTHESIS
# ─────────────────────────────────────────────

def make_kick(duration: float = 0.3) -> np.ndarray:
    """Synthesised kick drum."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
    freq_env = 120 * np.exp(-30 * t)
    click    = 0.5 * np.exp(-80 * t) * np.sin(2 * np.pi * 800 * t)
    body     = 0.8 * np.exp(-20 * t) * np.sin(2 * np.pi * freq_env * t)
    return np.clip(click + body, -1, 1).astype(np.float32)


def make_snare(duration: float = 0.2) -> np.ndarray:
    """Synthesised snare drum."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
    noise = np.random.randn(len(t)) * np.exp(-20 * t)
    tone  = 0.3 * np.exp(-40 * t) * np.sin(2 * np.pi * 200 * t)
    return np.clip(0.6 * noise + tone, -1, 1).astype(np.float32)


def make_hihat(duration: float = 0.05) -> np.ndarray:
    """Synthesised hi-hat."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
    noise = np.random.randn(len(t)) * np.exp(-80 * t)
    return apply_lowpass(np.clip(noise * 0.4, -1, 1), cutoff=8000.0)


def make_beat_track(tempo: int, bars: int, pattern: list) -> np.ndarray:
    """Create a drum beat track."""
    beat_dur   = 60.0 / tempo          # seconds per beat (quarter note)
    step_dur   = beat_dur / 2          # 8th-note grid
    total_steps = bars * 8
    total_samples = int(SAMPLE_RATE * step_dur * total_steps)
    track = np.zeros(total_samples, dtype=np.float32)

    kick_wave  = make_kick()
    snare_wave = make_snare()
    hihat_wave = make_hihat()

    pattern_len = len(pattern)
    for step in range(total_steps):
        start = int(step * step_dur * SAMPLE_RATE)
        pat_pos = step % pattern_len

        # kick on beat 1 and 5 (positions 0 and 4 in 8-step)
        if step % 8 in (0, 4):
            end = start + len(kick_wave)
            if end <= total_samples:
                track[start:end] += kick_wave * 0.9

        # snare on beat 3 and 7
        if step % 8 in (2, 6):
            end = start + len(snare_wave)
            if end <= total_samples:
                track[start:end] += snare_wave * 0.7

        # hi-hat on pattern positions
        if pattern[pat_pos]:
            end = start + len(hihat_wave)
            if end <= total_samples:
                track[start:end] += hihat_wave * 0.5

    return np.clip(track, -1, 1)


# ─────────────────────────────────────────────
# MELODY GENERATION
# ─────────────────────────────────────────────

def generate_melody(scale_freqs: list, tempo: int, bars: int,
                    waveform: str, rhythm_pattern: list) -> np.ndarray:
    """Generate an arpeggiated / melodic line."""
    beat_dur  = 60.0 / tempo
    step_dur  = beat_dur / 2
    total_steps = bars * 8
    total_samples = int(SAMPLE_RATE * step_dur * total_steps)
    melody = np.zeros(total_samples, dtype=np.float32)

    pattern_len = len(rhythm_pattern)
    for step in range(total_steps):
        if rhythm_pattern[step % pattern_len]:
            freq  = scale_freqs[step % len(scale_freqs)]
            note  = generate_waveform(freq, step_dur * 0.9, waveform, amplitude=0.35)
            start = int(step * step_dur * SAMPLE_RATE)
            end   = start + len(note)
            if end <= total_samples:
                melody[start:end] += note

    return melody


def generate_chord_track(scale_freqs: list, chord_prog: list, tempo: int,
                         bars: int, waveform: str) -> np.ndarray:
    """Generate a chord pad / rhythm-guitar track."""
    beat_dur      = 60.0 / tempo
    chord_dur     = beat_dur * 4           # one chord per bar
    total_samples = int(SAMPLE_RATE * beat_dur * 4 * bars)
    track = np.zeros(total_samples, dtype=np.float32)

    prog_len = len(chord_prog)
    for bar in range(bars):
        chord_root_idx = chord_prog[bar % prog_len]
        root_freq      = scale_freqs[chord_root_idx % len(scale_freqs)] / 2  # one octave down
        chord_freqs    = build_chord(root_freq, "minor" if random.random() < 0.3 else "major")
        start          = int(bar * chord_dur * SAMPLE_RATE)
        for f in chord_freqs:
            note = generate_waveform(f, chord_dur * 0.95, waveform, amplitude=0.18)
            end  = start + len(note)
            if end <= total_samples:
                track[start:end] += note

    return np.clip(track, -1, 1)


def generate_bassline(scale_freqs: list, chord_prog: list,
                      tempo: int, bars: int) -> np.ndarray:
    """Low bass line following chord roots."""
    beat_dur      = 60.0 / tempo
    step_dur      = beat_dur
    total_samples = int(SAMPLE_RATE * step_dur * bars * 4)
    bass = np.zeros(total_samples, dtype=np.float32)

    prog_len = len(chord_prog)
    for beat in range(bars * 4):
        bar      = beat // 4
        root_idx = chord_prog[bar % prog_len]
        freq     = scale_freqs[root_idx % len(scale_freqs)] / 4   # two octaves down
        note     = generate_waveform(freq, step_dur * 0.85, "sine", amplitude=0.45)
        start    = int(beat * step_dur * SAMPLE_RATE)
        end      = start + len(note)
        if end <= total_samples:
            bass[start:end] += note

    return np.clip(bass, -1, 1)


# ─────────────────────────────────────────────
# MAIN GENERATION API
# ─────────────────────────────────────────────

def generate_music(mood: str, genre: str, duration_bars: int = 8,
                   custom_tempo: int = None, custom_scale: str = None,
                   remix_intensity: float = 0.5) -> bytes:
    """
    Generate a full music clip for the given mood + genre.
    Returns raw WAV bytes.
    """
    mood_p  = MOOD_PROFILES[mood]
    genre_p = GENRE_PROFILES[genre]

    tempo     = custom_tempo if custom_tempo else mood_p["tempo"]
    scale     = custom_scale if custom_scale else mood_p["scale"]
    root      = mood_p["root"]
    octave    = mood_p["octave"]
    waveform  = mood_p["waveform"]
    reverb    = mood_p["reverb"]

    scale_freqs = build_scale_freqs(root, scale, octave, n_notes=24)

    # Shuffle scale order slightly when remix_intensity > 0
    if remix_intensity > 0.2:
        indices = list(range(len(scale_freqs)))
        chunks  = [indices[i:i+3] for i in range(0, len(indices), 3)]
        if remix_intensity > 0.6:
            random.shuffle(chunks)
        scale_freqs = [scale_freqs[i] for chunk in chunks for i in chunk]

    beat_track  = make_beat_track(tempo, duration_bars, genre_p["rhythm_pattern"])
    melody      = generate_melody(scale_freqs, tempo, duration_bars,
                                  waveform, genre_p["rhythm_pattern"])
    chord_track = generate_chord_track(scale_freqs, genre_p["chord_prog"],
                                       tempo, duration_bars, waveform)
    bassline    = generate_bassline(scale_freqs, genre_p["chord_prog"],
                                    tempo, duration_bars)

    # Trim all to shortest length
    min_len = min(len(beat_track), len(melody), len(chord_track), len(bassline))
    mix = (beat_track[:min_len] * 0.6 +
           melody[:min_len]     * 0.8 +
           chord_track[:min_len] * 0.55 +
           bassline[:min_len]   * 0.70)

    mix = apply_reverb(mix, amount=reverb)
    if waveform in ("sawtooth", "square"):
        mix = apply_lowpass(mix, cutoff=6000.0)
    if remix_intensity > 0.5:
        mix = apply_chorus(mix, depth=0.002 + 0.004 * remix_intensity)

    # Normalize
    peak = np.max(np.abs(mix))
    if peak > 0:
        mix = mix / peak * 0.85

    return _to_wav_bytes(mix)


def remix_audio(original_bytes: bytes, mood: str, genre: str,
                intensity: float = 0.5) -> bytes:
    """
    'Remix' an uploaded audio file by blending it with generated music.
    """
    import io as _io
    from scipy.io import wavfile as _wf

    mood_p  = MOOD_PROFILES[mood]
    genre_p = GENRE_PROFILES[genre]
    reverb  = mood_p["reverb"]

    # Read uploaded audio
    try:
        sr, data = _wf.read(_io.BytesIO(original_bytes))
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        # Resample if needed (simple linear interpolation)
        if sr != SAMPLE_RATE:
            new_len = int(len(data) * SAMPLE_RATE / sr)
            data = np.interp(np.linspace(0, len(data), new_len),
                             np.arange(len(data)), data)
        peak = np.max(np.abs(data))
        if peak > 0:
            data = data / peak * 0.7
    except Exception:
        # Fall back to pure generation if read fails
        return generate_music(mood, genre, duration_bars=8,
                               remix_intensity=intensity)

    # Determine bars from audio length
    tempo      = mood_p["tempo"]
    beat_dur   = 60.0 / tempo
    bar_dur    = beat_dur * 4
    bars       = max(4, int(len(data) / SAMPLE_RATE / bar_dur))
    bars       = min(bars, 16)

    scale_freqs = build_scale_freqs(mood_p["root"], mood_p["scale"],
                                    mood_p["octave"], n_notes=24)
    generated = generate_music(mood, genre, duration_bars=bars,
                                remix_intensity=intensity)
    sr2, gen_data = _wf.read(_io.BytesIO(generated))
    gen_data = gen_data.astype(np.float32)

    min_len = min(len(data), len(gen_data))
    blend = (data[:min_len] * (1 - intensity) +
             gen_data[:min_len] * intensity)
    blend = apply_reverb(blend, amount=reverb * intensity)
    peak = np.max(np.abs(blend))
    if peak > 0:
        blend = blend / peak * 0.85

    return _to_wav_bytes(blend)


# ─────────────────────────────────────────────
# WAV EXPORT
# ─────────────────────────────────────────────

def _to_wav_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 numpy array → WAV bytes."""
    audio_int16 = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()
