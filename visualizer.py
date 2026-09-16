"""
visualizer.py
Waveform and spectrum visualizations using Plotly.
"""

import numpy as np
import io
import wave
import plotly.graph_objects as go
from plotly.subplots import make_subplots


SAMPLE_RATE = 44100


def load_wav_bytes(wav_bytes: bytes) -> np.ndarray:
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    return data


def plot_waveform(wav_bytes: bytes, title: str = "Waveform") -> go.Figure:
    """Plot the audio waveform."""
    data = load_wav_bytes(wav_bytes)
    # Downsample for display
    step = max(1, len(data) // 4000)
    data_ds = data[::step]
    time_s  = np.arange(len(data_ds)) * step / SAMPLE_RATE

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_s, y=data_ds,
        mode="lines",
        line=dict(color="#3b82d4", width=1.2),
        fill="tozeroy",
        fillcolor="rgba(59,130,212,0.15)",
        name="Amplitude",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#1f2328")),
        xaxis=dict(title="Time (s)", color="#57606a", gridcolor="#e5e7eb"),
        yaxis=dict(title="Amplitude", color="#57606a", gridcolor="#e5e7eb",
                   range=[-1.1, 1.1]),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f7f8fa",
        margin=dict(l=50, r=20, t=45, b=40),
        height=220,
        showlegend=False,
    )
    return fig


def plot_spectrum(wav_bytes: bytes, title: str = "Frequency Spectrum") -> go.Figure:
    """Plot the frequency spectrum (FFT)."""
    data = load_wav_bytes(wav_bytes)
    # Use the middle 2 seconds for a representative spectrum
    mid   = len(data) // 2
    chunk = data[max(0, mid - SAMPLE_RATE): mid + SAMPLE_RATE]
    if len(chunk) < 512:
        chunk = data

    window    = np.hanning(len(chunk))
    fft_vals  = np.abs(np.fft.rfft(chunk * window))
    freqs     = np.fft.rfftfreq(len(chunk), d=1.0 / SAMPLE_RATE)
    db_vals   = 20 * np.log10(fft_vals + 1e-9)

    # Restrict to 20 Hz – 16 kHz
    mask = (freqs >= 20) & (freqs <= 16000)
    freqs  = freqs[mask]
    db_vals = db_vals[mask]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs, y=db_vals,
        mode="lines",
        line=dict(color="#7c5cd8", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(124,92,216,0.15)",
        name="dB",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#1f2328")),
        xaxis=dict(title="Frequency (Hz)", type="log", color="#57606a",
                   gridcolor="#e5e7eb"),
        yaxis=dict(title="Magnitude (dB)", color="#57606a", gridcolor="#e5e7eb"),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f7f8fa",
        margin=dict(l=55, r=20, t=45, b=40),
        height=220,
        showlegend=False,
    )
    return fig


def plot_piano_roll(scale_name: str, root: str, mood: str) -> go.Figure:
    """Draw a simple piano-roll-style scale diagram."""
    from music_engine import SCALES, NOTE_NAMES

    intervals  = SCALES.get(scale_name, [0, 2, 4, 5, 7, 9, 11])
    root_idx   = NOTE_NAMES.index(root)
    active     = [(root_idx + iv) % 12 for iv in intervals]
    labels     = NOTE_NAMES

    colors = ["#3b82d4" if i in active else "#e5e7eb" for i in range(12)]
    text   = [labels[i] if i in active else "" for i in range(12)]

    fig = go.Figure(go.Bar(
        x=labels,
        y=[1] * 12,
        marker_color=colors,
        text=text,
        textposition="inside",
        textfont=dict(color="white", size=12),
        hovertemplate="%{x}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"{mood} — {root} {scale_name} Scale",
                   font=dict(size=13, color="#1f2328")),
        xaxis=dict(showticklabels=True, color="#57606a"),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f7f8fa",
        margin=dict(l=20, r=20, t=45, b=30),
        height=160,
        showlegend=False,
        bargap=0.1,
    )
    return fig


def plot_mood_radar(mood: str) -> go.Figure:
    """Radar chart of mood characteristics."""
    mood_attrs = {
        "Happy":      [9, 3, 8, 5, 7, 4],
        "Sad":        [2, 8, 3, 7, 4, 6],
        "Energetic":  [8, 2, 9, 3, 6, 8],
        "Calm":       [5, 6, 2, 9, 3, 5],
        "Mysterious": [4, 7, 5, 6, 9, 4],
        "Romantic":   [7, 5, 4, 8, 5, 7],
        "Angry":      [3, 4, 9, 2, 6, 9],
        "Dreamy":     [6, 6, 3, 8, 8, 4],
    }
    attrs  = ["Energy", "Melancholy", "Intensity", "Calm", "Mystery", "Drive"]
    values = mood_attrs.get(mood, [5, 5, 5, 5, 5, 5])
    values_closed = values + [values[0]]
    attrs_closed  = attrs  + [attrs[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=attrs_closed,
        fill="toself",
        fillcolor="rgba(59,130,212,0.25)",
        line=dict(color="#3b82d4", width=2),
        name=mood,
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10],
                            color="#57606a", gridcolor="#e5e7eb"),
            angularaxis=dict(color="#1f2328"),
        ),
        title=dict(text=f"Mood Profile: {mood}", font=dict(size=13, color="#1f2328")),
        paper_bgcolor="#ffffff",
        margin=dict(l=30, r=30, t=55, b=20),
        height=280,
        showlegend=False,
    )
    return fig
