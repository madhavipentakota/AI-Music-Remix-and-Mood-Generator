# 🎵 AI Music Remix & Mood Generator

A **100% local, no-API-key** music creation and remixing platform built with **Python + Streamlit**.
Students can generate, remix, and explore music in any mood or genre without any musical expertise.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎼 **Generate Music** | Create original music by choosing a mood, genre, tempo, scale, and duration |
| 🔀 **Remix Upload** | Upload your own WAV file and blend it with AI-generated accompaniment |
| 🧭 **Explore & Learn** | Interactive scale visualizer, mood radar charts, genre breakdowns, and music theory glossary |
| 📚 **History** | Replay and download every track generated in your session |
| 📊 **Visualizations** | Real-time waveform, frequency spectrum, and mood radar charts |

---

## 🎭 Available Moods

`Happy` · `Sad` · `Energetic` · `Calm` · `Mysterious` · `Romantic` · `Angry` · `Dreamy`

## 🎸 Available Genres

`Pop` · `Hip-Hop` · `Rock` · `Jazz` · `Electronic` · `Classical` · `Reggae` · `R&B`

## 🎹 Available Scales

`Major` · `Minor` · `Pentatonic` · `Blues` · `Dorian` · `Mixolydian` · `Phrygian`

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install streamlit numpy scipy plotly pandas
```

> **Note:** `pydub` and `librosa` are optional (used only for advanced audio processing).
> The core app runs with just `streamlit numpy scipy plotly pandas`.

### 2. Run the app

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** in your browser.

---

## 🏗️ Project Structure

```
MusicProjectUsingAgenticAi/
├── app.py                  # Main Streamlit application
├── music_engine.py         # DSP-based music synthesis & remix engine
├── visualizer.py           # Plotly visualization helpers
├── requirements.txt        # Full dependencies (with optional libs)
├── requirements_minimal.txt # Minimal install (recommended for students)
├── .streamlit/
│   └── config.toml         # Theme configuration
└── README.md
```

---

## 🧠 How It Works (No AI Black Box!)

The "AI" in this app is **algorithmic music generation** using digital signal processing:

1. **Scale & Chord Theory** — Notes are selected from the chosen scale and root key
2. **Waveform Synthesis** — Sine, square, sawtooth, and triangle waves are synthesised
3. **ADSR Envelope** — Each note has attack/decay/sustain/release shaping
4. **Rhythm Patterns** — Genre-specific 8-step beat patterns drive the melody and drums
5. **Drum Synthesis** — Kick, snare, and hi-hat are synthesised from scratch
6. **Effects** — Reverb (comb filter), low-pass filter, and chorus are applied
7. **Mixing** — Melody, chords, bass, and drums are mixed and normalised

> Everything runs **100% on your machine** — no internet, no API keys, no data uploaded.

---

## 🎓 Educational Value

This project is designed for students to:
- Understand music theory concepts (scales, chords, tempo, modes)
- Explore how mood is expressed through musical parameters
- Learn about audio signal processing (waveforms, FFT, filters)
- Practice creative expression through AI-assisted composition
- Learn Python programming concepts (NumPy arrays, DSP, Streamlit UI)

---

## 🛠️ Customisation Ideas

- Add more scales (Lydian, Whole-tone, Chromatic)
- Add more waveforms (FM synthesis, wavetable)
- Implement export to MIDI using `midiutil`
- Add a lyrics generator using an LLM
- Deploy to Streamlit Cloud for classroom use

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `numpy` | Array math & DSP |
| `scipy` | Filters, FFT, WAV I/O |
| `plotly` | Interactive charts |
| `pandas` | Data tables in Explore tab |

---

*Built with ❤️ using Python + Streamlit · No API key · 100% local*
