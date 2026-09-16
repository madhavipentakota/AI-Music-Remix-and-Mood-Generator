"""
app.py — AI Music Remix & Mood Generator
Streamlit application entry point.
Run with:  streamlit run app.py
"""

import streamlit as st
import time
import random
import base64
import io

# ─── Page config must be first Streamlit call ───────────────────────────────
st.set_page_config(
    page_title="🎵 AI Music Remix & Mood Generator",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS (minimal, Streamlit-native) ──────────────────────────────────
st.markdown("""
<style>
/* Overall background */
.stApp { background-color: #f0f2f6; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    color: white;
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stSelectbox label { color: #a0aec0 !important; }

/* Cards */
.music-card {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

/* Mood badge */
.mood-badge {
    display: inline-block;
    background: #3b82f6;
    color: white;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 13px;
    font-weight: 600;
    margin: 4px 4px 4px 0;
}

/* Genre badge */
.genre-badge {
    display: inline-block;
    background: #7c3aed;
    color: white;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 13px;
    font-weight: 600;
    margin: 4px 4px 4px 0;
}

/* Section headers */
.section-title {
    font-size: 18px;
    font-weight: 700;
    color: #1f2328;
    border-left: 4px solid #3b82f6;
    padding-left: 10px;
    margin-bottom: 12px;
}

/* Info box */
.info-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    padding: 12px 16px;
    color: #1e40af;
    font-size: 14px;
}

/* Tip box */
.tip-box {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 8px;
    padding: 12px 16px;
    color: #166534;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


# ─── Imports (after set_page_config) ────────────────────────────────────────
from music_engine import (
    MOOD_PROFILES, GENRE_PROFILES, SCALES,
    generate_music, remix_audio,
)
from visualizer import (
    plot_waveform, plot_spectrum,
    plot_piano_roll, plot_mood_radar,
)


# ────────────────────────────────────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ────────────────────────────────────────────────────────────────────────────
if "generated_audio" not in st.session_state:
    st.session_state.generated_audio = None
if "remix_audio" not in st.session_state:
    st.session_state.remix_audio = None
if "history" not in st.session_state:
    st.session_state.history = []       # list of dicts
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "🎼 Generate"


# ────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎵 AI Music Studio")
    st.markdown("---")

    st.markdown("### 🎭 Mood")
    mood = st.selectbox(
        "Select mood",
        list(MOOD_PROFILES.keys()),
        index=0,
        label_visibility="collapsed",
    )
    mood_info = MOOD_PROFILES[mood]
    st.markdown(f"<small style='color:#a0aec0'>💬 {mood_info['description']}</small>",
                unsafe_allow_html=True)

    st.markdown("### 🎸 Genre")
    genre = st.selectbox(
        "Select genre",
        list(GENRE_PROFILES.keys()),
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("### ⚙️ Settings")
    duration_bars = st.slider("Duration (bars)", 4, 16, 8, step=4,
                               help="More bars = longer clip")
    custom_tempo = st.slider(
        "Tempo (BPM)", 40, 200,
        value=int(mood_info["tempo"]),
        help="Override the mood's default tempo",
    )
    custom_scale = st.selectbox(
        "Scale", list(SCALES.keys()),
        index=list(SCALES.keys()).index(mood_info["scale"]),
        help="Override the mood's default scale",
    )

    st.markdown("---")
    st.markdown("### 🔀 Remix Intensity")
    remix_intensity = st.slider(
        "Intensity", 0.0, 1.0, 0.5, 0.05,
        help="0 = faithful, 1 = experimental",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:#718096'>🤖 Powered by NumPy + SciPy DSP<br>"
        "No API key required · Runs 100% locally</small>",
        unsafe_allow_html=True,
    )


# ────────────────────────────────────────────────────────────────────────────
# HEADER
# ────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 20px 0 8px'>
  <h1 style='font-size:2.4rem; font-weight:800; color:#1f2328; margin-bottom:4px'>
    🎵 AI Music Remix & Mood Generator
  </h1>
  <p style='color:#57606a; font-size:1.05rem; margin:0'>
    Create, remix, and explore music in any mood or genre — no music skills needed!
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# MAIN TABS
# ────────────────────────────────────────────────────────────────────────────
tab_generate, tab_remix, tab_explore, tab_history = st.tabs([
    "🎼 Generate Music",
    "🔀 Remix Upload",
    "🧭 Explore & Learn",
    "📚 My History",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — GENERATE
# ══════════════════════════════════════════════════════════════════════════════
with tab_generate:
    col_left, col_right = st.columns([1.3, 1], gap="large")

    with col_left:
        st.markdown("<div class='section-title'>🎹 Your Music Settings</div>",
                    unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Mood:** <span class='mood-badge'>{mood}</span>",
                        unsafe_allow_html=True)
            st.markdown(f"**Scale:** `{custom_scale}` in `{mood_info['root']}`")
        with c2:
            st.markdown(f"**Genre:** <span class='genre-badge'>{genre}</span>",
                        unsafe_allow_html=True)
            st.markdown(f"**Tempo:** `{custom_tempo} BPM` · **Bars:** `{duration_bars}`")

        st.markdown("<br>", unsafe_allow_html=True)

        btn_col, rand_col = st.columns([2, 1])
        with btn_col:
            generate_clicked = st.button(
                "🎵 Generate Music", type="primary",
                use_container_width=True,
            )
        with rand_col:
            random_clicked = st.button(
                "🎲 Random", use_container_width=True,
                help="Generate with random mood + genre",
            )

        if random_clicked:
            st.session_state["rand_mood"]  = random.choice(list(MOOD_PROFILES.keys()))
            st.session_state["rand_genre"] = random.choice(list(GENRE_PROFILES.keys()))
            st.rerun()

        # Auto-apply random selections if they exist
        if "rand_mood" in st.session_state:
            mood  = st.session_state.pop("rand_mood")
            genre = st.session_state.pop("rand_genre", genre)
            st.info(f"🎲 Random pick: **{mood}** mood + **{genre}** genre")

        if generate_clicked:
            with st.spinner(f"🎼 Composing {mood} × {genre} music..."):
                progress = st.progress(0, text="Synthesising waveforms…")
                for p in range(0, 80, 20):
                    time.sleep(0.12)
                    progress.progress(p, text="Synthesising waveforms…")

                audio_bytes = generate_music(
                    mood=mood,
                    genre=genre,
                    duration_bars=duration_bars,
                    custom_tempo=custom_tempo,
                    custom_scale=custom_scale,
                    remix_intensity=remix_intensity,
                )
                progress.progress(90, text="Applying effects…")
                time.sleep(0.1)
                progress.progress(100, text="Done!")
                time.sleep(0.2)
                progress.empty()

                st.session_state.generated_audio = audio_bytes
                st.session_state.history.append({
                    "type": "Generate",
                    "mood": mood,
                    "genre": genre,
                    "tempo": custom_tempo,
                    "scale": custom_scale,
                    "bars": duration_bars,
                    "audio": audio_bytes,
                })
            st.success("✅ Music generated! Listen and download below.")

        if st.session_state.generated_audio:
            st.markdown("---")
            st.markdown("<div class='section-title'>▶️ Playback & Download</div>",
                        unsafe_allow_html=True)
            st.audio(st.session_state.generated_audio, format="audio/wav")

            st.download_button(
                "⬇️ Download WAV",
                data=st.session_state.generated_audio,
                file_name=f"ai_music_{mood}_{genre}.wav",
                mime="audio/wav",
                use_container_width=True,
            )

    with col_right:
        if st.session_state.generated_audio:
            st.markdown("<div class='section-title'>📊 Visualizations</div>",
                        unsafe_allow_html=True)
            viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Waveform", "Spectrum", "Radar"])
            with viz_tab1:
                st.plotly_chart(
                    plot_waveform(st.session_state.generated_audio,
                                  f"{mood} × {genre} — Waveform"),
                    use_container_width=True,
                )
            with viz_tab2:
                st.plotly_chart(
                    plot_spectrum(st.session_state.generated_audio,
                                  f"{mood} × {genre} — Spectrum"),
                    use_container_width=True,
                )
            with viz_tab3:
                st.plotly_chart(
                    plot_mood_radar(mood),
                    use_container_width=True,
                )
        else:
            st.markdown("""
<div class='info-box'>
ℹ️ <strong>How it works</strong><br><br>
1️⃣ Pick a <b>mood</b> and <b>genre</b> in the sidebar<br>
2️⃣ Adjust tempo, scale, and duration<br>
3️⃣ Hit <b>🎵 Generate Music</b><br>
4️⃣ Listen, download, or explore the visualizations!<br><br>
<em>No music knowledge or external API required.</em>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — REMIX
# ══════════════════════════════════════════════════════════════════════════════
with tab_remix:
    st.markdown("<div class='section-title'>🔀 Upload & Remix Your Track</div>",
                unsafe_allow_html=True)

    st.markdown("""
<div class='info-box'>
🎵 Upload a <strong>.wav</strong> file and the AI will blend it with a generated
accompaniment in your chosen <strong>mood + genre</strong>.
The <em>Remix Intensity</em> slider (sidebar) controls the balance between
your original track and the generated layer.
</div>
<br>
""", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a WAV file to remix",
        type=["wav"],
        help="WAV files only. Keep under 30 seconds for best results.",
    )

    if uploaded:
        st.audio(uploaded, format="audio/wav")
        st.caption("⬆️ Your original track")

        r_col1, r_col2 = st.columns([2, 1])
        with r_col1:
            remix_btn = st.button(
                "🔀 Create Remix", type="primary",
                use_container_width=True,
            )
        with r_col2:
            st.metric("Remix Intensity", f"{int(remix_intensity * 100)}%")

        if remix_btn:
            with st.spinner("🎛️ Blending your track with AI music…"):
                progress2 = st.progress(0, text="Analysing audio…")
                for p in range(0, 70, 20):
                    time.sleep(0.15)
                    progress2.progress(p)

                remix_bytes = remix_audio(
                    original_bytes=uploaded.read(),
                    mood=mood,
                    genre=genre,
                    intensity=remix_intensity,
                )
                progress2.progress(100, text="Done!")
                time.sleep(0.2)
                progress2.empty()

                st.session_state.remix_audio = remix_bytes
                st.session_state.history.append({
                    "type": "Remix",
                    "mood": mood,
                    "genre": genre,
                    "tempo": custom_tempo,
                    "scale": custom_scale,
                    "bars": duration_bars,
                    "audio": remix_bytes,
                })
            st.success("✅ Remix complete!")

    if st.session_state.remix_audio:
        st.markdown("---")
        st.markdown("<div class='section-title'>▶️ Your Remix</div>",
                    unsafe_allow_html=True)
        st.audio(st.session_state.remix_audio, format="audio/wav")

        dl_col, viz_col = st.columns(2)
        with dl_col:
            st.download_button(
                "⬇️ Download Remix WAV",
                data=st.session_state.remix_audio,
                file_name=f"remix_{mood}_{genre}.wav",
                mime="audio/wav",
                use_container_width=True,
            )
        with viz_col:
            if st.button("📊 Show Remix Waveform", use_container_width=True):
                st.plotly_chart(
                    plot_waveform(st.session_state.remix_audio, "Remix Waveform"),
                    use_container_width=True,
                )

    elif not uploaded:
        st.markdown("""
<div class='tip-box'>
💡 <strong>Tip:</strong> Record yourself humming, clapping, or singing — then upload
that WAV and the AI will build a full accompaniment around it!
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EXPLORE & LEARN
# ══════════════════════════════════════════════════════════════════════════════
with tab_explore:
    st.markdown("<div class='section-title'>🧭 Explore Music Theory & Moods</div>",
                unsafe_allow_html=True)

    exp_col1, exp_col2 = st.columns([1, 1], gap="large")

    with exp_col1:
        st.markdown("#### 🎹 Scale Visualizer")
        viz_mood   = st.selectbox("Mood for scale", list(MOOD_PROFILES.keys()),
                                   key="viz_mood")
        viz_scale  = MOOD_PROFILES[viz_mood]["scale"]
        viz_root   = MOOD_PROFILES[viz_mood]["root"]
        st.plotly_chart(
            plot_piano_roll(viz_scale, viz_root, viz_mood),
            use_container_width=True,
        )

        st.markdown("#### 🎵 Mood Details")
        for m_name, m_data in MOOD_PROFILES.items():
            with st.expander(f"{m_name}", expanded=(m_name == viz_mood)):
                ic1, ic2, ic3 = st.columns(3)
                ic1.metric("Tempo", f"{m_data['tempo']} BPM")
                ic2.metric("Scale", m_data["scale"])
                ic3.metric("Root", m_data["root"])
                st.markdown(f"*{m_data['description']}*")
                st.markdown(f"**Waveform:** `{m_data['waveform']}` · "
                            f"**Reverb:** `{m_data['reverb']}`")

    with exp_col2:
        st.markdown("#### 🕸️ Mood Radar Chart")
        radar_mood = st.selectbox("Pick a mood", list(MOOD_PROFILES.keys()),
                                   key="radar_mood")
        st.plotly_chart(plot_mood_radar(radar_mood), use_container_width=True)

        st.markdown("#### 🎸 Genre Breakdown")
        genre_table = []
        for g_name, g_data in GENRE_PROFILES.items():
            genre_table.append({
                "Genre": g_name,
                "Chord Prog": str(g_data["chord_prog"]),
                "Arp Speed": g_data["arp_speed"],
            })

        import pandas as pd
        df = pd.DataFrame(genre_table)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("#### 📖 Music Theory Glossary")
        glossary = {
            "Scale": "A set of musical notes ordered by pitch that form the basis of a melody.",
            "Chord": "Three or more notes played simultaneously to create harmony.",
            "Tempo (BPM)": "Beats Per Minute — how fast or slow a piece of music is.",
            "Reverb": "The echo effect that makes music sound like it's in a large space.",
            "Waveform": "The shape of a sound wave (sine=smooth, square=harsh, sawtooth=bright).",
            "Arpeggio": "Notes of a chord played one after another in sequence.",
            "Mode": "A type of scale with a specific pattern of intervals (e.g. Dorian, Phrygian).",
            "ADSR": "Attack-Decay-Sustain-Release — how a note fades in and out.",
        }
        for term, definition in glossary.items():
            with st.expander(f"📌 {term}"):
                st.markdown(definition)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("<div class='section-title'>📚 Generation History</div>",
                unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
<div class='info-box'>
No music generated yet in this session.<br>
Head to <b>🎼 Generate Music</b> or <b>🔀 Remix Upload</b> to start creating!
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown(f"**{len(st.session_state.history)} track(s)** generated this session.")
        st.markdown("---")
        for i, item in enumerate(reversed(st.session_state.history), 1):
            with st.expander(
                f"#{len(st.session_state.history) - i + 1}  |  "
                f"{item['type']} — {item['mood']} × {item['genre']}  "
                f"| {item['tempo']} BPM · {item['bars']} bars",
                expanded=(i == 1),
            ):
                hc1, hc2, hc3, hc4 = st.columns(4)
                hc1.metric("Type",  item["type"])
                hc2.metric("Mood",  item["mood"])
                hc3.metric("Genre", item["genre"])
                hc4.metric("Tempo", f"{item['tempo']} BPM")

                st.audio(item["audio"], format="audio/wav")
                st.download_button(
                    f"⬇️ Download #{len(st.session_state.history) - i + 1}",
                    data=item["audio"],
                    file_name=f"track_{len(st.session_state.history)-i+1}_{item['mood']}_{item['genre']}.wav",
                    mime="audio/wav",
                    key=f"dl_hist_{i}",
                )

        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.history = []
            st.rerun()


# ────────────────────────────────────────────────────────────────────────────
# FOOTER
# ────────────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<hr style='border:none;border-top:1px solid #e5e7eb;margin:0'>
<div style='text-align:center;padding:12px;color:#57606a;font-size:13px'>
  🎵 AI Music Remix &amp; Mood Generator &nbsp;·&nbsp;
  Built with <b>Python + Streamlit</b> &nbsp;·&nbsp;
  DSP by <b>NumPy / SciPy</b> &nbsp;·&nbsp;
  No API key · 100% local
</div>
""", unsafe_allow_html=True)
