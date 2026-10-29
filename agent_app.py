import streamlit as st
import chromadb
import ollama
import sqlite3
import pandas as pd
import time
import os
import glob
import sounddevice as sd

os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\ffmpeg-9.0.1-essentials_build\bin"

from speech_io import (record_audio, transcribe, speak_and_save, get_amplitude_envelope,
                        start_recording, stop_recording, play_audio_async, get_audio_duration)
from charts_module import generate_interactive_chart
from generative_art import generate_data_art
from analytics_extra import detect_anomalies, export_conversation_to_pdf
from lipsync_module import get_visemes, VISEME_TO_EMOJI
from casual_chat import casual_chat, EMOTION_EMOJI
from cartoon_avatar import generate_all_faces, get_face_path, draw_face
from global_state import request_stop, clear_stop, is_stop_requested

st.set_page_config(page_title="Beyond the Chart — AI Agent", page_icon="🤖", layout="wide")

COMPANIES = ["Amazon", "Apple", "Google", "Microsoft", "Accenture", "Nvidia"]

st.markdown("""
<style>
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(0, 212, 255, 0.6); }
    70% { box-shadow: 0 0 0 25px rgba(0, 212, 255, 0); }
    100% { box-shadow: 0 0 0 0 rgba(0, 212, 255, 0); }
}
.avatar-circle {
    width: 90px; height: 90px; border-radius: 50%;
    background: linear-gradient(135deg, #00D4FF, #1E2761);
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 10px;
}
.avatar-idle { animation: pulse 2s infinite; font-size: 45px; }
</style>
""", unsafe_allow_html=True)

# ==================== GLOBAL SIDEBAR STOP ====================
st.sidebar.markdown("## 🛑 Emergency Controls")
if st.sidebar.button("🛑 STOP EVERYTHING (voice + reply)", type="primary"):
    request_stop()
    sd.stop()
    st.session_state["analyst_question"] = ""
    st.session_state["casual_question"] = ""
    st.sidebar.success("Stopped. You can type a new question now.")
    st.rerun()
st.sidebar.markdown("---")

st.markdown("<h1 style='color:#00D4FF;'>Beyond the Chart — AI Agent</h1>", unsafe_allow_html=True)

# ---------------- Cached resources ----------------
@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection(name="stock_knowledge")

@st.cache_resource
def load_price_data():
    return pd.read_csv("data/df.csv", parse_dates=['date'])

collection = get_chroma_collection()
price_df = load_price_data()

def init_log_db():
    conn = sqlite3.connect("conversation_log.db")
    conn.execute("""CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        question TEXT, answer TEXT, model_used TEXT)""")
    existing_columns = [row[1] for row in conn.execute("PRAGMA table_info(logs)").fetchall()]
    if 'mode' not in existing_columns:
        conn.execute("ALTER TABLE logs ADD COLUMN mode TEXT DEFAULT 'analyst'")
    conn.commit()
    conn.close()

def log_interaction(question, answer, model_used, mode="analyst"):
    conn = sqlite3.connect("conversation_log.db")
    conn.execute("INSERT INTO logs (question, answer, model_used, mode) VALUES (?, ?, ?, ?)",
                 (question, answer, model_used, mode))
    conn.commit()
    conn.close()

init_log_db()

# Apply any pending transcription BEFORE the analyst_question widget is created this run
if "pending_analyst_transcript" in st.session_state:
    st.session_state["analyst_question"] = st.session_state.pop("pending_analyst_transcript")

def find_mentioned_company(text):
    for c in COMPANIES:
        if c.lower() in text.lower():
            return c
    return None

def animate_talking(avatar_slot, text, emotion="neutral"):
    generate_all_faces(emotion)
    speak_and_save(text, emotion=emotion)
    duration = get_audio_duration()
    play_audio_async()

    visemes = get_visemes()
    if visemes:
        start_time = time.time()
        for cue in visemes:
            if is_stop_requested():
                break
            wait = cue['start'] - (time.time() - start_time)
            if wait > 0:
                time.sleep(wait)
            face_path = get_face_path(cue['value'], emotion)
            avatar_slot.image(face_path, width=150)
    else:
        envelope = get_amplitude_envelope()
        frame_time = duration / len(envelope) if envelope else 0.15
        for loudness in envelope:
            if is_stop_requested():
                break
            viseme_key = "D" if loudness > 0.6 else ("B" if loudness > 0.2 else "X")
            face_path = get_face_path(viseme_key, emotion)
            avatar_slot.image(face_path, width=150)
            time.sleep(frame_time)

    idle_path = draw_face("closed", emotion, f"faces/idle_{emotion}.png")
    avatar_slot.image(idle_path, width=150)

# ==================== TABS ====================
tab_answers, tab_chat, tab_graphs, tab_generate, tab_database, tab_history, tab_manual = st.tabs(
    ["💬 Analyst Answers", "🗨️ Casual Talk", "📊 Graphs", "🎨 Generations", "🗄️ Database", "📜 History", "📖 Manual"]
)

# ---------------- TAB: Analyst Answers ----------------
with tab_answers:
    avatar_slot_1 = st.empty()
    avatar_slot_1.image(draw_face("closed", "neutral", "faces/idle_neutral.png"), width=150)

    model_choice = st.selectbox("Choose model:", ["llama3.2", "llama3.1:8b", "deepseek-r1:1.5b"])
    speak_answer = st.checkbox("Speak the answer out loud", key="speak_analyst")

    if 'recording' not in st.session_state:
        st.session_state.recording = False

    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input("Type your question:", key="analyst_question")
    with col2:
        if not st.session_state.recording:
            if st.button("🎤 Start"):
                st.session_state.recording = True
                start_recording()
                st.rerun()
        else:
            if st.button("⏹️ Stop"):
                st.session_state.recording = False
                stop_recording()
                transcribed = transcribe()
                st.session_state["pending_analyst_transcript"] = transcribed
                st.rerun()

    if question:
        clear_stop()
        results = collection.query(query_texts=[question], n_results=5)
        relevant_facts = "\n".join(results['documents'][0])
        prompt = f"""You are a financial analysis assistant. Answer using ONLY the facts below.

FACTS:
{relevant_facts}

QUESTION: {question}

ANSWER:"""
        from streaming_helper import start_streaming
        import time as _time

        state, thread = start_streaming(prompt, model_choice)

        st.subheader("Answer")
        answer_slot = st.empty()

        while not state.done and not is_stop_requested():
            answer_slot.write(state.text + " ▌")
            _time.sleep(0.15)

        if is_stop_requested():
            state.stop_requested = True

        answer_slot.write(state.text)
        answer = state.text

        if answer.strip():
            mentioned = find_mentioned_company(question)
            if mentioned:
                st.plotly_chart(generate_interactive_chart(price_df, mentioned), use_container_width=True)

            log_interaction(question, answer, model_choice, mode="analyst")

            pdf_path = export_conversation_to_pdf(question, answer)
            if pdf_path:
                with open(pdf_path, "rb") as f:
                    st.download_button("📄 Download as PDF", f, file_name="ai_report.pdf")

            if speak_answer:
                try:
                    animate_talking(avatar_slot_1, answer)
                except Exception as e:
                    st.warning(f"Voice playback issue: {e}")

# ---------------- TAB: Casual Talk ----------------
with tab_chat:
    st.write("A different AI personality — casual, warm, and expressive. Not restricted to your project data.")
    avatar_slot_2 = st.empty()
    avatar_slot_2.image(draw_face("closed", "neutral", "faces/idle_neutral.png"), width=150)

    speak_casual = st.checkbox("Speak the reply out loud", key="speak_casual")

    if 'casual_history' not in st.session_state:
        st.session_state.casual_history = []

    casual_msg = st.text_input("Say something:", key="casual_question")

    if casual_msg:
        clear_stop()
        with st.spinner("..."):
            reply, emotion = casual_chat(casual_msg, st.session_state.casual_history)

        emoji = EMOTION_EMOJI.get(emotion, "🙂")
        st.markdown(f"### {emoji} {reply}")

        st.session_state.casual_history.append({"role": "user", "content": casual_msg})
        st.session_state.casual_history.append({"role": "assistant", "content": reply})

        log_interaction(casual_msg, reply, "llama3.2", mode=f"casual-{emotion}")

        if speak_casual:
            try:
                animate_talking(avatar_slot_2, reply, emotion=emotion)
            except Exception as e:
                st.warning(f"Voice playback issue: {e}")

# ---------------- TAB: Graphs ----------------
with tab_graphs:
    graph_company = st.selectbox("Show interactive chart for:", COMPANIES, key="graph_sel")
    st.plotly_chart(generate_interactive_chart(price_df, graph_company), use_container_width=True)

    st.markdown("---")
    st.subheader("📁 Original Analysis Charts (from your Jupyter project)")
    original_files = sorted(glob.glob("original_charts/*.png"))
    if original_files:
        cols = st.columns(3)
        for i, f in enumerate(original_files):
            cols[i % 3].image(f, caption=os.path.basename(f), use_container_width=True)
    else:
        st.caption("Run extract_notebook_images.py once to populate this section.")

# ---------------- TAB: Generations ----------------
with tab_generate:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🎨 Data-Driven Art")
        art_company = st.selectbox("Company:", COMPANIES, key="art_sel")
        if st.button("Generate Data Art"):
            with st.spinner("Generating..."):
                path = generate_data_art(price_df[price_df['company']==art_company], art_company,
                                          filename=f"art_{art_company}.png")
                st.image(path)
    with col_b:
        st.subheader("🖼️ AI Image Generation")
        img_prompt = st.text_input("Prompt:")
        quality = st.radio("Speed vs Quality:", ["⚡ Fast (~1-2 min)", "🎯 Quality (~4-8 min)"])
        steps = 15 if "Fast" in quality else 35
        if st.button("Generate Image"):
            import subprocess
            subprocess.Popen(["python", "background_image_gen.py", img_prompt, "generated_image.png", str(steps)])
            st.warning("Generating in background — check the Database tab in a few minutes.")

# ---------------- TAB: Database ----------------
with tab_database:
    st.subheader("🗄️ Saved Files")
    files = sorted(glob.glob("*.png") + glob.glob("*.pdf"), key=os.path.getmtime, reverse=True)
    if files:
        for f in files:
            c1, c2, c3 = st.columns([3, 1, 1])
            c1.write(os.path.basename(f))
            if f.endswith(".png"):
                c2.image(f, width=60)
            if c3.button("🗑️ Delete", key=f"del_{f}"):
                os.remove(f)
                st.rerun()
    else:
        st.caption("No saved files yet.")

    st.markdown("---")
    if st.button("🗑️ Clear Entire Conversation Log"):
        conn = sqlite3.connect("conversation_log.db")
        conn.execute("DELETE FROM logs")
        conn.commit()
        conn.close()
        st.success("Cleared.")
        st.rerun()

# ---------------- TAB: History ----------------
with tab_history:
    conn = sqlite3.connect("conversation_log.db")
    rows = conn.execute("SELECT timestamp, question, answer, model_used, mode FROM logs ORDER BY id DESC LIMIT 20").fetchall()
    conn.close()
    for ts, q, a, m, mode in rows:
        st.markdown(f"**[{ts}] ({m} · {mode})** — *{q}*")
        st.write(a)
        st.markdown("---")

# ---------------- TAB: Manual ----------------
with tab_manual:
    st.markdown("""
| Tab | What It Does | Powered By |
|---|---|---|
| **💬 Analyst Answers** | Factual, grounded answers using only your real validated project data | Ollama + ChromaDB (RAG) |
| **🗨️ Casual Talk** | A separate, warm, emotionally expressive AI companion — not data-restricted | Ollama with a distinct personality prompt |
| **📊 Graphs** | Interactive charts + your original Jupyter analysis charts | Plotly + extracted notebook images |
| **🎨 Generations** | Data-driven generative art (instant) + AI image generation (slow, Stable Diffusion) | NumPy/Matplotlib + Diffusers |
| **🗄️ Database** | View and manually delete any saved chart, art, or PDF; clear history | Direct file management |
| **📜 History** | Full log of every question asked, with timestamps | SQLite |
| **📖 Manual** | This page | — |

**🛑 Emergency Stop (top of sidebar):** works from any tab. Stops audio instantly, halts any in-progress AI reply, and clears both question boxes so you can type immediately.
    """)