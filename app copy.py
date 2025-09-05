import streamlit as st
from audiorecorder import audiorecorder
import io
import openai
import os
from pydub import AudioSegment
from dotenv import load_dotenv
from query_engine import get_answer

# Load environment variables from the .env file
load_dotenv()

# Initialize OpenAI client with the API key from environment variables
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize chat session state
if "chat" not in st.session_state:
    st.session_state["chat"] = []

st.title("Hi! I'm Aggy, Your Medical Research Assistant")

st.markdown("### 🎤 What would you like to ask, please record your question.")

# --- Display existing chat messages
for msg in st.session_state["chat"]:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Assistant:** {msg['content']}")
        if "sources" in msg:
            st.markdown(f"*Sources:* {msg['sources']}")
        if "audio" in msg:
            st.audio(msg["audio"])

# --- Audio Recorder ---
audio_data_segment = audiorecorder("Click to record", "Recording...")

# Add a check to ensure we only process new audio
if len(audio_data_segment) > 0:
    # Check if this audio has already been processed in this session state
    if "last_audio" not in st.session_state or st.session_state["last_audio"] != audio_data_segment:
        st.session_state["last_audio"] = audio_data_segment
        
        # Convert AudioSegment to a byte stream
        wav_bytes_io = io.BytesIO()
        audio_data_segment.export(wav_bytes_io, format="wav")
        wav_bytes_io.seek(0)

        # Give the in-memory file a name and extension
        wav_bytes_io.name = "audio.wav"

        # Display the audio player
        st.audio(wav_bytes_io, format="audio/wav")

        with st.spinner("Transcribing..."):
            transcription = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=wav_bytes_io,
                language="en"
            )

        question_from_audio = transcription.text
        st.text_area("🎙️ Transcribed text (live check)", value=question_from_audio, height=80)

        with st.spinner("Thinking..."):
            # Pass the chat history to the get_answer function for multi-turn context
            answer, sources = get_answer(question_from_audio, chat_history=st.session_state["chat"])

        with st.spinner("🔊 Generating voice..."):
            audio_response = openai_client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=answer
            )
            audio_bytes = audio_response.read()

        # Save chat history
        st.session_state["chat"].append({"role": "user", "content": question_from_audio})
        st.session_state["chat"].append({"role": "assistant", "content": answer, "sources": sources, "audio": audio_bytes})
        
        st.rerun()

# --- Clear Chat ---
if st.button("Clear Chat"):
    st.session_state["chat"] = []
    st.success("Chat cleared!")