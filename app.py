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

# Initialize a key for the audio recorder to allow it to be programmatically cleared
if "audio_recorder_key" not in st.session_state:
    st.session_state["audio_recorder_key"] = 0

# Custom CSS for better styling
st.markdown("""
<style>
.chat-container {
    max-height: 400px;
    overflow-y: auto;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    background-color: #f9f9f9;
}

.question-box {
    background-color: #e3f2fd;
    padding: 10px;
    border-radius: 8px;
    margin: 8px 0;
    border-left: 4px solid #2196f3;
}

.answer-box {
    background-color: #f3e5f5;
    padding: 10px;
    border-radius: 8px;
    margin: 8px 0;
    border-left: 4px solid #9c27b0;
}

.conversation-header {
    font-size: 16px;
    font-weight: bold;
    color: #1976d2;
    margin: 15px 0 5px 0;
}
</style>
""", unsafe_allow_html=True)

st.title("Hi! I'm Aggy, Your Medical Research Assistant")
st.markdown("##### Ask anything from the document")

# --- Display Chat History in Proper Flow Format ---
if st.session_state["chat"]:
    st.markdown("#### Conversation History")
    
    with st.container():
        # Group messages by conversation pairs
        conversations = []
        current_conversation = {}
        
        for msg in st.session_state["chat"]:
            if msg["role"] == "user":
                if current_conversation:
                    conversations.append(current_conversation)
                current_conversation = {"question": msg}
            elif msg["role"] == "assistant":
                current_conversation["answer"] = msg
        
        if current_conversation:
            conversations.append(current_conversation)
        
        for i, conversation in enumerate(reversed(conversations), 1):
            conversation_num = len(conversations) - i + 1
            
            with st.expander(f"Conversation {conversation_num}", expanded=(i == 1)):
                st.markdown('<div class="conversation-header">🎙️ Your Question:</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="question-box">{conversation["question"]["content"]}</div>', unsafe_allow_html=True)
                
                if "audio" in conversation["question"]:
                    st.markdown("**🔊 Question Audio:**")
                    st.audio(conversation["question"]["audio"], format="audio/wav")
                
                if "answer" in conversation:
                    st.markdown('<div class="conversation-header">🤖 Assistant\'s Answer:</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="answer-box">{conversation["answer"]["content"]}</div>', unsafe_allow_html=True)
                    
                    if "sources" in conversation["answer"]:
                        st.markdown(f"**📚 Sources:** {conversation['answer']['sources']}")
                    
                    if "audio" in conversation["answer"]:
                        st.markdown("**🔊 Answer Audio:**")
                        st.audio(conversation["answer"]["audio"], format="audio/mp3")
                
                st.markdown("---")

# --- New Question Section ---
st.markdown("#### Ask a New Question")

# Audio Recorder with a dynamic key
audio_data_segment = audiorecorder("🎙️ Record", "🔴 Recording...", key=f"audio_recorder_{st.session_state['audio_recorder_key']}")

# Process new audio
if len(audio_data_segment) > 0:
    # Check if this is new audio
    audio_hash = hash(audio_data_segment.raw_data)
    
    if "last_audio_hash" not in st.session_state or st.session_state["last_audio_hash"] != audio_hash:
        st.session_state["last_audio_hash"] = audio_hash
        
        # Convert AudioSegment to a byte stream
        wav_bytes_io = io.BytesIO()
        audio_data_segment.export(wav_bytes_io, format="wav")
        wav_bytes_io.seek(0)
        wav_bytes_io.name = "audio.wav"
        
        question_audio_bytes = wav_bytes_io.getvalue()
        
        with st.spinner("Processing..."):
            wav_bytes_io.seek(0)
            transcription = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=wav_bytes_io,
                language="en"
            )
            question_text = transcription.text
            
            answer, sources = get_answer(question_text, chat_history=st.session_state["chat"])
        
        with st.spinner("🔊 Generating response..."):
            audio_response = openai_client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=answer
            )
            answer_audio_bytes = audio_response.read()
        
        st.session_state["chat"].append({
            "role": "user", 
            "content": question_text, 
            "audio": question_audio_bytes
        })
        st.session_state["chat"].append({
            "role": "assistant", 
            "content": answer, 
            "sources": sources, 
            "audio": answer_audio_bytes
        })
        
        st.rerun()

# --- Control Buttons ---
st.markdown("---")
st.markdown("##### ⚙️ Controls")
col1, col2 = st.columns([1, 2])

with col1:
    if st.button("## 🗑️ Clear Conversations"):
        # Reset the chat history
        st.session_state["chat"] = []
        
        # Increment the audio recorder key to reset the widget
        st.session_state["audio_recorder_key"] += 1
        
        # Remove the last audio hash to prevent old audio from being re-processed
        if "last_audio_hash" in st.session_state:
            del st.session_state["last_audio_hash"]
        
        st.success("All conversations cleared! Start a new one.")
        st.rerun()

with col2:
    if st.session_state["chat"]:
        export_text = "# Aggy Conversation Export\n\n"
        for i, msg in enumerate(st.session_state["chat"]):
            if msg["role"] == "user":
                export_text += f"**Question:** {msg['content']}\n\n"
            else:
                export_text += f"**Answer:** {msg['content']}\n"
                if "sources" in msg:
                    export_text += f"**Sources:** {msg['sources']}\n"
                export_text += "\n---\n\n"
        
        st.download_button(
            label="## 📥 Download Conversations",
            data=export_text,
            file_name=f"aggy_conversation.txt",
            mime="text/plain"
        )