import streamlit as st
import whisper
from transformers import pipeline
import tempfile
import os

# Load AI models
st.title("🎙️ AI-Powered Meeting Summarizer (Italian) 🇮🇹")
st.write("Upload a meeting audio file, and let AI generate a summary for you!")

# Upload audio file
audio_file = st.file_uploader("Upload an audio file (.mp3, .wav)", type=["mp3", "wav"])

if audio_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name

    # Whisper AI - Transcription
    st.write("🔄 Transcribing audio... (This may take a few minutes)")
    whisper_model = whisper.load_model("base")
    transcript = whisper_model.transcribe(temp_audio_path, language="it")["text"]
    
    # Save transcript
    with open("italian_transcript.txt", "w", encoding="utf-8") as f:
        f.write(transcript)

    st.success("✅ Transcription Completed!")
    st.text_area("📝 AI Transcription (Italian)", transcript, height=200)

    # Hugging Face Summarization
    st.write("🔄 Summarizing text...")
    summarizer = pipeline("summarization", model="it5/it5-small-summarization")
    summary = summarizer(transcript, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]

    # Save summary
    with open("ai_meeting_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    st.success("✅ AI Summary Generated!")
    st.text_area("📌 AI Summary", summary, height=150)

    # Manual Notes
    st.write("✍️ **Your Notes** - Compare them with AI's summary!")
    manual_notes = st.text_area("📝 Write your own meeting notes here", "", height=150)

    # Save manual notes if user enters them
    if st.button("Save Notes"):
        with open("manual_meeting_notes.txt", "w", encoding="utf-8") as f:
            f.write(manual_notes)
        st.success("✅ Your notes have been saved!")

    # Show AI vs. Human Notes
    if manual_notes:
        st.write("🤖 **AI vs. Human Challenge** 🧠")
        st.write("Compare AI-generated summary with your own notes. Who did better?")
        st.text_area("🤖 AI Summary", summary, height=150, disabled=True)
        st.text_area("🧠 Your Notes", manual_notes, height=150, disabled=True)

    # Cleanup temporary file
    os.remove(temp_audio_path)
