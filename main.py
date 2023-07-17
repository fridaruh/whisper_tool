import streamlit as st
import openai
import docx
from docx import Document

# Set your OpenAI API key
openai.api_key = 'your-openai-api-key'

#Connect to openAI
def openai_connect():
    credential_openai= st.secrets["openai_creds"]
    openai.api_key = credential_openai.openai_api_key
    return openai
    

def transcribe_audio(audio_file):
    transcript = openai_connect().Audio.transcribe("whisper-1", audio_file)
    return transcript.text

st.title('Audio Transcription with OpenAI Whisper')

uploaded_file = st.file_uploader("Choose an audio file...", type=['mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm'])

if uploaded_file is not None:
    if st.button('Transcribe'):
        with st.spinner('Transcribing...'):
            transcript_text = transcribe_audio(uploaded_file)
            st.text_area('Transcript:', value=transcript_text, height=200, max_chars=None)
            
            doc = Document()
            doc.add_paragraph(transcript_text)
            doc.save("transcript.docx")
            st.success('Transcription complete. The transcript has been saved as a .docx file.')
