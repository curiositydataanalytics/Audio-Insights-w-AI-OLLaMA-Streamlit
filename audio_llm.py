# Data manipulation
import numpy as np
import datetime as dt
import pandas as pd
import geopandas as gpd

# Database and file handling
import os

# Data visualization
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz
import pydeck as pdk

from io import BytesIO
import speech_recognition as sr
import pydub
from langchain_community.llms import Ollama

path_cda = '\\CuriosityDataAnalytics'
path_wd = path_cda + '\\wd'
path_data = path_wd + '\\data'

# App config
#----------------------------------------------------------------------------------------------------------------------------------#
# Page config
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>
    .element-container {
        margin-top: -5px;
        margin-bottom: -5px;
        margin-left: -5px;
        margin-right: -5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# App title
st.title("Audio to Insights w/ AI")
st.divider()

with st.sidebar:
    st.image(path_cda + '\\logo.png')

#
#
st.header('Import Audio')

cols = st.columns((0.2,0.8))

input_type = cols[0].radio('', ['Upload audio file', 'Record audio'])

if input_type=='Upload audio file':
    audio = cols[1].file_uploader('Import audio file')
        
elif input_type=='Record audio':
    audio = cols[1].experimental_audio_input('Record audio')

@st.cache_data
def load_data():
    llm = Ollama(model='llama3.2:1b')

    return (llm)
llm = load_data()

if 'audio_txt' not in st.session_state:
    st.session_state.audio_txt = None
if 'text_output' not in st.session_state:
    st.session_state.text_output = ""


if audio and st.session_state.audio_txt is None:
    with st.spinner('Loading Audio'):
        audio_raw = pydub.AudioSegment.from_file(audio)

        audio_wav = BytesIO()
        audio_raw.export(audio_wav, format="wav")

        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_wav) as source:
            audio_txt = recognizer.record(source)
        st.session_state.audio_txt = recognizer.recognize_google(audio_txt)

if audio:
    st.header('Analyze Audio')
    cols[1].audio(audio)

    cols = st.columns(5, vertical_alignment='bottom')
    st.divider()

    # Full Text
    if cols[0].button('Speech-to-Text', use_container_width=True):
        st.session_state.text_output = st.session_state.audio_txt

    # Main Topics
    if cols[1].button('Main Topics', use_container_width=True):
        with st.spinner('Loading...'):
            st.session_state.text_output = llm.invoke('Identify the three main topics from the following text. Be very quick and concise: ' + st.session_state.audio_txt)

    # Translate
    language = cols[2].selectbox('', options=['French', 'Spanish', 'German'], placeholder="Translate", index=None)
    if language:
        with st.spinner('Loading...'):
            output = llm.invoke('Translate in ' + language + ' the following text: ' + st.session_state.audio_txt)
            st.session_state.text_output = output

    # Search
    words = cols[3].text_input('', placeholder='Search')
    if words:
        with st.spinner('Loading...'):
            output = llm.invoke('From the following text : --' + st.session_state.audio_txt + '--, output the sentences where the following words are mentionned: ' + words)
            output = output.replace(words, '**:orange[' + words + ']**')
            st.session_state.text_output = output

    # Question
    question = cols[4].text_input('', placeholder='Question')
    if question:
        with st.spinner('Loading...'):
            output = llm.invoke('From the following text : --' + st.session_state.audio_txt + '--, answer the following question: ' + question)
            st.session_state.text_output = output

    st.markdown(st.session_state.text_output)