import streamlit as st
import os 
import ffmpeg

import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model

st.set_page_config(layout='wide')

options = os.listdir(os.path.join('data', 's1'))
selected_video = st.selectbox('Choose video', options)

col1, col2 = st.columns(2)

if options: 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')

        file_path = os.path.join('data','s1', selected_video)
        abs_path = os.path.abspath(file_path)
        
        if os.path.exists(abs_path):
            output_path = os.path.abspath('app/test_video.mp4')

            try:
                ffmpeg.input(abs_path).output(output_path, vcodec='libx264').run(overwrite_output=True)
            except ffmpeg.Error as e:
                raise

        else:
            raise FileNotFoundError(f"Video file not found at path: {file_path}")
        
        video = open('app/test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)


    with col2: 
        video = load_data(file_path)

        model = load_model(checkpoint_path='models-checkpoint96/checkpoint')
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()

        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
