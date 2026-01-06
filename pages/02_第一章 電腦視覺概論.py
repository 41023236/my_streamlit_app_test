import streamlit as st
import pandas as pd
import numpy as np
# import openpyxl
import xlrd

import cv2
from streamlit_webrtc import webrtc_streamer, WebRtcMode


# 在這裡添加實驗一的具體內容，如圖表、數據等
st.header("buger繞過障礙物")
sample_video = open("image/30.mp4", "rb").read()
# Display Video using st.video() function
st.video(sample_video, start_time = 0)
# 在這裡添加實驗一的具體內容，如圖表、數據等
st.header("停止")
sample_video = open("image/3.mp4", "rb").read()
# Display Video using st.video() function
st.video(sample_video, start_time = 0)
st.header("前進")
sample_video = open("image/8.mp4", "rb").read()
# Display Video using st.video() function
st.video(sample_video, start_time = 0)
st.header("左轉")
sample_video = open("image/5.mp4", "rb").read()
# Display Video using st.video() function
st.video(sample_video, start_time = 0)
st.header("右轉")
sample_video = open("image/6.mp4", "rb").read()
# Display Video using st.video() function
st.video(sample_video, start_time = 0)
st.header("倒退")
sample_video = open("image/7.mp4", "rb").read()
# Display Video using st.video() function
st.video(sample_video, start_time = 0)