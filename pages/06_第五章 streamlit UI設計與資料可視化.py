import streamlit as st

st.title("第五章 streamlit UI設計與資料可視化")


st.header("步驟1：安裝streamlit套件")
st.subheader("""main.py Code""")
code = '''"""pip install streamlit
'''
st.code(code, language='python')
st.header("步驟2：啟動streamlit，開啟網頁")
st.subheader("""main.py Code""")
code = '''"""streamlit run "01.py"  #01.py為範例檔，可檔名更改更改
'''
st.code(code, language='python')
st.markdown(
    '步驟3：開啟 **[streamlit](https://streamlit.io)** 官方的程式庫，從中可獲得各種程式以供網頁書寫'
)

st.header("顯示使用情況影片")
sample_video = open("image/29.mp4", "rb").read()
# Display Video using st.video() function
st.video(sample_video, start_time = 0)
