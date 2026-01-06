import streamlit as st

st.markdown("""
<h2 style="text-align:center;">智慧機械設計</h2>
<h3 style="text-align:center;">學期團隊作業 / 專案設計</h3>

<p style="text-align:center;">學年：114學年度第1學期</p>
<p style="text-align:center;">組別：第三組</p>
<p style="text-align:center;">題目：ROS自主移動平台與AI整合之研究</p>
<p style="text-align:center;">成員：11473134／黃柏閎</p>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .center-table {
        margin-left: auto;
        margin-right: auto;
        width: 70%;
        border-collapse: collapse;
    }
    .center-table th, .center-table td {
        border: 1px solid #999;
        padding: 8px;
        text-align: center;
        font-size: 16px;
    }
    .center-table th {
        background-color: #f2f2f2;
    }
    </style>

    <table class="center-table">
        <tr>
            <th>項次</th>
            <th>學號</th>
            <th>姓名</th>
            <th>分工內容</th>
            <th>貢獻度</th>
        </tr>
        <tr>
            <td>1</td>
            <td>11473134</td>
            <td>黃柏閎</td>
            <td>全篇主要程式架構撰寫、除錯與網頁報告統整、YOLO的照片標註、成果攝影與拍照、步驟截圖與報告構思</td>
            <td>100%</td>
        </tr>

    </table>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<p style="text-align:center;">貢獻度總計為100%，請自行核算。</p>
<p style="text-align:center;">完成日期：115年01月06日</p>
""", unsafe_allow_html=True)

st.markdown("""
<h3>說明</h3>
<p>
本人在此聲明，本設計作業皆由本人獨立完成，並無其他第三者參與作業之進行；
若有抄襲或其他違反正常教學之行為，自願接受該次成績以零分計。
同時本人亦同意在上述表格中所記載之作業貢獻度，
並以此計算本次個人作業成績。
</p>
<p><b>成員簽名：</b></p>
""", unsafe_allow_html=True)

st.image("image/1.png")
