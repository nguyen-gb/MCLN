import streamlit as st

with open("designing.css") as source_des:
    st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)

st.markdown("# Machine Learning :heartbeat:")
st.write("Giảng viên: Trần Tiến Đức")
st.write("Nhóm gồm 3 sinh viên: ")
st.write("+ Trần Văn Nguyên - 20110685")
st.write("+ Nguyễn Ngọc Hải - 20110639")
st.write("+ Phan Phục Nghi - 20110681")
container = st.container(); 

container.image('images/machineLearning.png')