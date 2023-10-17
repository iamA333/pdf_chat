import streamlit as st
from streamlit_extras import add_vertical_space
import numpy as np

st.write('ChatPDF 💭')
with st.sidebar:

    st.markdown('''     
    ## About Me 🙋🏻‍♂️
    This is a ChatPDF Clone
    Where you can upload a PDF and ask questions based on its contents
                
    [Github](https://github.com/iamA333) ''') 
    

d=st.file_uploader('File uploader',type='pdf')
st.write(d)

with st.chat_message("assistant"):

    st.write("Hello 👋")
 
    # st.line_chart(np.random.randn(30, 3))
c=st.chat_input("Say something")
if c is not None:
    with st.chat_message("user"):
        st.write(c)
    if (len(c)!=0):
        with st.chat_message("assistant"):
            st.write("Trial ")