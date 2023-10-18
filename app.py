import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import numpy as np
from PyPDF2 import PdfReader

def set_background():
    
    page_bg_img = '''
            <style>
            .stApp {
    background: rgb(2,0,36);
    background: linear-gradient(170deg, rgba(2,0,36,1) 9%, rgba(121,9,71,1) 45%, rgba(0,212,255,1) 100%);
            }
            </style>
            ''' 
    st.markdown(page_bg_img, unsafe_allow_html=True)
st.title('ChatPDF 💭')
with st.sidebar:
    set_background()
    # page_bg_img = f"""<style>
    #     [data-testid="stSidebar"].main {{
    # background: rgb(2,0,36);
    # background: linear-gradient(170deg, rgba(2,0,36,1) 9%, rgba(121,9,71,1) 45%, rgba(0,212,255,1) 100%);
    # background-position: center; 
    # background-repeat: no-repeat;
    # background-attachment: fixed;
    # }} 
    # </style>"""



    st.markdown('''     
    ## About 🙋🏻‍♂️
    This is a ChatPDF Clone
    Where you can upload a PDF and ask questions based on its contents built using:
                
    -Streamlit
                
    -Python
                
    -LangChain
                
     ''') 
    add_vertical_space(15)
    st.write('Made with ❤️ by  [Abhishek S](https://github.com/iamA333)')

def main():

    pdf=st.file_uploader('File uploader',type='pdf')
    if pdf is not None :
        pdf_reader=PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        st.write(text)

    # st.write(d)

    
if __name__ == "__main__":
    main()