import streamlit as st
import pickle 
from streamlit_extras.add_vertical_space import add_vertical_space
# import numpy as np
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
# from langchain.llms import vertexai
# VertexAIEmbeddings
# from google.cloud import aiplatform
# from langchain.llms import VertexAI
# import google.generativeai
# from langchain.embeddings import GooglePalmEmbeddings
# from langchain import HuggingFaceHub
# from langchain.llms import google_palm
from langchain.chains.question_answering import load_qa_chain
import os


#Background
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
st.title('ChatPDF 💭')#title
#SIDEBAR
with st.sidebar:
    set_background()
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
#MAIN FUNCTION
def main():

    pdf=st.file_uploader('File uploader',type='pdf')
    if pdf is not None :
        pdf_reader=PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        # st.write(text)
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks=text_splitter.split_text(text=text)
        # st.write(chunks)

        #EMBEDDINGS

        storename=pdf.name[:-4]
        if os.path.exists(f"{storename}.pkl"):
            with open(f"{storename}.pkl",'rb') as f:
                Vectorstore=pickle.load(f)

        else:
            embeddings = HuggingFaceEmbeddings()
            # embeddings = GooglePalmEmbeddings()
            Vectorstore=FAISS.from_texts(chunks,embedding=embeddings)
            with open(f"{storename}.pkl",'wb') as f:
                pickle.dump(Vectorstore,f)
            # st.write(embeddings)
        query=st.text_input("Enter question")
        # st.write(query)
        if query:
            docs=Vectorstore.similarity_search(query=query,k=3) 
            st.write(docs)
        # google_api_key=os.getenv('GOOGLE_API_KEY')
        # llm=HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":1e-10})
        # chain=load_qa_chain(llm=llm,chain_type="stuff")
        # response= chain.run(input_documents=docs, question= query)
        # st.write(response)

        # llm = google_palm(google_api_key=google_api_key)
        # llm.temperature = 0.1
        # llm =ctransformers(model="marella/gpt-2-ggml")
        # llm=vertexai()
    
if __name__ == "__main__":
    main()
