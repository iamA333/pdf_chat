import streamlit as st
import pickle 
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.llms import CTransformers
from langchain.chains import ConversationChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationChain
import os
from transformers import pipeline

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

        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks=text_splitter.split_text(text=text)
    

        #EMBEDDINGS

        storename=pdf.name[:-4]
        # add LLama
        if os.path.exists(f"{storename}.pkl"):
            with open(f"{storename}.pkl",'rb') as f:
                Vectorstore=pickle.load(f)

        else:
            embeddings = HuggingFaceEmbeddings()
           
            Vectorstore=FAISS.from_texts(chunks,embedding=embeddings)
            with open(f"{storename}.pkl",'wb') as f:
                pickle.dump(Vectorstore,f)
         
        query=st.text_input("Ask a question")
       
        if query:
            # if (query==1):
            # summarizer = pipeline("summarization", model="Azma-AI/bart-large-text-summarizer")
            # st.write(summarizer(text))
                
         
            docs=Vectorstore.similarity_search(query=query,k=3)
            llm = CTransformers(model="marella/gpt-2-ggml")
            chain=load_qa_chain(llm=llm,chain_type="stuff")          


            response= chain.run(input_documents=docs, question= query)
            st.write(response) 



    
if __name__ == "__main__":
    main()
