# importing required module 
import re
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai

# Loading the enviroment variable which is set for the api key
load_dotenv()


def extract_pdf_text(document):
    """ Extract the text from the document uploaded by the user 
    Args: document
    return: text extracted from the document"""

    text=""
    for pdf in document:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def extract_text_chunks(text):
    """ This function breaks down the text into chunk for further processing for huge document"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """This function performs text embedding as model understand numeric values only and the  FAISS is used for the vector store local."""
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def conversation_chain():
    """ Creation of chain for the chat model with the proper prompt template to train the model on basis of user information and document uploaded
        Args: NONE
        Returns : chain 
        """
    prompt_template = """
    Please respond to the following question as completely and correctly as feasible given the situation. Include all important facts from the context. If the solution cannot be found in the context, state "Answer is not available in the context." Avoid making up facts.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    User Information:\n
    Name: \n{user_name}\n
    Email: \n{user_email}\n
    Phone Number: \n{phone_number}\n
    when user ask the question greet them with their name and email
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables=["user_name", "user_email", "phone_number", "context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_prompt(user_question):
    """This is the function  UI and prompt processing of the question provided by the user using conversation chain """
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = conversation_chain()
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""
    if "user_email" not in st.session_state:
        st.session_state.user_email = ""
    if "phone_number" not in st.session_state:
        st.session_state.phone_number = ""
    
    response = chain.invoke(
        {"input_documents":docs, "question": user_question,
        "user_name": st.session_state.user_name,
        "user_email": st.session_state.user_email,
        "phone_number": st.session_state.phone_number}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])


def email_validation(email):
    """ This function checks if the user input email is in correct format or not with the help of regular expression
        Args: 
        email : email entered by the user.
        Returns:
        Boolean value: True if the email is valid and false if the email is invalid

    """
    # regular expression pattern for email validation 
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
    if(re.fullmatch(regex, email)):
        return True
    else:
        return False
    

def main():
    """ Main UI of the chatbot app"""
    st.title("Chatbot Project")
    st.info("Chatbot app implement with the help of lanchain and gemini LLM")
    user_email = None
    with st.form (key='myform'):
        # user information input textfield 
        user_name = st.text_input("Enter your name")
        phone_number_input = st.text_input("Enter your Phone Number")
        user_email_input = st.text_input("Enter you Email ID")
        submit_button = st.form_submit_button("Submit")

    if submit_button:
    #email validation 
        if  email_validation(user_email_input):
            user_email = user_email_input
           

            st.write("details you have provided")
            st.write( user_name)
            st.write( user_email)
            st.write( phone_number_input)

            # set current session of the user 
            st.session_state.user_name = user_name
            st.session_state.user_email = user_email
            st.session_state.phone_number = phone_number_input
        else:
            st.error("Please check the format of your email")
    

    with st.form (key='myform2'):
       
        pdf_docs = st.file_uploader("Upload your PDF Files ", accept_multiple_files=True)
        submit_button = st.form_submit_button("Submit")
        if submit_button:
            with st.spinner("Processing please wait..."):
                #when the document is uploaded all this function is called for further text processing requried for getting the response to the user
                raw_text = extract_pdf_text(pdf_docs)
                text_chunks = extract_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    st.info("Let's chat ")
    user_question = st.text_input("Ask your doubt?")

    if user_question:
        #if user ask the question to the chatbot the prompt function is hit 
        user_prompt(user_question)



main()
