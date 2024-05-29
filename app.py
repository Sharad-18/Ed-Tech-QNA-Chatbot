import streamlit as st
from laingchain_helper import get_qa_chain,create_vector_db
st.title("Ed Tech QA Chatbot")

btn=st.button("Create a knowledgebase")
if btn:
    create_vector_db()
question=st.text_input("Question: ")

if question:
    chain=get_qa_chain()
    response=chain(question)
    
    st.header("Answer")
    st.write(response["result"])
    
