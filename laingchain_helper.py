from langchain.llms import GooglePalm
from dotenv import load_dotenv
from langchain.document_loaders.csv_loader import CSVLoader
load_dotenv()
import os
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

llm=GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"],temprature=0.2)


embeddings = HuggingFaceInstructEmbeddings()

vector_db_file_path="faiss_index"

def create_vector_db():
    loader=CSVLoader(file_path='E:\material\GenerativeAI\langchain\qna_chatbot_for_edtech\codebasics_faqs.csv',source_column='prompt')
    data=loader.load()
    vectordb=FAISS.from_documents(documents=data,embedding=embeddings)
    vectordb.save_local(vector_db_file_path)


def get_qa_chain():
    vector_db=FAISS.load_local(vector_db_file_path,embeddings)
    retrviever=vector_db.as_retriever(score_threshold=0.7)



    prompt_template="""Given the following context and a question generate  an answer based on this content  only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""


    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}




    chain=RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",
                                      retriever=retrviever,
                                      input_key="querry",
                                      return_source_documents=True,
                                      chain_type_kwargs={"prompt":PROMPT})
    return chain

if __name__=="__main__":
    chain=get_qa_chain()
    print(chain("do you have emi option"))