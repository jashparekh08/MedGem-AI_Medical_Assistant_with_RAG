import os # This line is necessary to access environment variables
import streamlit as st # Streamlit is used to create the web interface for the chatbot

from langchain_huggingface import HuggingFaceEmbeddings # Used for generating embeddings
from langchain.chains import RetrievalQA # Used to create a question-answering chain

from langchain_community.vectorstores import FAISS # Used for storing and retrieving vector embeddings
from langchain_core.prompts import PromptTemplate # Used to create custom prompts
from langchain_huggingface import HuggingFaceEndpoint # Used to load the LLM from HuggingFace

from langchain_groq import ChatGroq  # Used to load the LLM from Groq


# ## Uncomment the following files if you're not using pipenv as your virtual environment manager
# #from dotenv import load_dotenv, find_dotenv
# #load_dotenv(find_dotenv())


DB_FAISS_PATH="vectorstore/db_faiss" # Path to the FAISS vector store database
@st.cache_resource  # Cache the vector store to avoid reloading it every time the app runs
def get_vectorstore(): # Function to get the vector store
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):  # Function to set a custom prompt template
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id, HF_TOKEN): # Function to load the LLM from HuggingFace
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"}
    )
    return llm


def main(): 
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state: # Used for save Chat history
        st.session_state.messages = []

    for message in st.session_state.messages: # To show messages saved in the above lists
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt) # 
        st.session_state.messages.append({'role':'user', 'content': prompt})


        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """
        
        # HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3" # PAID
        # HF_TOKEN=os.environ.get("HF_TOKEN")  

        #TODO: Create a Groq API key and add it to .env file
        
        try: 
            vectorstore=get_vectorstore() # Load the vector store
            if vectorstore is None: # Check if the vector store is loaded
                st.error("Failed to load the vector store")

            qa_chain = RetrievalQA.from_chain_type( # Create the QA chain
                llm=ChatGroq( # Load the LLM from Groq
                    model_name="meta-llama/llama-4-maverick-17b-128e-instruct",  # free, fast Groq-hosted model # Use a different model if you have a Groq API key
                    temperature=0.0,
                    groq_api_key=os.environ["GROQ_API_KEY"], # No api key needed for Groq-hosted models so GROQ_API_KEY is not required
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response=qa_chain.invoke({'query':prompt}) # Invoke the QA chain with the user's prompt

            result=response["result"] # Get the result from the response
            source_documents=response["source_documents"] # Get the source documents from the response

            # result_to_show=result+"\nSource Docs:\n"+str(source_documents) # Format the result to show the source documents
            result_to_show=result # Show only the result without source documents

            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()