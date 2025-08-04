import os # Used to access environment variables

from langchain_huggingface import HuggingFaceEndpoint # Used to load the LLM from HuggingFace
from langchain_core.prompts import PromptTemplate # Used to create custom prompts
from langchain.chains import RetrievalQA # Used to create a question-answering chain
from langchain_huggingface import HuggingFaceEmbeddings # Used for generating embeddings
from langchain_community.vectorstores import FAISS # Used for storing and retrieving vector embeddings


# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN=os.environ.get("HF_TOKEN") # Load HuggingFace token from environment variable
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3" # The HuggingFace repository ID for the LLM

def load_llm(huggingface_repo_id): # This function initializes the LLM from HuggingFace
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id, # The HuggingFace repository ID for the LLM
        temperature=0.5,            # Controls the randomness of the model's output
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"} # Additional model parameters, including the token for authentication and the maximum length of the generated text
    )
    return llm



# Step 2: Connect LLM with FAISS and Create chain

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
""" # This is a custom prompt template that instructs the LLM on how to respond to user queries based on the context provided.

def set_custom_prompt(custom_prompt_template): # This function creates a prompt template using the custom prompt template defined above
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"]) # Create a PromptTemplate instance
    return prompt

# Load Database
DB_FAISS_PATH="vectorstore/db_faiss" # Path to the FAISS database
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # Initialize the embedding model used for generating embeddings
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True) # Load the FAISS vector store from the specified path, allowing dangerous deserialization for compatibility with older versions of FAISS.

# Create QA chain
qa_chain=RetrievalQA.from_chain_type( # Create a question-answering chain using the loaded LLM and the FAISS vector store
    llm=load_llm(HUGGINGFACE_REPO_ID), # Load the LLM
    chain_type="stuff",                # Use the "stuff" chain type, which is suitable for simple question-answering tasks
    retriever=db.as_retriever(search_kwargs={'k':3}), # Use the FAISS vector store as a retriever, retrieving the top 3 relevant documents for each query
    return_source_documents=True,      # Return the source documents along with the answer
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)} # Set the custom prompt template for the chain
)

# Now invoke with a single query
user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])


