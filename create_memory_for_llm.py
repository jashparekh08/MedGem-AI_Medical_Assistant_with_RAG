from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader # used for loading documents
from langchain.text_splitter import RecursiveCharacterTextSplitter # used for splitting text into manageable chunks
from langchain_huggingface import HuggingFaceEmbeddings # Used for generating embeddings
from langchain_community.vectorstores import FAISS


# Step 1: Load raw PDF(s)

# Load the raw PDF(s) from the specified directory
DATA_PATH="data/" # Directory containing the PDF files

def load_pdf_files(data): # This functions Loads all .pdf files from the specified directory
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    
    documents=loader.load() # This loads the PDF files into a list of documents
    return documents


documents=load_pdf_files(data=DATA_PATH)
# print("Length of PDF pages: ", len(documents))


# Step 2: Create Chunks

def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=50) # This initializes a text splitter that will split the text into chunks of 500 characters with an overlap of 50 characters
    # chunk_size is the maximum size of each chunk
    # chunk_overlap is the number of characters that overlap between consecutive chunks
    # chunk_overlap is used to ensure that the chunks are not completely independent, which can help maintain context

    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=documents)
# print("Length of Text Chunks: ", len(text_chunks))



# Step 3: Create Vector Embeddings 

def get_embedding_model(): # This function initializes the HuggingFaceEmbeddings model
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # this model is used to generate embeddings for the text chunks
    # "sentence-transformers/all-MiniLM-L6-v2" is a pre-trained model that is efficient and effective for generating embeddings
    # all-MiniLM-L6-v2 is a sentence transformer model: It maps sentences to a 384-dimensional vector space, where semantically similar sentences are closer together in this space. and it is used from semantic search, clustering, and other NLP tasks.
    return embedding_model


embedding_model=get_embedding_model()



# Step 4: Store embeddings in FAISS

DB_FAISS_PATH="vectorstore/db_faiss" # Path to save the FAISS database
db=FAISS.from_documents(text_chunks, embedding_model) # In db, we create a FAISS vector store from the text chunks and the embedding model
# create embeddings from the text chunks using the embedding model and store them in a FAISS vector store

db.save_local(DB_FAISS_PATH)