import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import logging

### logger ###
logging.basicConfig(filename="lec_8.log", filemode="w", level=logging.INFO)
logger = logging.getLogger()

tokenizer = tiktoken.get_encoding("cl100k_base")

def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

### load the document and split it into chunks ###
pdf = "Simple Linear Regression.pdf"
logger.info(f"filename: {pdf}")
loader = PyPDFLoader(pdf)
pages = loader.load_and_split()

### split it into chunks ###
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,
    length_function=tiktoken_len
)
docs = text_splitter.split_documents(pages)

### embedding model ###
model_name = "BAAI/bge-small-en"
logger.info(f"model name: {model_name}")
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

### load it into Chroma ###
logger.info("Chroma")
db = Chroma.from_documents(docs, hf)

### query ###
query = "What is the hypothesis test of β0?"
logger.info(f"query: {query}")
docs = db.similarity_search(query)

### result ###
result = docs[0]
logger.info(f"similarity search result: {result}")

### save to disk ###
persist_directory="./chroma_db"
logger.info(f"Save the disk: {persist_directory}")
db2 = Chroma.from_documents(docs, hf, persist_directory=persist_directory)

### load from disk ###
db3 = Chroma(persist_directory=persist_directory, embedding_function=hf)
logger.info(f"Load from dist: {persist_directory}")
docs = db3.similarity_search(query)

### result ###
result = docs[0]
logger.info(f"similarity search result: {result}")

### FAISS ###
logger.info("FAISS")
db = FAISS.from_documents(docs, hf)
docs = db.similarity_search(query)

### Save to disk ###
file_name = "faiss_index"
logger.info(f"save to disk: {file_name}")
db.save_local(file_name)

### load from disk ###
logger.info(f"load from disk: {file_name}")
new_db = FAISS.load_local(file_name, hf)
docs = db3.similarity_search(query)

### result ###
result = docs[0]
logger.info(f"similarity search result: {result}")