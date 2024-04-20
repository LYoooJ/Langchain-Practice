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
logging.basicConfig(filename="postech.log", filemode="w", level=logging.INFO)
logger = logging.getLogger()

tokenizer = tiktoken.get_encoding("cl100k_base")

def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

### load the document and split it into chunks ###
pdf = "2023_curriculum.pdf"
logger.info(f"filename: {pdf}")
loader = PyPDFLoader(pdf)
pages = loader.load_and_split()

### split it into chunks ###
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=tiktoken_len
)
docs = text_splitter.split_documents(pages)

### embedding model ###
model_name = "jhgan/ko-sbert-nli"
logger.info(f"model name: {model_name}")
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# ### load it into Chroma ###
# logger.info("Chroma")
# db = Chroma.from_documents(docs, hf)

# ### save to disk ###
# persist_directory="./curriculum_db"
# logger.info(f"Save the disk: {persist_directory}")
# db2 = Chroma.from_documents(docs, hf, persist_directory=persist_directory)

### load from disk ###
persist_directory="./curriculum_db"
db = Chroma(persist_directory=persist_directory, embedding_function=hf)
logger.info(f"Load from disk: {persist_directory}")

### query ###
query = "창의적인 사고가 중요한 수업에는 뭐가 있어?"
logger.info(f"query: {query}")
docs = db.similarity_search_with_relevance_scores(query, k=3)

### result ###
print(docs)
result = docs[0]
logger.info(f"similarity search result: {result}")

### query ###
query = "논리적인 사고 능력을 기를 수 있는 수업에는 뭐가 있어?"
logger.info(f"query: {query}")
docs = db.similarity_search_with_relevance_scores(query, k=3)

### result ###
print(docs)
result = docs[0]
logger.info(f"similarity search result: {result}")