
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
import tiktoken

### load tokenizer ###
tokenizer = tiktoken.get_encoding("cl100k_base")

def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

### Load & Split PDF ###
loader = PyPDFLoader("2023_curriculum.pdf")
pages = loader.load_and_split()

### Chunking ###
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, length_function=tiktoken_len)
texts = text_splitter.split_documents(pages)

### embedding model ###
model_name = "jhgan/ko-sbert-nli"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

### Load to Chroma ###
docsearch = Chroma.from_documents(texts, hf)

### save to disk ###
persist_directory="./curriculum_db"
docsearch = Chroma.from_documents(texts, hf, persist_directory=persist_directory)