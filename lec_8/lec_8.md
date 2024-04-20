## Vector Store
Vector Store에는 여러 종류가 존재하며, `Pure vector database`는 데이터베이스에 벡터 형태의 데이터만 저장할 수 있고 DB가 가지는 용이한 기능들을 가지고 있으며 대표적으로 `Pinecone`, `qdrant`, `chroma`가 활용되고, `Vector libraries`는 벡터 유사도를 계산하는데 특화되어 있으며 대표적으로 `FAISS`가 활용된다.

## 실습
### Chroma
Chroma는 오픈소스 벡터 저장소로, `from_documents()`함수에 텍스트와 임베딩 함수를 지정해 호출하면, 지정된 임베딩 함수로 텍스트를 임베딩 벡터로 변환하고 임시 DB로 생성한다. `similarity_search()`함수에 쿼리를 지정하여 호출하면 쿼리와 가장 유사도가 높은 벡터를 찾아 자연어 형태로 출력한다.

```python
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

tokenizer = tiktoken.get_encoding("cl100k_base")

def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

### load the document and split it into chunks ###
pdf = "title.pdf"
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
db = Chroma.from_documents(docs, hf)

### query ###
query = "Query"
docs = db.similarity_search(query)

### result ###
result = docs[0]
```
이때, 활용하고자 하는 문서를 임시 DB가 아니라 디스크에 저장하고 활용할 수도 있으며, 이를 위해서는 `persist()`함수를 통해 벡터 저장소를 로컬에 저장하여 필요할 때 다시 불러와 사용할 수 있다.
```python
### save to disk ###
persist_directory="./chroma_db"
db2 = Chroma.from_documents(docs, hf, persist_directory=persist_directory)

### load from disk ###
db3 = Chroma(persist_directory=persist_directory, embedding_function=hf)
docs = db3.similarity_search(query)
```
### FAISS 
FAISS는 Facebook AI 유사성 검색으로 고밀도 벡터의 효율적인 유사성 검색과 클러스터링을 위한 라이브러리이다. Chroma와 유사한 방식으로 다음과 같이 사용가능하다.
```python
### FAISS ###
logger.info("FAISS")
db = FAISS.from_documents(docs, hf)
docs = db.similarity_search(query)

### Save to disk ###
file_name = "faiss_index"
db.save_local(file_name)

### load from disk ###
new_db = FAISS.load_local(file_name, hf)
docs = db3.similarity_search(query)
```