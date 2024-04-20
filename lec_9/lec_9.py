import os
import openai
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import tiktoken
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import logging

### logger ###
logging.basicConfig(filename="lec_9.log", filemode="w", level=logging.INFO)
logger = logging.getLogger()

### api.txt에서 API key 읽어오기 ###
f = open("api.txt", "r")
api_key = f.readline()
f.close()

### api.txt에서 읽어온 키를 넣어주기! ###
os.environ["OPENAI_API_KEY"] = api_key
os.environ["TOKENIZERS_PARALLELISM"] = "false"

### load tokenizer ###
logger.info("tokenizer: cl100k_base")
tokenizer = tiktoken.get_encoding("cl100k_base")

def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

### embedding model ###
model_name = "jhgan/ko-sbert-nli"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

logger.info(f"Embedding: {model_name}")
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

### load from disk ###
persist_directory="./curriculum_db"
docsearch = Chroma(persist_directory=persist_directory, embedding_function=hf)

### Load ChatGPT model###
logger.info("gpt-3.5-turbo")
chatgpt = ChatOpenAI(
    model_name="gpt-3.5-turbo", 
    streaming=True, 
    callbacks=[StreamingStdOutCallbackHandler()], 
    temperature=1
)

qa = RetrievalQA.from_chain_type(llm=chatgpt,
                                 chain_type="stuff",
                                 retriever=docsearch.as_retriever(
                                     search_type="mmr",
                                     search_kwargs={'k': 3, 'fetch_k': 7}),
                                 return_source_documents=True
)

query = "논리적인 사고를 기를 수 있는 수업에는 뭐가 있어? 그 이유도 알려줘."
query2 = "생명과학과에 대해서 설명해줘."

result = qa.invoke(query)
result2 = qa.invoke(query2)

logger.info(f"query: {query}\n answer: {result}")
logger.info(f"query: {query2}\n answer: {result2}")