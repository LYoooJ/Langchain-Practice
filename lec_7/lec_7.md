## Text Embeddings
Text Embeddings를 통해 텍스트를 숫자로 변환하여 문장 간의 유사성을 비교할 수 있으며, RAG 시스템에서 각 chunk를 임베딩 벡터로 변환하여 사용자의 질문과 가장 유사한 문장을 찾는데 활용된다. 대용량의 말뭉치로 사전학습된 Embedding model을 활용하여 Embedding을 수행할 수 있다.

- <strong>유료 임베딩 모델</strong></br>
OpenAI에서 제공하는 ada 모델 등이 있으며, 한국어를 포함해 다양한 언어에 대한 임베딩을 제공하고 GPU 없이도 빠른 임베딩이 가능하지만, API 통신을 이용하기 때문에 보안 문제가 있을 수 있고 비용이 든다는 단점이 있다.</br></br>
- <strong>로컬 임베딩 모델</strong></br>
HuggingFace의 모델들이 대표적이며, 무료이고 오픈소스 모델을 사용하기 때문에 보안이 우수하지만, 모델마다 지원되는 언어가 다르고 GPU가 없을 시 임베딩이 느리다는 단점이 있다.

## 실습
### OpenAIEmbeddings - ada-002
```python
from langchain_openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = api_key

embedding_model = OpenAIEmbeddings(openai_api_key=api_key)

embeddings = embedding_model.embed_documents(
    [
        "안녕하세요",
        "제 이름은 홍길동입니다.",
        "이름이 무엇인가요?",
        "랭체인은 유용합니다.",
        "Hello World!"
    ]
)

print(len(embeddings), len(embeddings[0]))
```
실행 결과
```
5 1536
```
입력한 문장마다 임베딩 벡터가 생성되어 embeddings의 크기는 5이고, 각 임베딩 벡터의 크기가 1536임을 확인할 수 있다.

```python
embedded_query_q = embedding_model.embed_query("이 대화에서 언급된 이름은 무엇입니까?")
embedded_query_a = embedding_model.embed_query("이 대화에서 언급된 이름은 홍길동입니다.")

print(len(embedded_query_q), len(embedded_query_a))
```
실행 결과
```
1536 1536
```
`embed_query()`를 통해 한 문장을 임베딩할 수 있으며, 마찬가지로 1536차원으로 임베딩되었음을 확인할 수 있다.

- 코사인 유사도로 문장 유사도 측정
```python
from langchain_openai import OpenAIEmbeddings
import os
from numpy import dot
from numpy.linalg import norm
import numpy as np

os.environ["OPENAI_API_KEY"] = api_key

embedding_model = OpenAIEmbeddings(openai_api_key=api_key)

embeddings = embedding_model.embed_documents(
    [
        "안녕하세요",
        "제 이름은 홍길동입니다.",
        "이름이 무엇인가요?",
        "랭체인은 유용합니다.",
        "Hello World!"
    ]
)

embedded_query_q = embedding_model.embed_query("이 대화에서 언급된 이름은 무엇입니까?")
embedded_query_a = embedding_model.embed_query("이 대화에서 언급된 이름은 홍길동입니다.")

def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

print(cos_sim(embedded_query_q, embedded_query_a))
print(cos_sim(embedded_query_q, embeddings[1]))
print(cos_sim(embedded_query_q, embeddings[3]))
```
실행 결과
```
0.9018785967795147
0.8503907115290837
0.7762114943154598
```
`이 대화에서 언급된 이름은 무엇입니까?`라는 질문에 대해 `이 대화에서 언급된 이름은 홍길동입니다.`, `제 이름은 홍길동입니다`, `랭체인은 유용합니다.`라는 문장이 각각 0.9018785967795147, 0.8503907115290837, 0.7762114943154598의 코사인 유사도를 가짐을 확인할 수 있다.

### HuggingFace Embedding

```python
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-small-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

hf = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

```
HuggingFace의 `BAAI/bge-small-en` 모델을 활용하고, `encoding_kwargs`의 `normalize_embeddings`를 True로 설정하여 임베딩을 정규화하여 임베딩 벡터간의 유사도 측정을 가능하게 한다.

```python
embeddings = hf.embed_documents(
    [
        "today is monday",
        "weather is nice today",
        "what's the problem?",
        "langchain is useful",
        "Hello World!",
        "my name is morris"
    ]
)

BGE_query_q = hf.embed_query("Hello? who is this?")
BGE_query_a = hf.embed_query("hi this is harrison.")

print(cos_sim(BGE_query_q, BGE_query_a))
print(cos_sim(BGE_query_q, embeddings[1]))
print(cos_sim(BGE_query_q, embeddings[5]))
```
실행 결과
```
0.8477948265615182
0.7411563327840657
0.7804736536538018
```
`Hello? who is this?`의 질문에 대해 `hi this is harrison.`, `weather is nice today`, `my name is morris`의 문장이 각각 0.8477948265615182, 0.7411563327840657, 0.7804736536538018의 코사인 유사도를 가지는 것을 확인할 수 있다. 

```python
sentences = [
    "안녕하세요",
    "제 이름은 홍길동입니다.",
    "이름이 무엇인가요?",
    "랭체인은 유용합니다.",
    "홍길동 아버지의 이름은 홍상작입니다."
]
ko_embeddings = hf.embed_documents(sentences)

BGE_query_q_2 = hf.embed_query("홍길동은 아버지를 아버지라 부르지 못하였습니다. 홍길동 아버지의 이름은 무엇입니까?")
BGE_query_a_2 = hf.embed_query("홍길동의 아버지는 엄했습니다.")

print("홍길동은 아버지를 아버지라 부르지 못하였습니다. 홍길동 아버지의 이름은 무엇입니까?\n", "-"*100)
print("홍길동의 아버지는 엄했습니다.\t문장유사도: ", cos_sim(BGE_query_q_2, BGE_query_a_2))
print(sentences[1] + "\t문장유사도: ", cos_sim(BGE_query_q_2, ko_embeddings[1]))
print(sentences[2] + "\t문장유사도: ", cos_sim(BGE_query_q_2, ko_embeddings[2]))
print(sentences[3] + "\t문장유사도: ", cos_sim(BGE_query_q_2, ko_embeddings[3]))
print(sentences[4] + "\t문장유사도: ", cos_sim(BGE_query_q_2, ko_embeddings[4]))
```
실행 결과
```
홍길동은 아버지를 아버지라 부르지 못하였습니다. 홍길동 아버지의 이름은 무엇입니까?
 ----------------------------------------------------------------------------------------------------
홍길동의 아버지는 엄했습니다.   문장유사도:  0.9483175290337432
제 이름은 홍길동입니다. 문장유사도:  0.8813273996994753
이름이 무엇인가요?      문장유사도:  0.8299129565546051
랭체인은 유용합니다.    문장유사도:  0.848813710030114
홍길동 아버지의 이름은 홍상작입니다.    문장유사도:  0.9226033401232842
```

해당 모델은 영어에 적합한 모델로, 실제로 한국어 문장의 임베딩에 대해 사용하는 경우 `홍길동은 아버지를 아버지라 부르지 못하였습니다. 홍길동 아버지의 이름은 무엇입니까?`라는 질문에 대해 `홍길동 아버지의 이름은 홍상작입니다.`보다 `홍길동의 아버지는 엄했습니다.`의 유사도가 더 높게 나오는 것을 확인할 수 있다.

### 한국어 사전학습 임베딩 모델 - ko-sbert-nli
```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from numpy import dot
from numpy.linalg import norm

def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

model_name = "jhgan/ko-sbert-nli"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

sentences = [
    "안녕하세요",
    "제 이름은 홍길동입니다.",
    "이름이 무엇인가요?",
    "랭체인은 유용합니다.",
    "홍길동 아버지의 이름은 홍상작입니다."
]
ko_embeddings = hf.embed_documents(sentences)

BGE_query_q_2 = hf.embed_query("홍길동은 아버지를 아버지라 부르지 못하였습니다. 홍길동 아버지의 이름은 무엇입니까?")
BGE_query_a_2 = hf.embed_query("홍길동의 아버지는 엄했습니다.")

print("홍길동은 아버지를 아버지라 부르지 못하였습니다. 홍길동 아버지의 이름은 무엇입니까?\n", "-"*100)
print("홍길동의 아버지는 엄했습니다.\t문장유사도: ", cos_sim(BGE_query_q_2, BGE_query_a_2))
print(sentences[1] + "\t문장유사도: ", cos_sim(BGE_query_q_2, ko_embeddings[1]))
print(sentences[2] + "\t문장유사도: ", cos_sim(BGE_query_q_2, ko_embeddings[2]))
print(sentences[3] + "\t문장유사도: ", cos_sim(BGE_query_q_2, ko_embeddings[3]))
print(sentences[4] + "\t문장유사도: ", cos_sim(BGE_query_q_2, ko_embeddings[4]))
```
실행 결과
```
홍길동은 아버지를 아버지라 부르지 못하였습니다. 홍길동 아버지의 이름은 무엇입니까?
 ----------------------------------------------------------------------------------------------------
홍길동의 아버지는 엄했습니다.   문장유사도:  0.4685280622619678
제 이름은 홍길동입니다. 문장유사도:  0.539185478002137
이름이 무엇인가요?      문장유사도:  0.5431911649464531
랭체인은 유용합니다.    문장유사도:  0.02957754816894958
홍길동 아버지의 이름은 홍상작입니다.    문장유사도:  0.6065364168067358
```
HuggingFace의 한국어 말뭉치를 통해 사전학습된 임베딩모델인 `ko-sbert-nli`를 활용하여 위의 예제를 다시 실행해보면 `홍길동은 아버지를 아버지라 부르지 못하였습니다. 홍길동 아버지의 이름은 무엇입니까?`라는 질문에 대해 `홍길동 아버지의 이름은 홍상작입니다.` 문장의 코사인 유사도가 가장 높게 나오는 것을 확인할 수 있다. 또한 `랭체인은 유용합니다.`라는 질문과 전혀 관계 없는 문장의 코사인 유사도가 아주 낮음을 볼 수 있다.