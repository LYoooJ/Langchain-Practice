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