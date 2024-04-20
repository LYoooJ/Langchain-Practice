import os
import logging
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import AIMessage, HumanMessage, SystemMessage

### logger ###
logging.basicConfig(filename="lec_3.log", filemode="w", level=logging.INFO)
logger = logging.getLogger()

### api.txt에서 API key 읽어오기 ###
f = open("api.txt", "r")
api_key = f.readline()
f.close()

### api.txt에서 읽어온 키를 넣어주기! ###
os.environ["OPENAI_API_KEY"] = api_key

### API를 통해 ChatGPT와 대화하기 ###
model_name = "gpt-3.5-turbo"
logger.info(f"model_name: {model_name}")
chatgpt = ChatOpenAI(model_name=model_name)
answer = chatgpt.invoke("why python is the most popular language? answer in Korean")
logger.info(f"answer: {answer}")

### Temperature 매개변수 조절 ###
# temperature = 0
logger.info(f"model_name: {model_name}, temperature: {0}")
chatgpt_0 = ChatOpenAI(model_name=model_name, temperature=0)
answer = chatgpt_0.invoke("why python is the most popular language? answer in Korean")
logger.info(f"answer: {answer}")

logger.info(f"model_name: {model_name}, temperature: {0}")
chatgpt_0 = ChatOpenAI(model_name=model_name, temperature=0)
answer = chatgpt_0.invoke("why python is the most popular language? answer in Korean")
logger.info(f"answer: {answer}")

# temperature = 1
logger.info(f"model_name: {model_name}, temperature: {1}")
chatgpt_0 = ChatOpenAI(model_name=model_name, temperature=1)
answer = chatgpt_0.invoke("why python is the most popular language? answer in Korean")
logger.info(f"answer: {answer}")

logger.info(f"model_name: {model_name}, temperature: {1}")
chatgpt_0 = ChatOpenAI(model_name=model_name, temperature=1)
answer = chatgpt_0.invoke("why python is the most popular language? answer in Korean")
logger.info(f"answer: {answer}")

### Streaming ###
logger.info(f"model_name: {model_name}, streaming={True}, temperature={1}")
chatgpt = ChatOpenAI(model_name=model_name, streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=1)
answer = chatgpt.invoke("why python is the most popular language? answer in Korean")
logger.info(f"answer: {answer}")

### System Message ###
logger.info(f"model_name: {model_name}, temperature={0}")
chatgpt = ChatOpenAI(model_name=model_name, temperature=0)

system_message = "You are a helpful assistant that translates English to Korean."
human_message = "I love langchain"
logger.info(f"System Message: {system_message}")
logger.info(f"Human Message: {human_message}")
messages = [
    SystemMessage(
        content=system_message
    ),
    HumanMessage(
        content=human_message
    )
]

response_langchain = chatgpt.invoke(messages)
logger.info(f"AIMessage: {response_langchain.content}")

### Practice ###
logger.info(f"model_name: {model_name}, streaming={True}, temperature={1}")
chatgpt = ChatOpenAI(model_name=model_name, streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=1)
system_message = "당신은 공부 계획을 세워주는 스터디 플래너 머신입니다. 사용자의 공부 주제를 입력 받으면, 이를 학습하기 위한 공부 계획을 작성합니다."
human_message = "Large Language Model에 대해서 공부하고 싶어요."
answer = chatgpt.invoke(
    [
        SystemMessage(
            content=system_message
        ),
        HumanMessage(
            content=human_message
        )
    ]
)

logger.info(f"AIMessage: {answer.content}")