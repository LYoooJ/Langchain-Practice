## LangChain 라이브러리를 이용한 ChatGPT API 활용 실습


### OpenAI API Key
OpenAI의 인공지능 서비스를 사용하기 위해서는 API key가 필요하다. 발급 방법은 아래와 같다.</br>
- https://platform.openai.com/account/api-keys 에 접속하여 로그인하기
- `Create new secret key`를 클릭하여 API key 생성하기

```python
import os

os.environ["OPENAI_API_KEY"] = 'API_KEY'
```
위와 같이 발급받은 API key를 저장해준다.

### Python 실습
- 필요 라이브러리: langchain, openai
- https://platform.openai.com/docs/libraries/python-library 참고하기</br></br>

#### API를 이용해 Chatgpt와 대화하기
```python
from langchain_openai import ChatOpenAI

chatgpt = ChatOpenAI(model_name="gpt-3.5-turbo")
answer = chatgpt.invoke("why python is the most popular language? answer in Korean")
print(answer)
```
실행 결과
```
파이썬은 현재 가장 인기 있는 프로그래밍 언어 중 하나입니다. 이러한 인기의 이유는 다음과 같습니다:

1. 문법의 간결함: 파이썬은 읽기 쉽고 이해하기 쉬운 문법을 가지고 있습니다. 이는 프로그래밍을 처음 접하는 사람들에게 특히 도움이 됩니다.

2. 다양한 용도로 사용 가능: 파이썬은 웹 개발, 데이터 분석, 인공지능, 자연어 처리 등 다양한 분야에서 사용될 수 있습니다. 이는 파이썬을 많은 사람들이 선택하는 이유 중 하나입니다.

3. 라이브러리와 생태계: 파이썬은 수많은 라이브러리와 패키지를 가지고 있습니다. 이러한 라이브러리들은 개발자들이 작업을 더욱 쉽게 수행할 수 있도록 도와줍니다. 또한 파이썬 커뮤니티는 활발하며 지속적인 지원과 업데이트를 제공합니다.

4. 큰 개발자 커뮤니티: 파이썬은 수많은 개발자들이 사용하며 커뮤니티가 활발합니다. 이는 다른 개발자들과의 협업, 지원 및 정보 공유에 도움을 줍니다.

5. 쉬운 학습 곡선: 파이썬은 초보자들에게 적합한 언어입니다. 문법이 간결하고 읽기 쉬우며, 커뮤니티에서 제공하는 풍부한 자료와 튜토리얼이 있어 학습 곡선을 낮출 수 있습니다.

이러한 이유로 파이썬은 많은 개발자들에게 인기를 얻고 있으며, 더 많은 사람들이 파이썬을 선택하고 사용하고 있습니다.
```

#### 매개 변수 조절
- temperature: ChatGPT가 제공하는 답변의 일관성을 조절하는 매개 변수(0~2, 0이면 일관성 높다. 2이면 같은 질문에 대해서도 여러가지 답변 가능)
```python
### Temperature 매개변수 조절 ###
# temperature = 0
print("temperature: 0")
chatgpt_0 = ChatOpenAI(model_name=model_name, temperature=0)
answer = chatgpt_0.invoke("why python is the most popular language? answer in Korean")
print(answer)

chatgpt_0 = ChatOpenAI(model_name=model_name, temperature=0)
answer = chatgpt_0.invoke("why python is the most popular language? answer in Korean")
print(answer)

# temperature = 1
print("temperature: 1")
chatgpt_1 = ChatOpenAI(model_name=model_name, temperature=1)
answer = chatgpt_1.invoke("why python is the most popular language? answer in Korean")
print(answer)

chatgpt_1 = ChatOpenAI(model_name=model_name, temperature=1)
answer = chatgpt_1.invoke("why python is the most popular language? answer in Korean")
print(answer)
```
실행 결과
```
temperature: 0
content='파이썬은 다양한 이유로 가장 인기 있는 프로그래밍 언어입니다. \n\n첫째, 파이썬은 배우기 쉽고 읽기 쉬운 문법을 가지고 있습니다. 이는 초보자들이 프로그래밍을 빠르게 익힐 수 있도록 도와줍니다. 또한, 파이썬은 간결한 코드 작성을 가능하게 하여 개발자들이 생산성을 높일 수 있습니다.\n\n둘째, 파이썬은 다양한 용도로 사용될 수 있습니다. 데이터 분석, 인공지능, 웹 개발, 자동화 등 다양한 분야에서 활용할 수 있어서 개발자들에게 많은 선택지를 제공합니다. 또한, 파이썬은 다른 언어와의 통합이 용이하고, 다양한 라이브러리와 프레임워크를 제공하여 개발자들이 효율적으로 작업할 수 있습니다.\n\n셋째, 파이썬은 커뮤니티와 생태계가 발달되어 있습니다. 파이썬은 오픈 소스 프로젝트로 개발되어 다양한 개발자들이 참여하고 있으며, 이에 따라 다양한 문제 해결 방법과 지원이 제공됩니다. 또한, 파이썬은 많은 개발자들이 사용하고 있어서 정보를 공유하고 협업하기에 용이합니다.\n\n이러한 이유들로 인해 파이썬은 가장 인기 있는 프로그래밍 언어 중 하나입니다.'

content='파이썬은 다양한 이유로 가장 인기 있는 프로그래밍 언어입니다. \n\n첫째, 파이썬은 배우기 쉽고 읽기 쉬운 문법을 가지고 있습니다. 이는 초보자들이 프로그래밍을 빠르게 익힐 수 있도록 도와줍니다. 또한, 파이썬은 간결한 코드 작성을 가능하게 하여 개발자들이 생산성을 높일 수 있습니다.\n\n둘째, 파이썬은 다양한 용도로 사용될 수 있습니다. 데이터 분석, 인공지능, 웹 개발, 자동화 등 다양한 분야에서 활용할 수 있어서 개발자들에게 많은 선택지를 제공합니다. 또한, 파이썬은 다른 언어와의 통합이 용이하고, 다양한 라이브러리와 프레임워크를 제공하여 개발자들이 효율적으로 작업할 수 있습니다.\n\n셋째, 파이썬은 커뮤니티와 생태계가 발달되어 있습니다. 파이썬은 오픈 소스 프로젝트로 개발되어 다양한 개발자들이 참여하고 있으며, 이에 따라 다양한 문제 해결 방법과 지원이 제공됩니다. 또한, 파이썬은 많은 개발자들이 사용하고 있어서 정보를 공유하고 협업하기에 용이합니다.\n\n이러한 이유들로 인해 파이썬은 가장 인기 있는 프로그래밍 언어 중 하나입니다.'

temperature: 1
content='Python은 여러 가지 이유로 인해 가장 인기 있는 프로그래밍 언어 중 하나입니다. 이에 대한 몇 가지 이유는 다음과 같습니다:\n\n1. 읽기 쉽고 이해하기 쉽습니다: Python은 간결하고 직관적인 문법을 제공하여 기존 프로그래머 및 새로운 학습자에게 모두 적합합니다. 다른 프로그래밍 언어보다 코드 가독성이 높아서 협업과 유지 보수가 용이합니다.\n\n2. 다양한 용도로 사용 가능합니다: Python은 웹 개발, 데이터 분석, 인공 지능, 기계 학습, 자동화 등 다양한 영역에서 사용될 수 있습니다. 이 다양성은 Python을 다양한 프로젝트와 산업에 적합하게 만듭니다.\n\n3. 강력하며 대중적인 라이브러리와 프레임워크가 많습니다: Python은 많은 라이브러리와 프레임워크가 있어서 개발자들이 쉽게 프로젝트에 통합할 수 있습니다. 예를 들면, Django와 Flask와 같은 웹 프레임워크, NumPy와 Pandas와 같은 데이터 분석 라이브러리, TensorFlow와 PyTorch와 같은 인공 지능/기계 학습 라이브러리 등이 있습니다.\n\n4. 커뮤니티의 지원과 개방성: Python은 활발하고 친절한 커뮤니티로 알려져 있으며, 오픈 소스 프로젝트에 대한 지원이 매우 높습니다. 많은 파이썬 개발자가 개방적이고 지식 공유에 열려 있어서, 도움을 받을 수 있는 기회가 많습니다.\n\n5. 높은 생산성과 빠른 개발 속도: Python은 간단한 구문과 미리 작성된 코드를 사용하여 빠르게 프로토타입을 작성할 수 있습니다. 이는 개발자가 애플리케이션을 더 효율적으로 개발하고 배포할 수 있게 도와줍니다.\n\n이러한 이유들로 Python은 많은 개발자들에게 인기가 있으며, 더 많은 사람들이 배우고 사용하기 시작하고 있습니다.'

content='파이썬은 현재 가장 인기있는 프로그래밍 언어로 인정받고 있습니다. 파이썬이 인기있는 이유는 다양합니다.\n\n첫째, 파이썬은 배우기 쉽고 읽기 쉬운 문법을 가지고 있습니다. 비전공자나 초보자도 쉽게 프로그래밍을 배울 수 있어 입문자에게 많은 인기를 끌고 있습니다. 또한 파이썬은 읽기 쉽고 명확한 코드 작성에 초점을 두기 때문에 유지보수가 용이합니다.\n\n둘째, 파이썬은 많은 라이브러리와 프레임워크가 있어 다양한 목적으로 사용할 수 있습니다. 데이터 분석, 인공지능, 웹 개발 등 다양한 분야에 적용할 수 있는 활용성이 높습니다. 또한 파이썬은 다른 프로그래밍 언어와의 통합이 용이하여 개발자들이 각자 필요한 라이브러리를 만들고 공유하기 쉽습니다.\n\n셋째, 파이썬은 큰 개발자 커뮤니티를 가지고 있습니다. 오픈소스로 개발되어 있어 많은 개발자들이 컨트리뷰션을 할 수 있으며 문제 발생 시 커뮤니티의 도움을 받을 수 있습니다. 이러한 커뮤니티의 협력과 개방성은 파이썬의 인기를 높이는 요인 중 하나입니다.\n\n이와 같은 이유로 파이썬은 다양한 분야에서 인기를 얻고 있으며, 더욱더 많은 개발자들이 선택하고 학습하는 언어입니다.'
```
temperature가 0일 때는 거의 동일한 답변이 나온 반면, temperature가 1일 때는 답변에 차이가 있음을 살펴볼 수 있다.

#### Streaming
답변이 한 번에 나오는 것이 아니라, 타이핑하듯이 나오게 할 수 있다.
```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

model_name = "gpt-3.5-turbo"

chatgpt = ChatOpenAI(model_name=model_name, streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=1)
answer = chatgpt.invoke("why python is the most popular language? answer in Korean")
```
실행 결과
```
파이썬은 가장 인기 있는 프로그래밍 언어인 이유는 여러 가지가 있습니다.

첫째, 파이썬은 배우기 쉽고 읽기 쉽습니다. 문법이 간결하고 직관적이어서 기초 개념을 이해하기 쉽습니다. 이로 인해 초보자도 빠르게 프로그래밍을 시작할 수 있습니다.

둘째, 파이썬은 다양한 용도로 활용할 수 있습니다. 웹 개발, 데이터 분석, 인공 지능, 사물 인터넷 등 여러 분야에서 사용됩니다. 유연하게 확장 가능한 기능과 라이브러리를 제공하여 프로젝트를 쉽게 구현할 수 있습니다.

셋째, 파이썬은 개발 생산성이 높습니다. 작성된 코드의 길이가 짧고, 라이브러리와 모듈을 쉽게 사용할 수 있어 개발 시간을 단축시킬 수 있습니다. 또한 디버깅과 유지보수도 용이하여 개발자의 업무 효율을 높일 수 있습니다.

넷째, 파이썬은 커뮤니티와 문서화가 잘 되어 있습니다. 전 세계적으로 활발하게 사용되고 있어서 어려운 문제에 대한 솔루션을 쉽게 찾을 수 있습니다. 또한 파이썬 공식 문서와 온라인 자료가 풍부하여 개발자들이 공부하고 참고하기에 용이합니다.
```
`streaming=True`로 설정하였기 때문에 `print()`없이도 답변이 생성되는 동안 타이핑하듯이 출력된다.

#### System Message
ChatGPT는 대화에 특화된 LLM으로 SystemMessage와 HumanMessage라는 독특한 매개변수를 가진다.
System Message를 이용하여 ChatGPT에서 특정한 역할을 부여하고 대화의 맥락을 설정할 수 있고, HumanMessage를 이용하여 ChatGPT에게 대화 또는 요청을 위한 메세지를 보낼 수 있다.
```python
from langchain.schema import AIMessage, HumanMessage, SystemMessage

model_name = "gpt-3.5-turbo"

chatgpt = ChatOpenAI(model_name=model_name, temperature=0)

messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to Korean."
    ),
    HumanMessage(
        content="I love langchain"
    )
]

response_langchain = chatgpt.invoke(messages)
print(response_langchain.content)
```
실행 결과
```
저는 랭체인을 사랑합니다.
```
주어진 영어 문장을 한글로 번역하는 역할을 부여했기 때문에, `I love langchain`이라는 사용자의 메세지를 `저는 랭체인을 사랑합니다`로 번역한 결과를 얻을 수 있다.

#### 실습
```python
model_name = "gpt-3.5-turbo"

chatgpt = ChatOpenAI(model_name=model_name, streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=1)
answer = chatgpt.invoke(
    [
        SystemMessage(
            content="당신은 공부 계획을 세워주는 스터디 플래너 머신입니다. 사용자의 공부 주제를 입력 받으면, 이를 학습하기 위한 공부 계획을 작성합니다."
        ),
        HumanMessage(
            content="Large Language Model에 대해서 공부하고 싶어요."
        )
    ]
)
```
실행 결과
```
공부 주제: Large Language Model

공부 계획:
1. 개념 이해 및 배경 조사
   - Large Language Model의 개념과 역사에 대해 알아보세요.
   - 트랜스포머(Transformers) 아키텍처에 대한 이해를 바탕으로 Large Language Model의 작동 원리를 이해하세요.
   - 기존의 NLP 모델과 Large Language Model의 차이점을 파악하세요.

2. 핵심 모델 학습
   - 대표적인 Large Language Model인 GPT-3에 대해 자세히 공부하세요.
   - GPT-3의 구현 방식, 학습 데이터 및 모델 아키텍처에 대한 이해를 도모하세요.
   - GPT-3의 실제 응용 사례 및 성능 분석에 대해 알아보세요.

3. 관련 기술과 응용
   - Large Language Model의 활용 분야와 응용 사례를 조사하세요. (예: 자연어 생성, 기계 번역, 질의응답 시스템 등)
   - Large Language Model을 사용하여 실제 문제를 해결하는 관련 기술 및 방법을 조사하고, 학습해 보세요.
   - 관련 논문을 읽고 최신 연구 동향을 파악하는 데 집중하세요.

4. 프로젝트 구현
   - Large Language Model을 활용한 실제 프로젝트를 구현해 보세요.
   - 예를 들어, 텍스트 생성 모델, 기계 번역 시스템, 감성 분석 모델 등을 만들어 볼 수 있습니다.
   - 프로젝트를 진행하며 발생하는 문제를 해결하고, 모델의 성능을 향상시킬 수 있는 방법을 탐구하세요.

5. 계획 검토 및 수정
   - 학습 과정 중에 생긴 질문이나 어려움을 해결하기 위해 공부 방법을 수정하고 개선하세요.
   - 완료한 항목을 체크하고 다음 항목으로 진행하세요.
   - 계획을 주기적으로 리뷰하고 필요한 조정을 하며 공부를 수행하세요.

6. 학습 내용 정리 및 복습
   - 공부한 내용을 정리하여 필요 시 나중에 다시 참고할 수 있도록 정리해 보세요.
   - 정리한 내용을 사용하여 복습하고, 추가적인 학습을 위한 자료를 탐색하세요.

이러한 계획을 통해 Large Language Model에 대한 깊은 이해를 얻을 수 있을 것입니다. 계획을 세워 진행하면서 자신의 학습 상태를 체크하고 조정하며 공부 계획을 완료하는 데 도움이 되길 바랍니다.
```
## 오류
- langchain 사용을 위해 Python 3.8.1 이상의 버전이 필요하다고 하여, python 3.8.13 버전으로 가상환경을 새롭게 구축</br></br>

-
  ```
  LangChainDeprecationWarning: The function 'predict' was deprecated in LangChain 0.1.7 and will be removed in 0.2.0. Use invoke instead.
  ```
  경고 발생하여 `predict()` 대신 `invoke()` 사용</br></br>

-  
  ```   
  LangChainDeprecationWarning: The class 'langchain_community.llms.openai.OpenAI' was deprecated in langchain-community 0.0.10 and will be removed in 0.2.0. An updated version of the class exists in the langchain-openai package and should be used instead. To use it run 'pip install -U langchain-openai' and import as 'from langchain_openai import OpenAI'.
  ```
  경고 발생하여 `pip install -U langchain-openai` 실행 후 `from langchain_openai import OpenAI`로 수정</br></br>

- 
  ```
  LangChainDeprecationWarning: The class 'langchain_community.chat_models.openai.ChatOpenAI' was deprecated in langchain-community 0.0.10 and will be removed in 0.2.0. An updated version of the class exists in the langchain-openai package and should be used instead. To use it run 'pip install -U langchain-openai' and import as 'from langchain_openai import ChatOpenAI'.
    ```
  경고 발생하여 `pip install -U langchain-openai` 실행 후 `from langchain_openai import ChatOpenAI`으로 수정

  ## 참고 문헌
  - https://www.youtube.com/watch?v=BLM3KDaOTJM&list=PLQIgLu3Wf-q_Ne8vv-ZXuJ4mztHJaQb_v&index=3