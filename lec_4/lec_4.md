## Prompt란?
Prompt는 모델에 대한 입력을 의미하며, ChatGPT에 보내는 커맨드를 Prompt라고 볼 수 있다. Langchain에서 제공하는 클래스와 함수를 통해 Prompt를 쉽게 구성할 수 있다.

- Prompt Template
일반적인 프롬프트 템플릿 생성을 위해 활용되는 프롬프트 템플릿, Competion model을 위한 템플릿
- Chat Prompt Template
채팅 LLM에 프롬프트를 전달하기 위해 활용되는 특화 프롬프트 템플릿, Chat completion model을 위한 템플릿
</br></br>

강의에서 사용하는 OpenAI의 `text-davinci-003`이 더 이상 지원하지 않는 모델이라, GPT-3 모델들과 비슷한 역량을 가지는 `gpt-3.5-turbo-instruct` 모델을 활용하였다.(OpenAI()의 default model_name)
(https://platform.openai.com/docs/models/moderation 참조)
</br>
## Prompt 사용해보기
```python
### Prompt Template ###
string_prompt = PromptTemplate.from_template("tell me a joke about {subject}")
string_prompt_value = string_prompt.format_prompt(subject="soccer")
print(string_prompt_value)
print(string_prompt_value.to_string())
```
실행 결과
```
text='tell me a joke about soccer'
tell me a joke about soccer
```
`from_template()`에 들어가는 문장 내에서 가변적으로 변경하고자 하는 변수를 중괄호로 묶는다. 위의 예시에서는 `subject` 부분만 사용자가 가변적으로 변경할 수 있도록 하였다.</br> `format_prompt()`에 변수의 값을 지정해주어 prompt를 생성할 수 있고, 이때 `to_string()`을 이용하면 raw text를 반환받을 수 있다.

```python
### Chat Prompt Template ###
chat_prompt = ChatPromptTemplate.from_template("tell me a joke about {subject}")
chat_prompt_value = chat_prompt.format_prompt(subject="soccer")
print(chat_prompt_value)
print(chat_prompt_value.to_string())
```
실행 결과
```
messages=[HumanMessage(content='tell me a joke about soccer')]
Human: tell me a joke about soccer
```
Chat Prompt Template도 동일한 방식으로 사용 가능하지만, 생성된 Prompt가 HumanMessage의 content로 들어가는 것을 확인할 수 있다. </br></br>
## Prompt 사용하여 GPT-3와 대화하기
```python
gpt = OpenAI(temperature=1)

template = """
너는 요리사야. 내가 가진 재료들로 만들 수 있는 요리를 추천하고, 그 요리의 레시피를 알려줘.
내가 가진 재료는 아래와 같아.

<재료>
{재료}
"""
prompt_template = PromptTemplate(
    input_variables = ['재료'],
    template = template
)

prompt = prompt_template.format(재료='양파, 계란, 사과, 빵')
answer = gpt.invoke(prompt)
print(prompt)
print(answer)
```
실행 결과
```
너는 요리사야. 내가 가진 재료들로 만들 수 있는 요리를 추천하고, 그 요리의 레시피를 알려줘.
내가 가진 재료는 아래와 같아.

<재료>
양파, 계란, 사과, 빵


네가 가진 재료로 만들 수 있는 요리는 아래와 같아.
1. 양파 계란말이
레시피:
재료: 양파, 계란, 소금, 후추
(1) 양파를 곱게 썰어서 젓가락으로 잘 섞는다.
(2) 계란을 납작하게 풀어서 후추와 소금을 넣고 잘 섞는다.
(3) 양파를 넣은 젓가락에 계란을 부어준다.
(4) 팬에 기름을 두르고 중약불로 달군 후, 양파 계란말이를 올린다.
(5) 양면이 적당히 익으면 접시에 담아서 바삭하게 먹는다.

2. 사과 토스트
레시피:
재료: 사과
```

## Chat Prompt Template 활용하여 ChatGPT와 대화하기
```python
# ChatGPT 모델 로드
chatgpt = ChatOpenAI(temperature=0)

# ChatGPT에 역할 부여
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

# 사용자가 입력할 매개변수 template 선언
human_template = "{재료}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# Chat prompt 템플릿 구성
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# 재료를 전달하여 Prompt 생성하고, 모델에 전달
prompt = chat_prompt.format_prompt(재료="양파, 계란, 사과, 빵").to_messages()
answer = chatgpt.invoke(prompt)

print(prompt)
print(answer)
```
실행 결과
```
[SystemMessage(content='\n너는 요리사야. 내가 가진 재료들로 만들 수 있는 요리를 추천하고, 그 요리의 레시피를 알려줘.\n내가 가진 재료는 아래와 같아.\n\n<재료>\n양파, 계란, 사과, 빵\n'), HumanMessage(content='양파, 계란, 사과, 빵')]

content='가지고 있는 재료로 만들 수 있는 요리 중 하나는 "양파 계란말이"야. 이 요리는 양파와 계란을 함께 볶아서 만드는 간단하면서도 맛있는 요리야. 빵은 따로 사용하지 않지만, 양파 계란말이와 함께 먹으면 좋을 거야.\n\n양파 계란말이의 레시피는 아래와 같아.\n\n<양파 계란말이 레시피>\n1. 양파를 깍두기 크기로 썰어줘.\n2. 계란을 풀어서 소금과 후추로 간을 해줘.\n3. 팬에 식용유를 두르고 양파를 넣어 중간 불에서 볶아줘.\n4. 양파가 투명해질 때까지 볶은 후, 계란을 부어줘.\n5. 계란이 익을 때까지 저어가며 볶아줘.\n6. 양파 계란말이가 완성되면 그릇에 담아서 서빙해줘.\n\n양파 계란말이는 밥과 함께 먹으면 맛있어. 빵 대신에 밥을 준비해서 양파 계란말이와 함께 즐겨봐. 맛있게 즐겨!'
```
## Prompt Template을 활용한 Few-shot
Few-shot은 딥러닝 모델에 예시 결과물을 제시함으로써 모델이 특정한 형태의 결과물을 출력하도록 유도하는 것으로, 특수한 형태의 결과물, 구조화된 답변을 원하는 경우 사용할 수 있다.

```python
gpt = OpenAI(temperature=1)

examples = [
    {
        "question": "아이유로 삼행시 만들어줘",
        "answer":
        """
        아: 아이유는
        이: 이런 강의를 들을 이
        유: 유가 없다.
        """
    },
    {
        "question": "김민수로 삼행시 만들어줘",
        "answer":
        """
        김: 김치는 맛있다
        민: 민달팽이도 좋아하는 김치!
        수: 수억을 줘도 김치는 내꺼!
        """
    }
]

example_prompt = PromptTemplate(input_variables=["question", "answer"], template="Question: {question}\n{answer}")

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"]
)

print(prompt.format(input="호날두로 삼행시 만들어줘"))
```
실행 결과
```
Question: 아이유로 삼행시 만들어줘

        아: 아이유는
        이: 이런 강의를 들을 이
        유: 유가 없다.
        

Question: 김민수로 삼행시 만들어줘

        김: 김치는 맛있다
        민: 민달팽이도 좋아하는 김치!
        수: 수억을 줘도 김치는 내꺼!
        

Question: 호날두로 삼행시 만들어줘
```
</br></br>
```python
question = "호날두로 삼행시 만들어줘"
print(gpt.invoke(question))
print(gpt.invoke(prompt.format(input=question)))
```
실행 결과
```
날짜는아직 먼 미래
복잡한 인간의 주인인 너
네 미래를 어떻게 이해해야 할지
알 수 없는 어둠 속에 너의 미래

축복으로 가득찬 그 날이 멀리
너의 꿈을 이루기를 빌며 기도하는 나
날짜는 멀게 느껴지지만 불안한 너의 걱정은
나는 나와 같이 당신의 곁에
같은 의지로 뭉쳐 우리에게 불러

07 07 이 말의 숫자들로
우리의 소원을 날리는 삼행시
다가오는 미래를 예측할 순 없지만
07 07에 우리의 꿈들이 이루어지기를


        호: 호날두는 축구의 신
        날: 날마다 뛰어나는 실력
        두: 두고봐야 되는 세계 최고의 선수 
       
```
 모델에 호날두 삼행시를 지어달라고 하면 전혀 관련되지 않은 답변을 주는 것을 확인할 수 있지만, `FewShotPromptTemplate`을 통해 예시들을 제공하여 삼행시를 짓도록 유도한 경우에는 호날두 삼행시를 적절히 답변하는 것을 확인할 수 있다.</br></br>

## Example Selector를 이용한 동적 Few-shot
Example Selector를 이용하여 Few-shot 예제를 동적으로 입력해, LLM이 사용자의 입력에 동적으로 반응해 원하는 형태의 출력을 하도록 유도할 수 있다.

```python
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma 
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI

gpt = OpenAI(temperature=1)

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="input: {input}\noutput: {output}"
)

examples = [
    {"input": "행복", "output": "슬픔"},
    {"input": "흥미", "output": "지루"},
    {"input": "불안", "output": "안정"},
    {"input": "긴 기차", "output": "짧은 기차"},
    {"input": "큰 공", "output": "작은 공"},
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    Chroma,
    k=1
)

Similar_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="주어진 입력에 대해 반대의 의미를 가진 단어를 출력해줘",
    suffix="input: {단어}\nOutput:",
    input_variables=["단어"]
)
print(Similar_prompt.format(단어="무서운"))
```
실행 결과
```
input: 불안
output: 안정

input: 무서운
Output:
```
Example Selector에 의해 `무서운`과 가장 비슷한 예시로 `input: 불안 output: 안정`이 선택되어 위와 같이 출력된다.</br></br>

```python
query="큰 비행기"
print(Similar_prompt.format(단어=query))
print(gpt.invoke(Similar_prompt.format(단어=query)))
```
실행 결과
```
주어진 입력에 대해 반대의 의미를 가진 단어를 출력해줘

input: 긴 기차
output: 짧은 기차

input: 큰 비행기
Output:
 작은 비행기
```
`큰 비행기`로 prompt를 만들면, Example Selector에 의해 `input: 긴 기차 output: 짧은 기차`의 예시가 선택되고 따라서 모델이 `작은 비행기`라는 답을 잘 출력하고 있는 것을 확인할 수 있다.</br></br>

## Output Parser로 출력 형식 고정
OutputParser를 통해 LLM의 답변을 리스트, JSON 등 특정한 형식으로 고정할 수 있다.

```python
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import OpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser

gpt = OpenAI(temperature=1)

output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()

prompt = PromptTemplate(
    template="{subject} 5개를 추천해줘.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions}
)

input = prompt.format(subject="영화")
output = gpt.invoke(input)

print(input)
print(output)
print(output_parser.parse(output))
```

```
영화 5개를 추천해줘.
Your response should be a list of comma separated values, eg: `foo, bar, baz`

 La La Land, The Shawshank Redemption, The Grand Budapest Hotel, Parasite, Inception
['La La Land', 'The Shawshank Redemption', 'The Grand Budapest Hotel', 'Parasite', 'Inception']
```

## 참고 문헌
- https://www.youtube.com/watch?v=y6D5Hn_k4lE&list=PLQIgLu3Wf-q_Ne8vv-ZXuJ4mztHJaQb_v&index=4