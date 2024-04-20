import os
import logging
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma 
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI, ChatOpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import (
    PromptTemplate, 
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage, 
    HumanMessage, 
    SystemMessage
)

### logger ###
logging.basicConfig(filename="lec_4.log", filemode="w", level=logging.INFO)
logger = logging.getLogger()

### api.txt에서 API key 읽어오기 ###
f = open("api.txt", "r")
api_key = f.readline()
f.close()

### api.txt에서 읽어온 키를 넣어주기! ###
os.environ["OPENAI_API_KEY"] = api_key

### Prompt Template ###
string_prompt = PromptTemplate.from_template("tell me a joke about {subject}")
string_prompt_value = string_prompt.format_prompt(subject="soccer")
logger.info(f"Prompt Template result: {string_prompt_value}")

### Chat Prompt Template ###
chat_prompt = ChatPromptTemplate.from_template("tell me a joke about {subject}")
chat_prompt_value = chat_prompt.format_prompt(subject="soccer")
logger.info(f"Chat Prompt Template result: {chat_prompt_value}")

### Example of using Prompt Template ###
gpt = OpenAI(temperature=1)
logger.info(f"model_name: {gpt.model_name}")

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

logger.info(f"Prompt: {prompt}")
logger.info(f"Answer: {answer}")

### Example of using Chat Prompt Template ###
chatgpt = ChatOpenAI(temperature=0)
logger.info(f"model_name: {chatgpt.model_name}")

system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{재료}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

prompt = chat_prompt.format_prompt(재료="양파, 계란, 사과, 빵").to_messages()
answer = chatgpt.invoke(prompt)

logger.info(f"Prompt: {prompt}")
logger.info(f"Answer: {answer}")

### Few-shot using Prompt Template ###
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

question = "호날두로 삼행시 만들어줘"
logger.info(prompt.format(input="호날두로 삼행시 만들어줘"))
logger.info(f"answer for {question}: {gpt.invoke(question)}")
logger.info(f"answer for {prompt.format(input=question)}: {gpt.invoke(prompt.format(input=question))}")

### Example Selector ###
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

query="큰 비행기"
logger.info(Similar_prompt.format(단어=query))
logger.info(gpt.invoke(Similar_prompt.format(단어=query)))

### Example of Output Parser ###
output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()

prompt = PromptTemplate(
    template="{subject} 5개를 추천해줘.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions}
)

input = prompt.format(subject="영화")
output = gpt.invoke(input)

logger.info(f"input: {input}")
logger.info(f"output: {output}")
logger.info(output_parser.parse(output))
