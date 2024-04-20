## Langchain이란?
Langchain은 언어 모델로 구동되는 애플리케이션을 개발하기 위한 프레임워크이다. </br>Langchain은 언어 모델을 API를 통해 호출할 뿐만 아니라 언어 모델을 다른 데이터 소스에 연결하고, 언어 모델이 환경과 상호 작용할 수 있게 하여 애플리케이션에서 언어 모델을 더 잘 활용할 수 있도록 도와준다. 
 
## ChatGPT의 한계점
1. <strong>정보 접근 제한</strong></br>
Chatgpt는 학습 데이터 업데이트 이후에 발생한 정보에 대해서는 거짓된 답변을 제공하거나 답변을 하지 못한다.
2. <strong>토큰 제한</strong></br>
Chatgpt는 특정 토큰만큼만을 사용자와의 대화에서 기억할 수 있다는 토큰 제한을 가지고 있다.
3. <strong>환각 현상(Hallucination)</strong></br>
Chatgpt는 데이터로부터 학습한 통계 패턴을 기반으로 문장을 생성해나가는 '추론 엔진'이기 때문에, 정확하지 않은 문장을 생성하여 사실인 것처럼 말하는 환각 현상이 나타난다. 

## Langchain의 필요성
Changpt의 한계점을 극복할 수 있는 방안으로는 Fine-tuning, N-short Learning, In-context Learning이 존재하는데, Langchain은 In-context Learning의 도구로서 위와 같은 LLM의 한계점을 극복하고 더 잘 활용하는데 도움이 될 수 있다. 

## Langchain의 구조
- <strong>LLM</strong>: 초거대 언어 모델로, Langchain의 핵심 구성요소이다.
- <strong>Prompts</strong>: LLM에 지시하는 명령문
- <strong>Index</strong>: LLM이 문서를 쉽게 탐색할 수 있도록 구조화하는 모듈
- <strong>Memory</strong>: LLM이 문맥을 기억하고, 이를 기반으로 대화할 수 있도록 하는 모듈
- <strong>Chain</strong>: LLM Chain을 통해, 연속적으로 LLM을 호출할 수 있도록 하는 핵심 구성요소
- <strong>Agents</strong>: LLM이 기존의 Prompt Template을 통해서는 수행할 수 없는 작업을 가능하게 하는 모듈
## 참고문헌
https://www.youtube.com/watch?v=WWRCLzXxUgs&list=PLQIgLu3Wf-q_Ne8vv-ZXuJ4mztHJaQb_v&index=2