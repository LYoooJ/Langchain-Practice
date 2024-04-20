## LLM이란? (Large Language Model)
Large Language Model(초거대 언어 모델)은 방대한 양의 언어 데이터를 기반으로 학습된 초대형 딥러닝 모델로 수천억개의 parameter를 가지며, 다양한 자연어 처리 task를 수행할 수 있다. 

## 트랜스포머(Transformer)
현재 자연어처리(NLP) 모델 대부분의 기반이 되는 아키텍쳐로, 주로 인코더(Encoder)와 디코더(Decoder) 블록을 통해 구성된다. 
</br>인코더(Encoder)는 입력 문장에 대한 표현을 도출해내는 것으로 모델이 입력을 잘 이해할 수 있도록 만드는 블록이며, 디코더(Decoder)는 인코더가 도출한 표현과 다른 입력을 사용해 출력 문장을 생성하는 것으로 모델이 출력을 잘 할 수 있도록 만드는 블록이다.
 
## Closed Source vs Open Source
- Closed Source
    모델의 구성 요소 등 상세 내용에 대해서 알 수 없고, 모델을 가져다 쓸 때 돈을 지불해야 한다.
    * 장점: 모델이 뛰어난 성능을 가지고, API 방식으로 편리하게 사용 가능하다.
    * 단점: 보안을 보장할 수 없고, API 호출 비용이 발생한다.
- Open Source
    모델이 외부에 공개되어 있어 자유롭게 활용 가능하다.
    * 장점: 모델이 Closed Source 못지 않은 성능을 가지며, 높은 보안성과 비용이 적게 든다는 장점이 있다.
    * 단점: 개발 난이도가 높고, 모델을 돌리기 위한 GPU 서버가 필요하다.

## 참고문헌
- https://www.youtube.com/watch?v=XwlLeVhWCCc&list=PLQIgLu3Wf-q_Ne8vv-ZXuJ4mztHJaQb_v&index=2
- https://wikidocs.net/166788