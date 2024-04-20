## RAG(Retrieval Augmented Generation)
LLM이 외부 데이터를 참조하여 답변할 수 있도록 하는 프레임워크</br>
Fine-tuning을 통해 LLM이 가지고 있지 않은 지식을 새롭게 학습시킬 수 있지만, LLM 모델의 학습을 위해 GPU 등의 고급 장비가 필요하다는 단점이 있어 RAG가 널리 퍼지게 되었다.</br>
사용자가 질문을 하면 Q/A 시스템이 Vector DB, Feature Store 등 외부 데이터 저장소에서 사용자의 질문과 유사한 문장을 검색하고, 유사 문장과 질문을 합쳐 LLM에게 Prompt로 전달해줌으로서 LLM으로부터 답변을 얻을 수 있다.
</br></br>
## Retrieval
Langchain에서는 `Document Loaders`를 통해 문서를 불러오고, `Text Splitters`를 통해 문서를 여러 텍스트로 분할하고, `Vector Embeddings`를 통해 텍스트를 수치화한 후 벡터 저장소에 저장하며, `Retrievers`를 통해 사용자의 질문과 유사한 문장을 검색한다.
</br></br>
## Document Loaders
pdf, ppt, word 등 다양한 형태의 문서를 RAG 전용 객체로 불러들이는 모듈이다. Document Loader를 통해 문서를 불러오면 두 가지의 구성요소를 가지게 되는데, 문서의 내용인 `Page_content`와 문서의 위치, 제목, 페이지 넘버 등을 나타내는 `Metadata`이다.
</br></br>
- URL Document Loader

    URL Loader인 `WebBaseLoader`, `UnstructuredURLLoader`를 통해 웹에 기록된 글을 텍스트 형식으로 가져올 수 있다.

    - WebBaseLoader
        ```python
        from langchain_community.document_loaders import WebBaseLoader

        loader = WebBaseLoader("URL")
        data = loader.load()
        ```

    - UnstructuredURLLoader
        ```python
        from langchain_community.document_loaders import UnstructuredURLLoader

        urls = [
            "URL_1",
            "URL_2"
        ]

        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()
        ```
</br>

- PDF Document Loader
    ```python
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader("title.pdf")
    pages = loader.load_and_split()
    ```
    `load_and_split()`을 통해 페이지별로 나눌 수 있다.
</br></br>
- Word Document Loader
    ```python
    from langchain_community.document_loaders import Docx2txtLoader

    loader = Docx2txtLoader("title.docx")
    data = loader.load()
    ```
</br>

- CSV Document Loader
    ```python
    from langchain_community.document_loaders.csv_loader import CSVLoader

    loader = CSVLoader(
        file_path="file_path", csv_args={
            'delimiter': ',',
            'quotechar': '"',
            'fieldnames': ['name_1', 'name_2']
    })

    data = loader.load()
    ```

## 참고 문헌
- https://www.youtube.com/watch?v=tIU2tw3PMUE&list=PLQIgLu3Wf-q_Ne8vv-ZXuJ4mztHJaQb_v&index=5