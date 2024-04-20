## Text Splitter
Text Splitter는 문서를 Chunk로 분할하여 토큰 제한을 가지는 LLM이 여러 개의 문장을 참고해서 답변할 수 있도록 하며, Chunk 하나가 Vector store의 하나의 임베딩 벡터로 저장된다. Vector store에서 사용자의 질문과 가장 유사한 임베딩 벡터의 chunk와 사용자의 질문이 합쳐져 최종적으로 LLM에 전달되는 Prompt가 된다.

- <strong>CharacterTextSpliter</strong>
</br>구분자 1개를 기준으로 분할하여, max_token을 지키지 못하는 경우가 발생할 수 있다.

    ```python
    from langchain.text_splitter import CharacterTextSplitter

    with open("file_name.txt") as f:
        content = f.read()

    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )

    texts = text_splitter.split_text(content)
    doc = text_splitter.create_documents([content])
    ```
    `\n\n`을 기준으로 텍스트 분할, `chunk_size`은 1000으로 하며, 이때 길이는 `len`을 이용하므로 글자수를 의미한다. `chunk_overlap`을 통해 이전 chunk의 끝부분을 포함하도록 할 수 있다.

    ```python
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import CharacterTextSplitter

    loader = PyPDFLoader("title.pdf")
    pages = loader.load_and_split()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )

    texts = text_splitter.split_documents(pages)
    ```
    PDF 문서를 로드하고 Text Splitter를 이용해 텍스트를 분할하는 예제로, pages는 `Document` 객체이기 때문에 Text Splitter을 이용할 때 `split_documents()` 함수를 이용한다.
</br></br>
- <strong>RecursiveCharacterTextSplitter</strong>
</br>줄바꿈, 마침표, 쉼표 순으로 재귀적으로 분할하여 chunk가 max_token을 넘지 않도록 하기 때문에 max_token을 지켜 분할하며, 문장들의 의미를 최대한 보존하며 분할할 수 있다.

    ```python
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    with open("file_name.txt") as f:
        content = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
    )

    texts = text_splitter.create_documents([content])
    ```
    </br>
- <strong>기타 Splitter</strong>
</br>코드, latex 등과 같은 컴퓨터 언어로 작성된 문서는 `text_splitter`로 처리할 수 없어 해당 언어를 위해 특별하게 분할하는 `splitter`가 필요하다. 예시로 Python 문서를 분할할 때는 `def`, `class`처럼 하나의 단위로 묶이는 것을 기준으로 삼을 수 있다.

    ```python
    from langchain.text_splitter import (
        RecursiveCharacterTextSplitter,
        Language
    )

    separator = RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON)
    print(separator)
    ```

    실행 결과
    ```
    ['\nclass ', '\ndef ', '\n\tdef ', '\n\n', '\n', ' ', '']
    ```
    python 텍스트를 분할할 때 구분자로 사용하는 기준을 출력한다.
</br></br>

    ```python
    from langchain.text_splitter import (
        RecursiveCharacterTextSplitter,
        Language
    )

    PYTHON_CODE = """
    def hello_world():
        print("Hello, World!")

    # Call the function
    hello_world()
    """

    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, 
        chunk_size=50,
        chunk_overlap=0
    )

    python_docs = python_splitter.create_documents([PYTHON_CODE])
    print(python_docs)
    ```
    `chunk_size`는 50, `chunk_overlap` 없이 Python 언어에 대해 분할한다.


    실행 결과
    ```
    [Document(page_content='def hello_world():\n    print("Hello, World!")'), Document(page_content='# Call the function\nhello_world()')]
    ```

- <strong>토큰 단위 Text Splitter</strong>
LLM은 정해진 토큰 이상의 텍스트는 처리할 수 없기 때문에 LLM이 처리할 수 있는 토큰에 맞추어 chunk를 제한하는 것이 필요하며, 이를 위해 사용할 LLM의 토큰 제한을 알고 해당 LLM이 사용하는 Embedder을 기반으로 토큰 수를 계산하여 텍스트를 분할해야 한다.

    `cl100k_base`는 gpt 모델에서 사용하는 embedding model이며, `tiktoken.get_encoding()`을 통해 해당 모델에 대한 토크나이저를 로드하여 토큰 수를 계산하는 `tiktoken_len`함수를 정의하고, `RecursiveCharacterTextSplitter` 사용 시 `length_function`을 `tiktoken_len`으로 지정해줌으로서 토큰 단위로 텍스트를 분할할 수 있다.
    
    ```python
    import tiktoken
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def tiktoken_len(text):
        tokens = tokenizer.encode(text)
        return len(tokens)
    
    loader = PyPDFLoader("title.pdf")
    pages = loader.load_and_split()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        length_function=tiktoken_len
    )

    texts = text_splitter.split_documents(pages)
    ```