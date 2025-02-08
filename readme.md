
If you want to see the detail of this project, please check the following languages.

- [English](README.en.md)
- [Korean](README.ko.md)

# LFSS (Local File Semantic Search)
  
This project is a document search project that uses a local file vector database. It helps users search for documents stored in the local file system and find similar documents. It operates **entirely in a local environment**.

## Features

- Supported file formats: [pdf, docx, doc, txt]
- Available OS : [Windows, Linux, MacOS]
- LLM : Gemma2:2B
- Vector DB : Faiss 
<!-- - 기능 2: [기능에 대한 설명] -->
<!-- - 기능 3: [기능에 대한 설명] -->

## Installation & Execution

To install this project in a local environment, please follow these steps.

First, you need to install Ollama. please refer to [Ollama](https://ollama.com) to install it. and pull the Gemma2:2B model.


### Install Local LLM Model

1. Pull Gemma2:2B model
    ```bash
    ollama pull gemma2:2b
    ```

---

Second, you need to install the project. you can install it by using the following git command.

### Install project & Setup the project

1. Clone the repository

   ```bash
   git clone https://github.com/jeean0668/LFSS.git
   cd LFSS
   ```

2. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

3. Run the application

   ```bash
   streamlit run app.py
   ```

## Usage

After running the project, please refer to [usage] to use it.

## Contribution

If you want to contribute or have any questions about this project, please contact me.

email : gijeean0668@gmail.com
## License

It can be used for free and commercial use. For more details, please refer to the `LICENSE` file.

## Project Installation

```bash
pip install -r requirements.txt
```

## Project Execution

```bash
streamlit run app.py
```

## Current Issues
- 2025.01.26 : if we upload multiple files(more than 100), the searching engine's accuracy is becoming worse.

1000개 이상의 파일 업로드가 필요하고, 빠른 속도를 달성해야 하기 때문에 FAISS 사용 결정. 하지만 실시간 파일 업데이트 부분에서 문제가 있음. 

- 2025.01.26 : 파일 업로드 시 벡터 DB 초기화 및 저장 부분 추가. update_rag.py 파일에 vectorDB 초기화 및 저장 부분을, run.py 파일에 파일 업로드 부분을 추가. 

## How it works?

Before:
```bash
/home/user/sample_data/
├── 2025년_산재보험료율표_다운로드.pdf
├── [양식] 계좌입금신청서(사업자)양식.pdf
├── announcement.pdf
├── test.pdf
└── 로켓피치_AWS_우수상_아마존악어떼_김지안,홍수정,남혜원,김태수,오재혁,한대명_250123_143423.pdf

Question :
How to pay for the insurance?
```

Result:
```bash 
You have to visit the website and pay for the insurance. Web site is https://www.rocketpich.com/ you have to prepare your account information.

source :
1. [/home/user/sample_data/2025년_산재보험료율표_다운로드.pdf](https://www.rocketpich.com/wp-content/uploads/2025년_산재보험료율표_다운로드.pdf)

2. [/home/user/sample_data/계좌입금신청서(사업자)양식.pdf](https://www.rocketpich.com/wp-content/uploads/계좌입금신청서(사업자)양식.pdf)
```


export OPENAI_API_KEY=fake-key
export OPENAI_BASE_URL=http://localhost:11434/v1