
If you want to see the detail of this project, please check the following languages.

- [English](README.en.md)
- [Korean](README.ko.md)

# LFSS (Local File Semantic Search)
  
This project is a document search project that uses a local file vector database. It helps users search for documents stored in the local file system and find similar documents. It operates **entirely in a local environment**.

## Features

you can search for documents stored in the local file system and find similar documents. especially if you select a directory, **all of the files in the directory will be uploaded and searched.**

- Supported file formats: [pdf, docx, txt]
- Available OS : [Linux, MacOS]
- LLM : Gemma2:2B
- Vector DB : Faiss
- Python version : 3.12.1
- Target Users : 
    - Users **who work in a company** and want to search for documents stored in the local file system. 
    - Users **who want to find similar documents** in the local file system.

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

## Contribution

If you want to contribute or have any questions about this project, please contact me.

- email : gijeeankyung@gmail.com 
- github : https://github.com/jeean0668

## License

It can be used for free and commercial use. For more details, please refer to the `LICENSE` file.

## Current Issues
- 2025.01.26: **Real-time file update issues**. Decided to use FAISS due to the need to upload more than 1000 files and achieve fast speed. However, there are issues with real-time file updates.
- 2025.01.26: **Searching engine's accuracy issues**. When uploading multiple files (more than 100), the searching engine's accuracy is becoming worse.
- 2025.01.26: **Vector generation issues**. Due to the token limit of the gemma2:2b model, as the number of files increases, it is not good at generating the vector.
- 2025.02.07: **Korean file search performance issues**. Korean file search performance is lower compared to English. It is estimated that data loss during vector compression is large because Korean tokens are larger than alphabets.
- 2025.02.07: **Windows OS support issues**. Windows OS is not supported. We will support it in the future.
- 2025.02.07: **LLM response issues**. The LLM response is not good. We will improve it in the future.

