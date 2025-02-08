import os
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, TextLoader
from utils.loader.docx_directory_loader import DocxDirectoryLoader

class FileLoader:
    """다양한 파일 형식을 로드하는 클래스"""
    def __init__(self, folder_path, show_progress=False):
        self.folder_path = folder_path
        self.show_progress = show_progress

    def load_txt_files(self):
        txt_loader = DirectoryLoader(self.folder_path, glob="**/*.txt", loader_cls=TextLoader, show_progress=self.show_progress)
        return txt_loader.load()

    def load_pdf_files(self):
        pdf_loader = DirectoryLoader(self.folder_path, glob="**/*.pdf", loader_cls=PyMuPDFLoader, show_progress=self.show_progress)
        return pdf_loader.load()

    def load_docx_files(self):
        docx_loader = DocxDirectoryLoader(self.folder_path, show_progress=self.show_progress)
        return docx_loader.load()

    def load_all_files(self):
        txt_docs = self.load_txt_files()
        pdf_docs = self.load_pdf_files()
        docx_docs = self.load_docx_files()
        return txt_docs + pdf_docs + docx_docs 