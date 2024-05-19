import streamlit as st
import os
# from langchain_community.document_loaders import TextLoader
# from langchain_community.document_loaders import UnstructuredMarkdownLoader
# from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.document_loaders import PyMuPDFLoader

import base64
from pathlib import Path

@st.experimental_fragment
class FileManager():
    def __init__(self):
        pass

    def list_selected_files(self, path, 확장자):
        file_list = os.listdir(path)
        selected_files = [file for file in file_list if file.endswith(확장자)]
        return selected_files

    def list_all_files(self, path):
        file_list = os.listdir(path)
        selected_files = [file for file in file_list]
        return selected_files


# class Parsing():
#     def __init__(self):
#         pass

#     def load_TextLoader(self, path):
#         loader = TextLoader(path)
#         data = loader.load()
#         return data
    
#     def load_UnstructuredMarkdownLoader(self, path):
#         loader = UnstructuredMarkdownLoader(path, mode='elements') # elements or single
#         data = loader.load()
#         return data
    
#     def load_CSVLoader(self, path):
#         loader = CSVLoader(file_path=path)
#         data = loader.load()
#         return data
    
#     def load_PyPDFLoader(self, path):
#         loader = PyPDFLoader(path)
#         pages = loader.load()
#         return pages
    
#     def load_PyMuPDFLoader(self, path):   
#         loader = PyMuPDFLoader(path)
#         data = loader.load()
#         return data
    

class ShowPdf():
    def __init__(self):
        pass

    def show_pdf(self, path):
        pdf_path1 = Path(path)
        base64_pdf = base64.b64encode(pdf_path1.read_bytes()).decode("utf-8")
        pdf_display = f"""
            <iframe src="data:application/pdf;base64,{base64_pdf}" width="800px" height=1800" type="application/pdf"></iframe>
        """
        st.markdown(pdf_display, unsafe_allow_html=True)


import pdfplumber
from spire.pdf.common import *
from spire.pdf import *
from typing import Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from pathlib import Path
import fitz

def block_based_parsing_by_page(pdf_path, page_num, crop:bool):
    results = ""
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        if crop:
            bounding_box = (3, 70, 590, 770)   #default : (0, 0, 595, 841)
            page = page.crop(bounding_box, relative=False, strict=True)
        else: pass
        words = page.extract_words()
        lines = {}
        for word in words:
            line_top = word['top']
            if line_top not in lines:
                lines[line_top] = []
            lines[line_top].append(word['text'])
        
        # Sort and print lines based on their y-coordinate
        for top in sorted(lines.keys()):
            result = ""
            if len(lines[top]) > 1:
                result = ' '.join(lines[top])
                # print(result)
            results = results + "\n" + result
    return results

def table_parser(pdf_path, page_num, crop):
    full_result = []
    
    pdf = pdfplumber.open(pdf_path)
    # Find the examined page
    table_page = pdf.pages[page_num]
    if crop:
        bounding_box = (3, 70, 590, 770)   #default : (0, 0, 595, 841)
        table_page = table_page.crop(bounding_box, relative=False, strict=True)
    else: pass
    tables = table_page.extract_tables()
    if tables:
        for table in tables:
            table_string = ''
            # Iterate through each row of the table
            for row_num in range(len(table)):
                row = table[row_num]
                # Remove the line breaker from the wrapped texts
                cleaned_row = [item.replace('\n', ' ') if item is not None and '\n' in item else 'None' if item is None else item for item in row]
                # Convert the table into a string 
                table_string+=('|'+'|'.join(cleaned_row)+'|'+'\n')
            # Removing the last line break
            table_string = table_string[:-1]

        full_result.append(table_string)
        return table_string

# def image_extractor(pdf_path, page_num, prefix):
#     doc = PdfDocument()
#     doc.LoadFromFile(pdf_path)
#     page = doc.Pages[page_num]
#     images = []
#     for image in page.ExtractImages():
#         images.append(image)
#     index = 0

#     directory = f"./images/{prefix}"
#     Path(directory).mkdir(parents=True, exist_ok=True)

#     image_filenames = []
#     for image in images:
#         imageFileName = f'{prefix}_{page_num}_{index}.png'
#         imageSaveName = f'./images/{prefix}/{page_num}_{index}.png'
#         image_filenames.append(imageFileName)
#         index += 1
#         image.Save(imageSaveName, ImageFormat.get_Png())
#     doc.Close()
#     return image_filenames

def flags_decomposer(flags):
        """Make font flags human readable."""
        l = []
        if flags & 2 ** 0:
            l.append("superscript")
        if flags & 2 ** 1:
            l.append("italic")
        if flags & 2 ** 2:
            l.append("serifed")
        else:
            l.append("sans")
        if flags & 2 ** 3:
            l.append("monospaced")
        else:
            l.append("proportional")
        if flags & 2 ** 4:
            l.append("bold")
        return ", ".join(l)

# def extract_colored_font(doc, page_num):
#     results = []
    # for page in doc:
    # read page text as a dictionary, suppressing extra spaces in CJK fonts
    # blocks = doc[page_num].get_text("dict", flags=11)["blocks"]
    # for b in blocks:  # iterate through the text blocks
    #     for l in b["lines"]:  # iterate through the text lines
    #         for s in l["spans"]:  # iterate through the text spans
    #             font_properties = "Font: '%s' (%s), size %g, color #%06x" % (
    #                 s["font"],  # font name
    #                 flags_decomposer(s["flags"]),  # readable font flags
    #                 s["size"],  # font size
    #                 s["color"],  # font color
    #             )
    #             if s["color"] != 0:
    #                 results.append(s["text"])
                    # st.markdown(f"Text: {s['text']}, color: {s['color']}")  # simple print of text
                    # st.markdown(font_properties)
    # return results



class CustomPDFLoader(BaseLoader):
    def __init__(self) -> None:
        pass

    def lazy_load(self, file_path, crop:bool) -> Iterator[Document]:  # <-- Does not take any arguments
        full_result = []
        prefix = file_path.split("\\")[-1].split(".")[0].strip()
        with pdfplumber.open(file_path) as pdf1:
            page_number = 0
            docs_for_color = fitz.open(file_path)
            for _ in pdf1.pages:
                page_result = block_based_parsing_by_page(file_path, page_number, crop)
                table_result = table_parser(file_path, page_number, crop)
                # image_files = image_extractor(file_path, page_number, prefix)

                if table_result:
                    total_pag_result = page_result + "\n\n" + table_result
                    result = Document(
                        page_content=total_pag_result,
                        metadata={"page_number": page_number, "keywords":prefix, "source": file_path},
                    )
                else:
                    result = Document(
                        page_content=page_result,
                        metadata={"page_number": page_number, "keywords":prefix, "source": file_path},
                    )
                full_result.append(result)
                page_number += 1


        return full_result

    





