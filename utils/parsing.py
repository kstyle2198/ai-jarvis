import streamlit as st
import os
from pathlib import Path

@st.experimental_fragment
class FileManager():
    def __init__(self):
        pass

    def list_all_files(self, path):
        file_list = os.listdir(path)
        selected_files = [file for file in file_list]
        return selected_files

    def list_selected_files(self, path, 확장자):
        file_list = os.listdir(path)
        selected_files = [file for file in file_list if file.endswith(확장자)]
        return selected_files



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

import fitz
import os

def image_extractor(file_path, page_num, prefix):
    images_path = f"./images/{prefix}/"
    Path(images_path).mkdir(parents=True, exist_ok=True)
    pdf_file = fitz.open(file_path)
    images_list = []
    page_content = pdf_file[page_num]
    images_list.extend(page_content.get_images())

    if len(images_list)!=0:
        for i, img in enumerate(images_list, start=1):
            #Extract the image object number
            xref = img[0]
            #Extract image
            base_image = pdf_file.extract_image(xref)
            #Store image bytes
            image_bytes = base_image['image']
            #Store image extension
            image_ext = base_image['ext']
            #Generate image file name
            image_name = str(page_num)+'_'+ str(i) + '.' + image_ext
            #Save image
            with open(os.path.join(images_path, image_name) , 'wb') as image_file:
                image_file.write(image_bytes)
                image_file.close()
    else:
        pass

from paddleocr import PaddleOCR
from pdf2image import convert_from_path
def pdf_to_image(path):
    pages = convert_from_path(path)
    return pages

def save_pdf_to_image(pdf_path, prefix):
    pages = pdf_to_image(pdf_path) 
    folder_path = f"D:/ai_jarvis/pdf_to_images/{prefix}/"
    Path(folder_path).mkdir(parents=True, exist_ok=True)     
    for idx, page in enumerate(pages):
        page.save(f'D:/ai_jarvis/pdf_to_images/{prefix}/{prefix}_{idx}.png', 'PNG')

class CustomPDFLoader(BaseLoader):
    def __init__(self) -> None:
        pass

    def lazy_load(self, file_path, crop:bool) -> Iterator[Document]:  # <-- Does not take any arguments
        full_result = []
        prefix = file_path.split("\\")[-1].split(".")[0].strip()
        with pdfplumber.open(file_path) as pdf1:
            page_number = 0
            # docs_for_color = fitz.open(file_path)
            for _ in pdf1.pages:
                page_result = block_based_parsing_by_page(file_path, page_number, crop)
                table_result = table_parser(file_path, page_number, crop)
                image_files = image_extractor(file_path, page_number, prefix)

                if page_result == "":
                    return False
            
                else:
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

    def ocr_parsing(self, pdf_path):
        prefix = pdf_path.split("\\")[-1].split(".")[0].strip()
        save_pdf_to_image(pdf_path, prefix)

        file_list = os.listdir(f"D:/ai_jarvis/pdf_to_images/{prefix}")
        selected_files = [file for file in file_list if file.endswith("png")]

        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        full_result = []
        for page_num, img_path in enumerate(selected_files):
            print(img_path)
            img_path = f"D:/ai_jarvis/pdf_to_images/{prefix}/"+img_path
            result = ocr.ocr(img_path)
            for idx in range(len(result)):
                res = result[idx]
                texts = []
                for line in res:
                    print(line)
                    print(line[1][0])
                    texts.append(line[1][0])
                resulting_string = " ".join(texts)

                result = Document(
                    page_content=resulting_string,
                    metadata={"page_number": page_num, "keywords":prefix, "source": pdf_path},
                )
            full_result.append(result)
        return full_result




if __name__ == "__main__":

    cpl = CustomPDFLoader()
    pdf_path = "D:/ai_jarvis/data/FWG.pdf"
    prefix = "FWG"
    result = cpl.ocr_parsing(pdf_path, prefix)
    print(result)


    # from PIL import Image
    # pages = pdf_to_image("../data/FWG.pdf")
    # for idx, page in enumerate(pages):
    #     print(page)
    #     # Save the image as a PNG file
    #     page.save(f'../pdf_to_images/FWG/path_to_save_image_{idx}.png', 'PNG')
    
    # file_list = os.listdir("../pdf_to_images/FWG")
    # selected_files = [file for file in file_list if file.endswith("png")]
    # print(selected_files)

    # from paddleocr import PaddleOCR
    # ocr = PaddleOCR(use_angle_cls=True, lang='en')
    # # for img_path  in selected_files:
    # img_path = "D:/ai_jarvis/pdf_to_images/FWG/path_to_save_image_2.png"
    # result = ocr.ocr(img_path)
    # # print(result)
    # for idx in range(len(result)):
    #     res = result[idx]
    #     texts = []
    #     for line in res:
    #         print(line)
    #         print(line[1][0])
    #         texts.append(line[1][0])
    #     resulting_string = " ".join(texts)
    #     print(resulting_string)
    #     break


   






