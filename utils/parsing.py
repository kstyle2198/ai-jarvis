import os
from pathlib import Path
import pdfplumber
from spire.pdf.common import *
from spire.pdf import *
from typing import Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from pathlib import Path
import fitz

'''
상단은 헬퍼 함수.. 메인 클레스는 맨 하단의 CustomPDFLoader임
'''

#### [Start] File handler class ########################################
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
#### [Start] File handler class ########################################

#### [Start] PDF parsing help function ############################################3
def block_based_parsing_by_page(pdf_path, page_num, crop:bool):  # 텍스트 파싱 (크롭 적용)
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
            results = results + "\n" + result
    return results

def table_parser(pdf_path, page_num, crop):   # 테이블을 마크다운 형식으로 파싱
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

# def image_extractor(file_path, page_num, prefix):   # 추출 이미지 파일 따로 저장하기
#     images_path = f"./images/{prefix}/"
#     Path(images_path).mkdir(parents=True, exist_ok=True)
#     pdf_file = fitz.open(file_path)
#     images_list = []
#     page_content = pdf_file[page_num]
#     images_list.extend(page_content.get_images())

#     if len(images_list)!=0:
#         for i, img in enumerate(images_list, start=1):
#             #Extract the image object number
#             xref = img[0]
#             #Extract image
#             base_image = pdf_file.extract_image(xref)
#             #Store image bytes
#             image_bytes = base_image['image']
#             #Store image extension
#             image_ext = base_image['ext']
#             #Generate image file name
#             image_name = str(page_num)+'_'+ str(i) + '.' + image_ext
#             #Save image
#             with open(os.path.join(images_path, image_name) , 'wb') as image_file:
#                 image_file.write(image_bytes)
#                 image_file.close()
#     else:
#         pass
#### [End] PDF parsing help function ############################################3



##### [Start] OCR helper function ######################################################
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from tqdm import tqdm
def save_pdf_to_image(pdf_path, prefix):  # pdf page 전체를 이미지로 저장

    pages = convert_from_path(pdf_path) 
    print(f"convert_from_path: {pages}")
    print(f"prefix: {prefix}")
    folder_path = f"./images/{prefix}/"
    print(f"folder_path: {folder_path}")
    Path(folder_path).mkdir(parents=True, exist_ok=True)     
    for idx, page in enumerate(tqdm(pages)):
        page.save(f'./images/{prefix}/{prefix}_{idx}.png', 'PNG')
##### [End] OCR helper function ######################################################


###  [Start Main Class] ############################################################3
class CustomPdfParser(BaseLoader):
    def __init__(self) -> None:
        pass
    def pdf_parsing(self, file_path, crop:bool) -> Iterator[Document]:  # <-- Does not take any arguments
        full_result = []
        status = False
        prefix = file_path.split("/")[-1].split(".")[0].strip()
        with pdfplumber.open(file_path) as pdf1:
            page_number = 0
            print("----------- starting pdf parsing ----------- ")
            for _ in tqdm(pdf1.pages):
                page_result = block_based_parsing_by_page(file_path, page_number, crop)
                table_result = table_parser(file_path, page_number, crop)
                # image_files = image_extractor(file_path, page_number, prefix)
                # print(page_result)

                if page_result:  # page contents 없는 깡통 PDF 체크
                    status = True
                else: pass
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
            print("----------- completed pdf parsing ----------- ")

            print("----------- starting saving page images ----------- ")
            save_pdf_to_image(file_path, prefix)
            print("----------- Completed saving page images -----------")

            return status, full_result

    def ocr_parsing(self, pdf_path):
        '''
        ocr parsing은 table markdown 미포함 (깡통 파싱)
        '''
        prefix = pdf_path.split("/")[-1].split(".")[0].strip()

        print("----------- starting saving page images ----------- ")
        save_pdf_to_image(pdf_path, prefix)
        print("----------- Completed saving page images -----------")

        file_list = os.listdir(f"./images/{prefix}")
        selected_files = [file for file in file_list if file.endswith("png")]

        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        full_result = []
        for page_num, img_path in enumerate(tqdm(selected_files)):
            print(img_path)
            img_path = f"./images/{prefix}/"+img_path
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
###  [End Main Class] ############################################################3

if __name__ == "__main__":

    cpl = CustomPdfParser()
    pdf_path = "D:/ai_jarvis/data/FWG.pdf"
    prefix = "FWG"
    result = cpl.ocr_parsing(pdf_path, prefix)
    print(result)


   






