import os
import re
from llama_parse import LlamaParse
from pymupdf import Document, Rect


def name_divider_util(name: str, part: int):
    split_name = name.split("_", 3)
    return f"{'_'.join(split_name[:3])}.{part}_{split_name[3]}"


def transform_text_to_markdown(text: str):
    # Define the expressions to be marked with double hash (excluding "Research" and "Description")
    double_hash_terms = [
        "Habitat & Cultivation", "Parts Used", "Constituents", "History & Folklore", "Research", "Part Used",
        "Medicinal Actions & Uses", "Cautions", "Caution", "Related Species", "Self-help Use",
        "RCautions", "RCaution", "QCaution", "QCautions", "RQCaution", "RQCautions",  # <- read caution alternations
    ]
    # Create regex pattern for the double hash terms
    double_hash_pattern = r'\b(?:' + '|'.join(re.escape(term) for term in double_hash_terms) + r')\b'
    # Add single hash before plant names (text between start and 'Description')
    # (?<=\n)(.*?)(?= Description)
    d_str = "Description"
    d_pattern = r'\n*([^.]*)\bDescription\b'
    s = re.sub(d_pattern, lambda m: '\n# ' + m.group().replace('\n', ' ').replace(d_str, "\n## " + d_str + "\n"), text)
    # Add double hash before each term in the double hash list (excluding "Description")
    s = re.sub(double_hash_pattern, r'## \g<0>\n', s)
    return s


def load_document_by_llama_parse(pdf_source_path: str, output_file_path: str, parsing_instruction: str, use_gpt4o: bool) -> str:
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r', encoding="utf-8") as f:
            doc = f.read()
    else:
        parser = LlamaParse(
            api_key=os.environ["LLAMA_CLOUD_KEY"],
            result_type="markdown",
            parsing_instruction=parsing_instruction,
            gpt4o_mode=use_gpt4o,
            max_timeout=5000,
            ignore_errors=False
            )
        llama_parse_doc = parser.load_data(pdf_source_path)[0]
        doc = llama_parse_doc.get_content()
    return doc


def load_document_by_pymupdf(pdf_source_path: str, output_file_path: str) -> str:
    # 3 columns starting/ending coordinates
    # col_separators = [40, 230, 420, 610]
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r', encoding="utf-8") as f:
            doc = f.read()
    else:
        pdf_doc = Document(pdf_source_path)
        doc = ""
        with open(output_file_path, 'a+', encoding="utf-8") as f:
            for page in pdf_doc.pages():
                page_text = "\n"
                page_text += page.get_textbox(Rect(40, 30, 230, 755)) + "\n"
                page_text += page.get_textbox(Rect(230, 30, 420, 755)) + "\n"
                page_text += page.get_textbox(Rect(420, 30, 610, 755))
                f.write(page_text)
                doc += page_text
    return doc


def load_multiple_doc_pages_data(names: list[str], merge_docs=False):
    docs = []
    merged_doc_name = "_".join(names[0][:3]) + '_' + names[0][4]
    for name in names:
        split_name = name.split("_", 4)
        pages = split_name[3][1:].split("-")
        del split_name[3]
        pages_part = f"_p{pages[0]}-{pages[1]}_"
        file_name = "_".join(split_name[:3]) + pages_part + split_name[3]
        if os.path.exists(f"./data_store/{file_name}.md"):
            with open(f"./data_store/{file_name}.md", 'r', encoding="utf-8") as f:
                doc = f.read()
        else:
            doc = load_document_by_pymupdf(f"./raw_docs/{file_name}.pdf", f"./data_store/{file_name}.txt")
            output = transform_text_to_markdown(doc)
            with open(f"./data_store/{file_name}.md", 'w', encoding="utf-8") as f:
                f.write(output)
        docs.append(doc)
    if merge_docs:
        with open(f"./data_store/{merged_doc_name}.md", 'w', encoding="utf-8") as f:
            for doc in docs:
                f.write(doc+"\n")
    return docs


def load_single_doc_data(part_name: str, parsing_instr=""):
    if os.path.exists(f"./data_store/{part_name}.md"):
        with open(f"./data_store/{part_name}.md", 'r', encoding="utf-8") as f:
            doc = f.read()
    else:
        doc = load_document_by_llama_parse(f"./raw_docs/{part_name}.pdf", f"./data_store/{part_name}.txt", parsing_instr, True)
        output = transform_text_to_markdown(doc)
        with open(f"./data_store/{part_name}.md", 'w', encoding="utf-8") as f:
            f.write(output)
    return doc


parsing_instructions = {  # llama_parse instruction for different chapters & text structures
    "general_info": """
        The document is a collection of natural medicines knowledge.
         I want it to be parsed into markdown format by following rules:
    """,
    "multiple_columns_instruction": """
        Read the whole content within green vertical lines by text blocks structured in columns, start from left column,
         read to the bottom, then switch to another column on the right. Ignore images and their captions while parsing.
    """,
    "single_column_instruction": """
        Read content within green vertical lines by text blocks, skip text pieces placed out of green vertical lines.
    """,
    "plants_section_instruction": """
        Read content within green vertical lines by text blocks, start from left column, read to the bottom,
         then switch to another column on the right. Submit the content found in green rectangular frame at the end.
    """,
    "others_section_instruction": """
        Ignore botanical name written in capital letters located inside the green rectangle at the top of each page.
         Read content from three columns; start from the left one and read to the bottom, then go to another column 
         in the middle and repeat with the third one on the right. If you encounter an image, skip it and read further
         content below that will belong to started plant's section.
    """,
    "output_desirables": """
        As to markdown output file, save text blocks' headers with single hash mark and paragraphs' names with double
         hash mark. Don't include any HTML and CSS.
    """
}

standard_instruction = f"{parsing_instructions['general_info']} " \
                  f"{parsing_instructions['multiple_columns_instruction']} " \
                  f"{parsing_instructions['output_desirables']}"

single_column_instruction = f"{parsing_instructions['general_info']} " \
              f"{parsing_instructions['single_column_instruction']} " \
              f"{parsing_instructions['output_desirables']}"

plants_section_instruction = f"{parsing_instructions['general_info']} " \
              f"{parsing_instructions['plants_section_instruction']} " \
              f"{parsing_instructions['output_desirables']}"

others_section_instruction = f"{parsing_instructions['general_info']} " \
              f"{parsing_instructions['others_section_instruction']} " \
              f"{parsing_instructions['output_desirables']}"
