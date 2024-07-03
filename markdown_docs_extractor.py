import re

from langchain_core.documents import Document as Langchain_Doc
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter


class MarkdownDocsExtractor:
    def __init__(self, input_file: str, char_eraser: str = None):
        self.loader = TextLoader(f'./data_store/{input_file}.md', encoding="utf-8")
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "subject"),
                ("##", "section"),
                ("###", "paragraph"),
            ],
            strip_headers=True
        )
        self.char_eraser = char_eraser

    def extract_docs_by_categories(self, categories_groups: dict[str, set]):
        documents = self.loader.load()
        doc_chunks_list = [self.md_splitter.split_text(doc.page_content) for doc in documents]
        doc_chunks = [single_chunk for chunks in doc_chunks_list for single_chunk in chunks]
        first_group_name = list(categories_groups.keys())[0]
        name_store = ""
        new_doc_contents = {}
        new_chunks = []
        for doc in doc_chunks:
            current_subject = doc.metadata.get("subject")
            section = doc.metadata.get("section")
            paragraph = doc.metadata.get("paragraph")
            doc_content = doc.page_content.replace(self.char_eraser, " ") if self.char_eraser else doc.page_content
            common_name = re.search(r'(?<=\*)(.*?)(?=\*)', current_subject).group(0)
            if paragraph:
                middle_text = " - " + paragraph
            else:
                middle_text = "; " + section if section else ""
            doc_title = common_name + middle_text + ":\n"
            if name_store == current_subject:
                for group_name, old_categories in categories_groups.items():
                    if section in old_categories:
                        new_doc_contents[group_name] += doc_title
                        new_doc_contents[group_name] += doc_content + "\n"
            else:
                for new_category, content in new_doc_contents.items():
                    new_chunks.append(Langchain_Doc(page_content=content, metadata={"subject": name_store, "section": new_category}))
                name_store = current_subject
                new_doc_contents.update({group_name: "" for group_name in categories_groups.keys()})
                new_doc_contents[first_group_name] += doc_title
                new_doc_contents[first_group_name] += doc_content + "\n"
        return new_chunks

    def extract_docs(self):
        documents = self.loader.load()
        doc_chunks_list = [self.md_splitter.split_text(doc.page_content) for doc in documents]
        doc_chunks = doc_chunks_list[0]
        titled_chunks = []
        for chunk in doc_chunks:
            subject = chunk.metadata.get("subject")
            section = chunk.metadata.get("section")
            paragraph = chunk.metadata.get("paragraph")
            if paragraph:
                middle_text = " - " + paragraph
            else:
                middle_text = "; " + section if section else ""
            titled_chunks.append(
                Langchain_Doc(page_content=subject + middle_text + ":\n" + chunk.page_content, metadata=chunk.metadata))
        return titled_chunks
