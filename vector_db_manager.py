import os
import uuid
import logging

from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer  # "sentence-transformers/multi-qa-mpnet-base-cos-v1"
from InstructorEmbedding import INSTRUCTOR
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException


class QdrantManager:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(url="http://localhost:6333")
        self.model = INSTRUCTOR('hkunlp/instructor-large')

    def create_vector_database(self):
        if not self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=768,
                    distance=models.Distance.COSINE,
                    datatype=models.Datatype.FLOAT32,  # UINT8 made vectors filled with zeros
                    on_disk=True
                )
            )

    def populate_vector_database(self, doc_name: str, doc_specifier: str):
        loader = TextLoader(f'./data_store/{doc_name}.md', encoding="utf-8")
        documents = loader.load()

        markdown_headers = [
            ("#", "subject"),
            ("##", "section"),
            ("###", "paragraph"),
        ]
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=markdown_headers,
            strip_headers=True
        )
        doc_chunks = [md_splitter.split_text(doc.page_content) for doc in documents]
        docs_chunks = [single_chunk for chunks in doc_chunks for single_chunk in chunks]
        doc_instruction = f"Represent the {doc_specifier} Natural remedies paragraph for retrieval: "
        try:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[models.PointStruct(
                    id=uuid.uuid4().hex,
                    payload={
                        "spec": f'{chunk.metadata.get("subject")} - '
                                f'{chunk.metadata.get("section", chunk.metadata.get("paragraph"))}',
                        "content": chunk.page_content,
                        "tags": chunk.metadata
                    },
                    vector=self.model.encode(
                        sentences=(f'{doc_instruction} """ {chunk.page_content} """')
                    )
                ) for chunk in docs_chunks]
            )
        except ResponseHandlingException as e:
            logging.exception(e.source)
        finally:
            logging.info(f"End of '{doc_name}' processing.")

    def make_query(self, query: str):
        query_instruction = "Represent the question for retrieving supporting documents: "
        np_vector = self.model.encode(f'{query_instruction} """ {query} """')
            #f'{query_instruction} """ {query} """')
        #vector = [float(vector) for vector in np_vectors]
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=np_vector,
            search_params=models.SearchParams(hnsw_ef=128, exact=True),
            limit=5
        )
        return [answer.payload for answer in results]

    def get_records_by_ids(self, recs_ids: list[str]):
        return self.qdrant_client.retrieve(self.collection_name, recs_ids, with_vectors=True)

    def close(self):
        self.qdrant_client.close()
