import os
import uuid
import logging

from InstructorEmbedding import INSTRUCTOR
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException


class QdrantManager:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_DB_URL"),
            api_key=os.getenv("QDRANT_KEY")
        )
        self.model = INSTRUCTOR(os.getenv("INSTRUCTOR_LOCAL_PATH"))

    def create_vector_collection(self):
        if not self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=768,
                    distance=models.Distance.COSINE,
                    datatype=models.Datatype.FLOAT32,  # UINT8 made vectors filled with zeros (too little precision)
                )
            )

    def populate_vector_collection(self, doc_name: str, doc_specifier: str, docs_chunks: list):
        doc_instruction = f"Represent the {doc_specifier} Natural remedies paragraph for retrieval: "
        try:  # upsert has a limit of processing files with +5000 lines & may return an error
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[models.PointStruct(
                    id=uuid.uuid4().hex,
                    payload={
                        "ebook_chapter": doc_name,
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

    def make_query(self, query: str):
        query_instruction = "Represent the question for retrieving supporting documents: "
        np_vector = self.model.encode(f'{query_instruction} """ {query} """')
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=np_vector,
            search_params=models.SearchParams(hnsw_ef=128, exact=True),
            score_threshold=0.8,
            limit=10
        )
        return "[" + "], [".join(map(str, [answer.payload for answer in results])) + "]"

    def get_records_by_ids(self, recs_ids: list[str]):
        return self.qdrant_client.retrieve(self.collection_name, recs_ids, with_vectors=True)

    def close(self):
        self.qdrant_client.close()
