import logging
from datetime import datetime

from qdrant_manager import QdrantManager
from markdown_docs_extractor import MarkdownDocsExtractor
from pdf_parser_utils import name_divider_util

book_name = "Encyclopedia of Herbal Medicine_part"
history = book_name + "_1_history"
cultures = book_name + "_2_cultures"
key_medicinal_plants = book_name + "_3_key_medicinal_plants"
other_medicinal_plants = book_name + "_4_other_medicinal_plants"
cultivation_and_usage = book_name + "_5_cultivation_and_usage"

book_util_dict = {
    history: ("General history of", "DEFAULT_EXTRACTION", None),
    cultures: ("Cultural customs of", "DEFAULT_EXTRACTION", None),
    name_divider_util(key_medicinal_plants, 1): ("Properties of", "CATEGORIES_EXTRACTION", None),
    name_divider_util(key_medicinal_plants, 2): ("Properties of", "CATEGORIES_EXTRACTION", None),
    name_divider_util(other_medicinal_plants, 1): ("Features of", "CATEGORIES_EXTRACTION", "\n"),
    name_divider_util(other_medicinal_plants, 2): ("Features of", "CATEGORIES_EXTRACTION", "\n"),
    name_divider_util(other_medicinal_plants, 3): ("Features of", "CATEGORIES_EXTRACTION", "\n"),
    name_divider_util(other_medicinal_plants, 4): ("Features of", "CATEGORIES_EXTRACTION", "\n"),
    cultivation_and_usage: ("Use of", "DEFAULT_EXTRACTION", None),
}

groups_mapper = {
        key_medicinal_plants: {
            "Medicine characteristics": {"Habitat & Cultivation", "Research", "Related Species", "Parts Used"},
            "Use": {"Key Constituents", "Key actions", "Traditional & Current Uses", "Key Preparations & Their Uses"}
        },
        other_medicinal_plants: {
            "Medicine characteristics": {"Habitat & Cultivation", "Description", "Related Species", "Habitat & Cultivation",
                                         "Parts Used", "Part Used", "Research"},
            "Use": {"Constituents", "Medicinal Actions & Uses", "Caution", "Cautions", "Self-help Use", "Self-help Uses"}
        }
    }

if __name__ == "__main__":
    qdrant_manager = QdrantManager("medical_herbs_rag_instructor_embeddings")
    qdrant_manager.create_vector_collection()
    for doc_name, doc_instr in book_util_dict.items():
        doc_start_timestamp = datetime.now()
        md_docs_extractor = MarkdownDocsExtractor(
            input_file=doc_name,
            char_eraser=doc_instr[2]
        )
        if doc_instr[1] == "DEFAULT_EXTRACTION":
            doc_chunks = md_docs_extractor.extract_docs()
        else:
            doc_name_compact = doc_name[:38]+doc_name[40:]
            doc_chunks = md_docs_extractor.extract_docs_by_categories(groups_mapper[doc_name_compact])
        qdrant_manager.populate_vector_collection(
            doc_name=doc_name,
            doc_specifier=doc_instr[0],
            doc_chunks=doc_chunks
        )
        logging.log(logging.INFO, f"{doc_name}: {(datetime.now() - doc_start_timestamp).seconds}")
    qdrant_manager.close()
