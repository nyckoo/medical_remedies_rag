from datetime import datetime
from pprint import pprint
from dotenv import load_dotenv

from vector_db_manager import QdrantManager

book_name = "Encyclopedia of Herbal Medicine_part"
history = book_name + "_1_history"
cultures = book_name + "_2_cultures"
key_medicinal_plants = book_name + "_3_key_medicinal_plants"
other_medicinal_plants = book_name + "_4_other_medicinal_plants"
cultivation_and_usage = book_name + "_5_cultivation_and_usage"


def doc_divider_util(name: str, part: int):
    split_name = name.split("_", 3)
    return f"{split_name[0]}_{split_name[1]}_{split_name[2]}.{part}_{split_name[3]}"


book_util_dict = {
    # history: "General history of",
    # cultures: "Cultural customs of",
    doc_divider_util(key_medicinal_plants, 1): "Properties of",
    doc_divider_util(key_medicinal_plants, 2): "Properties of",
    doc_divider_util(other_medicinal_plants, 1): "Features of",
    doc_divider_util(other_medicinal_plants, 2): "Features of",
    doc_divider_util(other_medicinal_plants, 3): "Features of",
    doc_divider_util(other_medicinal_plants, 4): "Features of",
    # cultivation_and_usage: "Use of",
}

if __name__ == "__main__":
    load_dotenv()
    vector_db_manager = QdrantManager("medical_herbs_rag_instructor_embeddings")

    # vector_db_manager.create_vector_database()
    # for book_chapter, doc_instr in book_util_dict.items():
    #     print(book_chapter, datetime.now())
    #     vector_db_manager.populate_vector_database(book_chapter, doc_instr)

    question = "What medicines should one prepare to prevent from a cold, flu or fever in autumn?"
    q = "What's the history of natural remedies usage in Europe between 1800 and 1900?"
    answers = vector_db_manager.make_query(question)
    for a in answers:
        pprint(a.content)

    # results = vector_db_manager.get_records_by_ids(["00eb5774-f567-4b3a-9e7e-b39e4b301804"])
    # for x in results:
    #     print(x)
    vector_db_manager.close()
