from datetime import datetime
from qdrant_manager import QdrantManager
from pdf_parser_utils import name_divider_util

book_name = "Encyclopedia of Herbal Medicine_part"
history = book_name + "_1_history"
cultures = book_name + "_2_cultures"
key_medicinal_plants = book_name + "_3_key_medicinal_plants"
other_medicinal_plants = book_name + "_4_other_medicinal_plants"
cultivation_and_usage = book_name + "_5_cultivation_and_usage"


book_util_dict = {
    history: "General history of",
    cultures: "Cultural customs of",
    # name_divider_util(key_medicinal_plants, 1): "Properties of",
    # name_divider_util(key_medicinal_plants, 2): "Properties of",
    # name_divider_util(other_medicinal_plants, 1): "Features of",
    # name_divider_util(other_medicinal_plants, 2): "Features of",
    # name_divider_util(other_medicinal_plants, 3): "Features of",
    # name_divider_util(other_medicinal_plants, 4): "Features of",
    cultivation_and_usage: "Use of",
}

if __name__ == "__main__":
    start_model = datetime.now()
    qdrant_cm = QdrantManager("medical_herbs_rag_instructor_embeddings")
    end_model = datetime.now()
    print("model_init:", (end_model - start_model).microseconds)

    qdrant_cm.create_vector_collection()
    for doc_name, doc_instr in book_util_dict.items():
        doc_start_timestamp = datetime.now()
        qdrant_cm.populate_vector_collection(doc_name, doc_instr)
        print(doc_name, (datetime.now() - doc_start_timestamp).seconds)

    # questions = [
    #     "What home-made medicines should one prepare to prevent from a cold, flu or fever in autumn?",
    #     "What's the history of natural remedies usage in Europe between 1800 and 1900?"
    # ]
    # middle = datetime.now()
    # answers = vector_db_manager.make_query(questions[1])
    # for idx, a in enumerate(answers):
    #     print(f"{idx+1}: \n", a["content"], '\n', a["spec"])

    # results = vector_db_manager.get_records_by_ids(["00eb5774-f567-4b3a-9e7e-b39e4b301804"])
    # for x in results:
    #     print(x)

    qdrant_cm.close()
