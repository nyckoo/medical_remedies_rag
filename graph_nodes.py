from typing import List
from typing_extensions import TypedDict
from langchain_community.tools.tavily_search import TavilySearchResults
from llm_chain_components import rag_chain, retrieval_grader, question_rewriter, answer_reviser
from qdrant_manager import QdrantManager


class GraphClass(TypedDict):
    """
    Represents the state of the graph.

    Attributes:
        question: user question
        collection_name: name of the vector database collection
        generation: LLM generation
        documents: list of documents
        query_correction_count: number of query transformation times
        web_search_docs: results of a web search
    """

    question: str
    collection_name: str
    generation: str
    documents: List[str]
    query_correction_count: int
    web_search_docs: List[str]


def retrieve(state):
    """
    Retrieve documents from knowledge base.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    collection_name = state["collection_name"]

    # Retrieval
    documents = QdrantManager(collection_name).make_query(question)
    # for idx, doc in enumerate(documents):
    #     print(idx+1, doc)
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for doc in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": doc}
        )
        if score.binary_score == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append({"content": doc["content"], "source": doc["ebook_chapter"]})

    return {"documents": filtered_docs, "question": question}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    query_correction_count = state["query_correction_count"] + 1

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    print("---IMPROVED QUESTION---")
    print(better_question)
    return {"question": better_question, "query_correction_count": query_correction_count}


def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates web_search_answer key with web results
    """

    print("---WEB SEARCH---")
    question = state["question"]

    # Web search
    docs = TavilySearchResults(k=3).invoke({"query": question})
    web_results = [{"content": doc["content"], "source": doc["url"]} for doc in docs]

    return {"web_search_docs": web_results, "question": question}


def revision(state):
    """
    Revision of an answer of conducted web search.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---ANSWER REVISION---")
    question = state["question"]
    documents = state["documents"]
    web_results = state["web_search_docs"]

    for web_doc in web_results:
        filtered_content = answer_reviser.invoke({"question": question, "answer": web_doc["content"]})
        documents.append({"content": filtered_content, "source": web_doc["source"]})
        print("---FILTERED WEB CONTENT---")
        print(filtered_content)

    return {"documents": documents}


# Edges
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")

    relevant_documents = len(state["documents"])
    query_correction_count = state["query_correction_count"]
    print(f"---RELEVANT DOCUMENTS: {relevant_documents}---")
    if relevant_documents > 2 > query_correction_count:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"
    elif query_correction_count == 2:
        # Too many attempts to transform query, so do a websearch
        print("---DECISION: WEB SEARCH---")
        return "web_search"
    else:
        # There are not enough documents after filtering, so regenerate a new query
        print("---DECISION: NOT ENOUGH DOCUMENTS ARE RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "transform_query"
