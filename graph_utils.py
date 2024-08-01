from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from llm_chain_components import rag_chain, retrieval_grader, question_rewriter
from qdrant_manager import QdrantManager


class GraphClass(TypedDict):
    """
    Represents the state of the graph.

    Attributes:
        question: user question
        collection_name: name of the vector database collection
        generation: LLM generation decision
        documents: list of documents
        query_correction_count: number of query transformation times
    """

    question: str
    collection_name: str
    generation: str
    documents: List[str]
    query_correction_count: int


def retrieve(state):
    """
    Retrieve documents

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
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
    if len(filtered_docs) < 3:
        tavily_search = "Yes"
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
    documents = state["documents"]
    query_correction_count = state["query_correction_count"] + 1

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    print("---IMPROVED QUESTION---")
    print(better_question)
    return {"documents": documents, "question": better_question, "query_correction_count": query_correction_count}


def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    docs = TavilySearchResults(k=3).invoke({"query": question})
    web_results = "\n".join([doc["content"] for doc in docs])
    documents.append(web_results)

    return {"documents": documents, "question": question}


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

    relevant_documents = state["documents"]
    query_correction_count = state["query_correction_count"]

    if relevant_documents > 2 and query_correction_count < 3:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"
    elif query_correction_count > 2:
        # Too many attempts to transform query, so do a websearch
        print("---DECISION: WEB SEARCH---")
        return "web_search_node"
    else:
        # There are not enough documents after filtering, so regenerate a new query
        print("---DECISION: NOT ENOUGH DOCUMENTS ARE RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "transform_query"
