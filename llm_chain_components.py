import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser

# RAG chain
rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that loves sharing knowledge about medical plants. "
                   "Given the following sections in brackets from an encyclopedia, answer the question using "
                   "only that information, outputted in markdown format. If you are unsure and answer is not "
                   "explicitly written in given sections, say 'Sorry, I don't know how to help with that question.'\n"
                   "Context sections: \n"
                   "{context}"
         ),
        ("user", "{question}")
    ]
)

llm_rag = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.0,
    max_retries=2,
    api_key=os.environ.get("GROQ_KEY")
)

rag_chain = rag_prompt | llm_rag | StrOutputParser()


# Grader chain
class DocsGrader(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


llm_grader = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.0,
    max_retries=2,
    api_key=os.environ.get("GROQ_KEY")
).with_structured_output(DocsGrader)

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | llm_grader

# Corrective chain
llm_rewriter = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.0,
    max_retries=2,
    api_key=os.environ.get("GROQ_KEY")
)

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You a question re-writer that converts an input question to a better version that is optimized \n 
            for web search. Look at the input and try to reason about the underlying semantic intent / meaning. \n
            Please include nothing else than improved version of the question in the output."""),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = rewrite_prompt | llm_rewriter | StrOutputParser()

# Web search revision chain
llm_reviser = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.0,
    max_retries=2,
    api_key=os.environ.get("GROQ_KEY")
)

revision_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """You're an answer reviser that filters relevant data from a piece of text based on given question. \n 
            Look at the answer and try to select only those parts that correspond to the received question. \n
            Please return nothing else than the found contents in given text as an output."""),
        (
            "human",
            "Here is the question: \n\n {question} \n --- \n And the answer: \n\n {answer}",
        ),
    ]
)

answer_reviser = revision_prompt | llm_reviser | StrOutputParser()
