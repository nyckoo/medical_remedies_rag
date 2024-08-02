from langgraph.graph import START, END, StateGraph
from graph_nodes import GraphClass, retrieve, grade_documents, generate, transform_query, web_search, revision, \
    decide_to_generate

workflow = StateGraph(GraphClass)

# Define the nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search", web_search)
workflow.add_node("revision", revision)

# Build graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
        "web_search": "web_search"
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("web_search", "revision")
workflow.add_edge("revision", "generate")
workflow.add_edge("generate", END)

# Graph Initiation
agent = workflow.compile()

inputs = {
    "question": "What are the properties of passionflower plant and what can I use it for as to healing aspects?",
    "collection_name": "medical_herbs_rag_instructor_embeddings",
    "query_correction_count": 0
}

result = agent.invoke(inputs)
