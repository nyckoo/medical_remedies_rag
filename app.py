import json
from flask import Flask, Response, request
from flask_cors import CORS
from graph_workflow import workflow


app = Flask(__name__)
cors = CORS(
    app,
    origins="*",
    allow_headers="*",
)


@app.route("/is_running", methods=['GET'])
def is_running():
    return Response("Running!", 200)


@app.route("/get_answer/<query>", methods=['GET'])
def get_answer(query: str):
    collection_name = request.headers.get("Collection-Name")
    inputs = {
        "question": query,
        "collection_name": collection_name,
        "query_correction_count": 0,
        "last_iteration_state": {}
    }
    agent = workflow.compile()
    results = agent.invoke(inputs)

    return Response(json.dumps({
        "generation": results["generation"],
        "documents": results["documents"]
    }), 200)
