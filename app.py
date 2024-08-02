import json
from flask import Flask, Response, request
from flask_cors import CORS
from graph_workflow import agent

app = Flask(__name__)
app.debug = True
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
        "query_correction_count": 0
    }
    results = agent.invoke(inputs)

    return Response(json.dumps({
        "generation": results["generation"],
        "documents": results["documents"]
    }), 200)
