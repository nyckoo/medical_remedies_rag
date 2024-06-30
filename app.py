from flask import Flask, Response, request
from flask_cors import CORS
from qdrant_manager import QdrantManager
from groq_manager import GroqManager

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
    qdrant_manager = QdrantManager(collection_name)
    answer_sections = qdrant_manager.make_query(query)
    qdrant_manager.close()
    groq_manager = GroqManager()
    groq_res = groq_manager.generate_response(context=answer_sections, question=query)
    return Response(groq_res.choices[0].message.content, 200)
