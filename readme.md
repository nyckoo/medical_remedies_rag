## Brief introduction

This project is of CRAG type utilizing neural search wrapped in Flask API & is about helping users to find natural medications data <br> 
like history, cultivation, preparation, use etc. <br>
It's based on the **Encyclopedia Of Herbal Medicine** from following link:
http://repo.upertis.ac.id/1889/1/Encyclopedia%20Of%20Herbal%20Medicine.pdf

The final version makes it an API-wrapped CRAG agent searching for relevant documents with utilities of rephrasing the question and supplying insufficient results with web search. <br>
Here's the model of the agent:
![crag_natural_remedies_graph][agent_graph]

[agent_graph]: https://github.com/user-attachments/assets/4dcb5162-f899-4a68-8593-171191771610 "Agent Graph Model"
## Project phases

### 1. ETL

To begin with, I divided the pdf to 5 parts based on chapters and their blocks text structure:
1. History (p. 12-33)
2. Cultures (p. 34-55)
3. Key medicinal plants (p. 58-157)
4. Other medicinal plants (p. 160-285)
5. Cultivation and usage (p. 288-322)

I used [LlamaParse](https://github.com/run-llama/llama_parse) to read each chapter except for 4th (explanation later on), even though it didn't always work well with text blocks continuing to next columns or pages.
Anyway it kept the structure more or less in a consistent way, so I did manual corrections after parsing.

Example of LLamaParse instruction used for 3rd book part:
```
The document is a collection of natural medicines knowledge.  
I want it to be parsed into markdown format by following rules:
Read content within green vertical lines by text blocks, start from left column, read to the bottom,  
then switch to another column on the right. Submit the content found in green rectangular frame at the end.
As to markdown output file, save text blocks' headers with single hash mark and paragraphs' names with double  
hash mark. Don't include any HTML and CSS.
```

For the 4th part I couldn't make the parser to read 3 columns of text blocks so that it reads full text without omitting some parts - I did an explanation on that in [my post on Issues](https://github.com/run-llama/llama_parse/issues/229).
As a solution I used [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/) and explicitly specified coordinates of text columns to be ordered as expected. After cleaning all redundant data of images, links etc. I made the final markdown files follow the same structure with headers, so it's prepared for splitting data into chunks.

Function handling the 4th chapter: *(pdf_parser_utils.py)*
```
def load_document_by_pymupdf(pdf_source_path: str, output_file_path: str) -> str:  
	# 3 columns starting/ending coordinates  
	# col_separators = [40, 230, 420, 610]  
	if os.path.exists(output_file_path):  
		with open(output_file_path, 'r', encoding="utf-8") as f:  
			doc = f.read()  
	else:  
		pdf_doc = Document(pdf_source_path)  
		doc = ""  
		with open(output_file_path, 'a+', encoding="utf-8") as f:  
			for page in pdf_doc.pages():  
				page_text = "\n"  
				page_text += page.get_textbox(Rect(40, 30, 230, 755)) + "\n"  
				page_text += page.get_textbox(Rect(230, 30, 420, 755)) + "\n"  
				page_text += page.get_textbox(Rect(420, 30, 610, 755))  
				f.write(page_text)  
				doc += page_text  
	return doc
```
### 2. Embeddings & Vector DB

I decided to use [Instructor model](https://instructor-embedding.github.io/) to create embeddings, because with specific description given to each text chunk it adds another layer of category understanding to what's its purpose it the whole dataset. It clears a bit the clutter of many self-references inside the book.

My implementation of chunks' specifications to Instructor model:
> Embedding:<br>
> `f"Represent the {doc_specifier} Natural remedies paragraph for retrieval: "`<br>
> Doc specifiers to corresponding chapters:<br>
> 1.`General history of`<br>
> 2.`Cultural customs of` <br>
> 3.`Properties of`<br>
> 4.`Features of`<br>
> 5.`Use of`<br>
> Query:<br>
> `"Represent the question for retrieving supporting documents: "`<br>

I chose [Qdrant](https://qdrant.tech/) as a vector database and populated the collection. To do so I had to divide large markdown files to be less than 5000 lines of text because of experiencing TimeoutError otherwise with bulk loading.

Inside of Vector DB populating method: *(qdrant_manager.py)*
```
self.qdrant_client.upsert(  
	collection_name=self.collection_name,  
	points=[models.PointStruct(  
		id=uuid.uuid4().hex,  
		payload={  
			"ebook_chapter": doc_name,  
			"content": chunk.page_content,  
			"tags": chunk.metadata  
		},  
		vector=self.model.encode(  
			sentences=(f'{doc_instruction} """ {chunk.page_content} """')  
		)  
	) for chunk in doc_chunks]  
)
```
### 3. Creating Neural search query with Qdrant

Then I made a simple query making method to vector database for retrieval based on cosine similarity. 
Initially the results from neural search were interpreted as an additional query with Ollama 3 model to finally present relevant answer to the question in a neat form.

Neural search query: *(qdrant_manager.py)*
```
def make_query(self, query: str):  
	query_instruction = "Represent the question for retrieving supporting documents: "  
	np_vector = self.model.encode(f'{query_instruction} """ {query} """')  
	results = self.qdrant_client.search(  
	    collection_name=self.collection_name,  
	    query_vector=np_vector,  
	    search_params=models.SearchParams(hnsw_ef=128, exact=True),  
	    score_threshold=0.8,  
	    limit=5  
	)  
	return "[" + "], [".join(map(str, [answer.payload for answer in results])) + "]"
```

### 4. CRAG agent development

After time, I decided to turn this project into more advanced, CRAG agent using langchain with state management state. 
Now the agent works by implemented graph behaviour using llm chains. The graph is [right at the top][#brief-introduction].

Workflow implementation: *(graph_workflow.py)*, <br>
Nodes implementation: *(graph_nodes.py)*, <br>
LLM chains implementation: *(llm_chain_components.py)* <br>

### 5. Flask API

Lastly there's an API endpoint wrapper around all of agent's actions, for every request there's new agent object created from compiled graph workflow.
The endpoint returns JSON object with 'generation' and 'documents' keys, that include LLM answer and all supported documents for a user query with content and sources.

CRAG app endpoint: *(app.py)*
```
@app.route("/get_answer/<query>", methods=['GET'])
def get_answer(query: str):
    collection_name = request.headers.get("Collection-Name")
    inputs = {
        "question": query,
        "collection_name": collection_name,
        "query_correction_count": 0
    }
    agent = workflow.compile()
    results = agent.invoke(inputs)

    return Response(json.dumps({
        "generation": results["generation"],
        "documents": results["documents"]
    }), 200)
```