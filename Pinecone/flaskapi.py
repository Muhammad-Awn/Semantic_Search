from flask import Flask, request, jsonify
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import logging
import os

app = Flask(__name__)

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
api_key = os.getenv('API_KEY')

# Initialize Pinecone
pc = Pinecone(api_key=api_key)
spec = ServerlessSpec(cloud='aws', region='us-east-1')
index_name = 'semantic-search'

# Check if the index exists
if index_name not in pc.list_indexes().names():
    app.logger.error(f"Index '{index_name}' does not exist.")
else:
    pinecone_index = pc.Index(index_name)

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the endpoint for the semantic search
@app.route('/', methods=['GET'])
def semantic_search():
    try:
        # query = request.json.get('query')
        # if not query:
        #     return jsonify({"error": "Query text is missing"}), 400
        
        # app.logger.debug(f"Received query: {query}")

        # query = "Do not go astray"

        query = request.args['query']

        # Convert query text to embedding
        query_vector = model.encode(query).tolist()
        
        # Perform the search
        response = pinecone_index.query(vector=query_vector, top_k=5, include_metadata=True)

        matches = response.get('matches', [])

        for result in matches:
            if result is None:
                app.logger.debug('A result in matches is None')

        # Initialize an empty list to store the formatted dictionaries
        final_output = []

        # Extract and format text values
        for match in matches:
            text = match['metadata']['text']
            final_output.append({'text': text})

        app.logger.debug(final_output, {})

        return jsonify(final_output), 200

    except Exception as e:
        app.logger.error(f"An error occurred: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True)
