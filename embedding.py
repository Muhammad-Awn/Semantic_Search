from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import sys

# Suppress future warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

query = sys.argv[1]

# Convert query text to embedding
query_vector = model.encode(query).tolist()

# Print the vector as a JSON string
# print(query_vector)

# api_key = os.getenv('API_KEY')

# Initialize Pinecone
pc = Pinecone(api_key="e6016e2c-2539-4f49-820b-f077fcb07cc1")
spec = ServerlessSpec(cloud='aws', region='us-east-1')
index_name = 'semantic-search'

# Check if the index exists
if index_name not in pc.list_indexes().names():
    print(f"Index '{index_name}' does not exist.")
else:
    pinecone_index = pc.Index(index_name)

response = pinecone_index.query(vector=query_vector, top_k=5, include_metadata=True)

matches = response.get('matches', [])

for result in matches:
    if result is None:
        print('A result in matches is None')
    else:
        # Initialize an empty list to store the formatted dictionaries
        final_output = []

        # Extract and format text values
        for match in matches:
            text = match['metadata']['text']
            final_output.append({'text': text})


print(final_output)
