# Lab 9: Creating and using a RAG vector store 

# Chroma is an open-source vector database tailored to applications with large language models
# https://docs.trychroma.com/docs/overview/getting-started

# Chroma uses sentence transformer all-MiniLM-L6-v2 for a 384 vector embedding
# See more at https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 and
# https://docs.trychroma.com/docs/embeddings/embedding-functions 

# Run the two pip commands below at the terminal prompt before running the code for the first time
# pip install chromadb
# pip install pysqlite3-binary

query = "What plants might bring the most color to a landscape?"

import ast # python parser we use to turn a string to list, i.e., []
__import__('pysqlite3') #  module name determined at runtime (two commands below)
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3') # identifies the correct (vs default) sqlite3 version
import chromadb

string_of_facts = open("facts.txt", "r")  # r for read file into memory, consider instead: with open('file.txt', 'r') as file:
data = string_of_facts.read() 
list = ast.literal_eval(data)
print(data)

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="my_collection") # get if exists, else create

# Create embedding entries into vectorDB
collection.upsert( # upsert adds only if not already in list
    documents=list,
    ids=["id1", "id2", "id3"] # synch this with the number of entries in your list
)

# Embed query and search embeddings for most similar entry
# Chroma similarity default uses sum of squares of differences (taking the square would result in Euclidean distance)
# see more at https://docs.trychroma.com/docs/collections/configure#configuring-chroma-collections
results = collection.query(
    query_texts=[query], # Chroma will embed the query and then perform a similarity search
    n_results=1 # how many results to return, default is 10
)

print(results)
print('--------------------')
returned_message = results['documents'][0][0]
print("Query: ", query)
print("Response: ", returned_message)