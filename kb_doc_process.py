from http import client
import pandas as pd
import json

# Core Libraries
import openai
import spacy
import torch

# NLP and Transformer Libraries
from transformers import AutoTokenizer, AutoModel

# Document Parsing Libraries
# from PyMuPDF
# import fitz  # For PDF parsing
from bs4 import BeautifulSoup  # For HTML parsing
import whisper  # For audio transcription

# Milvus Vector Database
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection,utility, MilvusClient

# -------------------- Milvus Setup --------------------

# Connect to Milvus
connections.connect(alias="default", host="127.0.0.1", port="19530")
print(f"Connected to Milvus: {connections.get_connection_addr('default')}")

print("Connected to Milvus!")

# Define schema for Milvus collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="metadata", dtype=DataType.JSON)
]


schema = CollectionSchema(fields, "Document embeddings and metadata storage")



# all_collections = connections.list_collections()
all_collections = utility.list_collections()

print(f"Collections: {all_collections}")


# Create or load collection
collection_name = "document_chunks"

# chekc collection is exist #
if utility.has_collection(collection_name):
    print(f"Collection '{collection_name}' exists.")
else:
    client.create_collection(
        collection_name="new_collection",

        dimension=768,  # The vectors we will use in this demo has 768 dimensions

    )

if collection_name not in utility.list_collections():
# if collection_name not in connections.list_collections():
    collection = Collection(name=collection_name, schema=schema)
    print(f"give me the collection name {collection}")
else:
    collection = Collection(name=collection_name)
print(f"Collection schema: {collection.schema}")


print(f"Collection '{collection_name}' exists: {utility.has_collection(collection_name)}")
print(f"Collections: {utility.list_collections()}")



index_params = {
    "index_type": "IVF_FLAT",  # Choose an index type: IVF_FLAT, IVF_SQ8, HNSW, etc.
    "metric_type": "L2",       # Metric type: L2 (Euclidean distance) or IP (Inner Product)
    "params": {"nlist": 128}   # Parameter specific to the index type
    # "params": {"nprobe": 10}
}

collection.create_index(field_name="embedding", index_params=index_params)
print(f"Index created: {collection.has_index()}")

############################### check data insertion #################

# Load collection into memory
# collection.load()
# print("Collection loaded successfully.")
try:
    collection.load()  # Attempt to load the collection
    print(f"Collection '{collection_name}' loaded successfully.")
except Exception as e:
    print(f"Error loading collection '{collection_name}': {e}")


if not collection.is_empty:
    print(f"Collection '{collection_name}' is not empty. Ready for queries.")
else:
    print(f"Collection '{collection_name}' is empty!")


# -------------------- Document Parsing Functions --------------------

# def parse_html(file_content):
#     """Extract text from an HTML document."""
#     print(f"file content is :{file_content}")
#     soup = BeautifulSoup(file_content, "html.parser")
#     return soup.get_text()

def parse_html(file_content):
    """Extract text from an HTML document."""
    if not file_content:
        print("Empty file content!")
        return ""
    print(f"Raw file content: {file_content}")

    try:
        soup = BeautifulSoup(file_content, "html.parser")
        print(f"Parsed text: {soup.get_text()}")
        return soup.get_text()
    except Exception as e:
        print(f"Error parsing HTML: {e}")
        return ""


# -------------------- Text Processing Functions --------------------

# Load spaCy and Tokenizer
nlp = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def clean_text(text):
    """Clean and tokenize text using spaCy."""
    doc = nlp(text)
    return " ".join([token.text for token in doc if not token.is_punct])


def split_text(text, chunk_size=300):
    """Split text into manageable chunks based on the tokenizer."""
    tokens = tokenizer.tokenize(text)
    chunks = [" ".join(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]
    return chunks


# -------------------- Embedding Model --------------------

# Load SentenceTransformer Model
embedding_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
embedding_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")


def get_embedding(text):
    """Generate embeddings for a given text."""
    inputs = embedding_tokenizer(text, return_tensors="pt", truncation=True)
    # print(f"input is : {input}")
    outputs = embedding_model(**inputs)
    # print(f"output is :{outputs}")
    embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()
    # print(f"Generated embedding for: {text}\n{embedding}")
  
    return embedding

# -------------------- Data Insertion Verification --------------------

def verify_data_insertion(collection):
    """Verify if data has been inserted into the Milvus collection."""
    try:
        # Query the collection for all documents
        results = collection.query(
            expr="id >= 0",  # Fetch all rows with id >= 0
            output_fields=["id", "metadata"]  # Specify fields to output
        )
        if results:
            print(f"Data successfully inserted. Retrieved {len(results)} records:")
            for record in results:
                print(record)
        else:
            print("No data found in the collection.")
    except Exception as e:
        print(f"An error occurred while querying the collection: {e}")



# -------------------- Vector Database Functions --------------------

def store_chunks_in_milvus(chunks, metadata):
    """Store text chunks and embeddings in Milvus."""
    print(f"Storing chunks: {chunks}")
    print(f"Storing metadata : {metadata}")

    # Generate embeddings for the chunks
    embeddings = [get_embedding(chunk) for chunk in chunks]

    # Prepare data to insert into Milvus (metadata should match the collection schema)
    metadata_list = [{"content": chunk, **metadata} for chunk in chunks]

    # Inserting the data into Milvus (only passing embeddings and metadata)
    try:
        collection.insert([embeddings, metadata_list])
        print("Data successfully inserted.")
    except Exception as e:
        print(f"Error during data insertion: {e}")

    # Check the number of entities in the collection
    num_entities = collection.num_entities
    print(f"Number of entities in collection '{collection_name}': {num_entities}")
    
    if num_entities > 0:
        print("Data successfully inserted.")
    else:
        print("Data insertion failed or no data found.")

    verify_data_insertion(collection)
    print("Chunks stored successfully in Milvus.")
    

def search_vector_db(query, top_k=5):
    """Search Milvus for relevant chunks."""
    query_embedding = get_embedding(query)

    # print(f"get querry embeding : {query_embedding}")
    print(f"get querry embeding : {query}")
    search_params = {"metric_type": "L2", "params": {"nlist": 128}}
    results = collection.search(
        [query_embedding.tolist()],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["metadata"]
    )
    print(f"Search results: {results}")
    chunks = []
    
    for hit in results[0]:  # Iterate through the search results (usually a list of hits)
        try:
            # Extract metadata based on Hit's structure
            print(f"the hi is:{hit}")
            metadata = hit.entity.metadata
            print(f"Hit metadata: {metadata}")

            if metadata:
                chunks.append(metadata)
                # chunks.append(metadata.get("content", "No content found"))
        except AttributeError as e:
            print(f"Error accessing metadata in hit: {e}")
     
    return chunks
    # return results


# -------------------- Document Ingestion --------------------

def prepare_metadata(doc_id, source_type, source_name):
    """Prepare metadata for a document."""
    return {
        "doc_id": doc_id,
        "source_type": source_type,
        "source_name": source_name
    }

def load_and_process_document_from_csv(csv_file_path):
    """Load and process documents from a CSV file row by row."""
    try:
        # Load CSV into a DataFrame
        # df = pd.read_csv(csv_file_path, on_bad_lines='skip')
        # print(f"CSV Data (First 5 Rows):\n{df.head()}")

        # # Check required columns
        # required_columns = ['id', 'title', 'editor_html', 'description']
        # if not all(col in df.columns for col in required_columns):
        #     print(f"Missing required columns. Expected: {required_columns}")
        #     return
        import csv
        with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            result = [row for row in csv_reader]
        print(result)
        # Process each row in the DataFrame
        for index, row in enumerate(result):
            doc_id = row.get('id')
            title = row.get('title')
            editor_html = row.get('editor_html')
            description = row.get('description')

            print(f"\nProcessing Row {index}: Document ID = {doc_id}, Title = {title}")

            # Skip rows with invalid `editor_html`
            if pd.isna(editor_html) or not str(editor_html).strip():
                print(f"Skipping row {index}: editor_html is empty or invalid.")
                continue

            # Debug: Print raw HTML
            print(f"Raw HTML:\n{editor_html}")

            # Parse HTML content
            html_content = parse_html(editor_html)
            if not html_content:
                print(f"Skipping row {index}: Unable to parse HTML.")
                continue

            # Clean and split text
            cleaned_text = clean_text(html_content)
            chunks = split_text(cleaned_text)

            # Prepare metadata
            metadata = prepare_metadata(doc_id=doc_id, source_type="html", source_name=title)

            # Debug: Print metadata and sample chunks
            print(f"Metadata for Row {index}:\n{metadata}")
            print(f"Sample Chunks for Row {index}:\n{chunks[:3]}...")

            # Store chunks in Milvus
            store_chunks_in_milvus(chunks, metadata)
            print(f"Row {index} processed and stored successfully.")

    except FileNotFoundError:
        print(f"The file at {csv_file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")



# -------------------- Retrieval Functions --------------------

def retrieve_chunks(query, top_k=5):
    """Retrieve relevant chunks from the vector database."""
    results = search_vector_db(query, top_k=top_k)
    return [res.entity.get("metadata")["content"] for res in results]


def retrieve_html_by_title(title, top_k=5):
    """Retrieve HTML content by title."""
    query = title
    print(f"Querying for title: {query}")
    results = search_vector_db(query, top_k=top_k)

    print(f"result is : {results}")
    for res in results:
        print(f"Result type: {type(res)}")  # Check the type of 'res' object
        if isinstance(res, dict):  # If the result is already a string
            metadata = res
            print(f"Metadata: {metadata}")
            if metadata and metadata.get("source_name") == title:
                return metadata["content"]
        else:
            print(f"Unexpected result: {res}")
    return None

# -------------------- Answer Generation --------------------

# Set OpenAI API Key
openai.api_key = "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"


def generate_answer(context, question):
    """Generate an answer using OpenAI GPT-3"""
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    # response = openai.Completion.create(
    response = openai.ChatCompletion.create(
        
        model="gpt-3.5-turbo",
        # model = "text-davinci-003",
        messages=prompt,
        max_tokens=500
    )

    return response.choices[0].text.strip()



# -------------------- RAG Pipeline --------------------

def rag_pipeline(query,top_k = 3):
    """Complete Retrieval-Augmented Generation (RAG) pipeline."""
    # Step 1: Retrieve relevant chunks
    chunks = retrieve_chunks(query, top_k=top_k)
    context = " ".join(chunks)

    # Step 2: Generate answer based on query and context
    answer = generate_answer(context, query)
    return answer


# -------------------- Usage Example --------------------

if __name__ == "__main__":
    # Example: Load and process CSV
    csv_path = '/home/pavankumarb/Documents/project/new-samples.csv'

    load_and_process_document_from_csv(csv_path)

    # Test query
    # query = "What is load balancing?"
    # print("RAG Pipeline Answer as per question:", rag_pipeline(query))

    # Retrieve HTML content by title
    title = "What is an Installment Loan?"
    html_content = retrieve_html_by_title(title)
    print("HTML Content:", html_content)
