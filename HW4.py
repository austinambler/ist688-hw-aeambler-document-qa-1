import streamlit as st
from openai import OpenAI
import sys

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from pathlib import Path
from bs4 import BeautifulSoup
import numpy as np
import re
import os

if 'openai_client' not in st.session_state:
    api_key = st.secrets["OPEN_AI_KEY"]
    st.session_state.openai_client = OpenAI(api_key=api_key)

# Create ChromaDB client
chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_HW')
collection = chroma_client.get_or_create_collection('HW4Collection')


# --- Extract text from HTML ---
def extract_text_from_html(html_path):
    with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
        soup = BeautifulSoup(f, 'html.parser')

    # remove script/style tags — they add noise, not content
    for tag in soup(['script', 'style']):
        tag.decompose()

    text = soup.get_text(separator=' ', strip=True)
    return text


# --- Split text into sentences ---
def split_into_sentences(text):
    # simple sentence splitter; good enough for most prose
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


# --- Semantic chunking ---
# I decided to use semantic chunking as it is a good middle ground between fixed size and ML chunking.
# This allows there to be some meaning in what is getting stored, while there is no meaning in the fixed size.
# This does take longer to run than fixed size, but it is quicker than ML, so it makes a good middle ground.
def semantic_chunk(text, client, similarity_threshold=0.75, max_chunk_chars=2000):
    sentences = split_into_sentences(text)

    if len(sentences) <= 1:
        return [text] if text.strip() else []

    # embed all sentences in one batched call
    response = client.embeddings.create(
        input=sentences,
        model='text-embedding-3-small'
    )
    sentence_embeddings = [np.array(item.embedding) for item in response.data]

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    chunks = []
    current_chunk = [sentences[0]]
    current_length = len(sentences[0])

    for i in range(1, len(sentences)):
        sim = cosine_sim(sentence_embeddings[i - 1], sentence_embeddings[i])
        sentence_len = len(sentences[i])

        # start a new chunk if topic shifts OR chunk is getting too long
        if sim < similarity_threshold or (current_length + sentence_len) > max_chunk_chars:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentences[i]]
            current_length = sentence_len
        else:
            current_chunk.append(sentences[i])
            current_length += sentence_len

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


# --- Add chunks to collection ---
def add_to_collection(collection, chunks, file_name, client):
    if not chunks:
        return

    # batch-embed all chunks for this file in one API call
    response = client.embeddings.create(
        input=chunks,
        model='text-embedding-3-small'
    )
    embeddings = [item.embedding for item in response.data]

    ids = [f"{file_name}_chunk{i}" for i in range(len(chunks))]
    metadatas = [{"source": file_name, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas
    )


# --- Populate collection with HTML files ---
def load_html_to_collection(folder_path, collection):
    loaded_files = []
    client = st.session_state.openai_client

    html_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.html', '.htm'))]
    total = len(html_files)

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, file_name in enumerate(html_files):
        html_path = os.path.join(folder_path, file_name)
        text = extract_text_from_html(html_path)

        if text.strip():
            chunks = semantic_chunk(text, client)
            add_to_collection(collection, chunks, file_name, client)
            loaded_files.append(file_name)
        else:
            print(f"Warning: no text extracted from {file_name}")

        progress_bar.progress((idx + 1) / total)
        status_text.text(f"Processed {idx + 1}/{total}: {file_name}")

    status_text.text(f"Done! Loaded {len(loaded_files)} files.")
    return loaded_files


# Check if collection is empty and load HTML files
if collection.count() == 0:
    loaded = load_html_to_collection('./su_orgs', collection)


    
st.title("My HW4 Chatbot Using RAG")





openAI_model = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a helpful assistant. Follow this conversation pattern strictly:

- Use simple, everyday words and short sentences.
- Avoid jargon and technical terms; if you must use one, explain it simply right after.

You will be given "Reference material" pulled from HTML documents, along with the user's question.
- If the reference material is relevant, base your answer on it, and briefly mention that you're drawing from the course documents (e.g., "Based on the course materials...").
- If the reference material doesn't help answer the question, rely on your own knowledge instead and don't force a connection.
- Never pretend information came from the documents if it didn't.

1. Wait for the user to ask a question.
2. Answer the question clearly and concisely.
3. After answering, ask: "Do you want more info?"
4. If the user says yes (or anything affirmative):
   - Provide additional, more detailed information on the same topic.
   - Then ask again: "Do you want more info?"
   - Repeat this loop for as long as the user keeps saying yes.
5. If the user says no (or anything negative):
   - Respond with something like "Sounds good!" and then ask: "What can I help you with?"
   - Wait for a new question and start the pattern over from step 1.

Always keep track of the current topic so that "more info" responses stay relevant to the original question, until the user moves on to a new question.
"""

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": "What can I help you with?"}
    ]

for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

# set buffer limit
max_messages = 5

def get_buffered_messages(messages, max_messages=max_messages):
    system_msgs = [m for m in messages if m["role"] == "system"]
    user_idxs = [i for i, m in enumerate(messages) if m["role"] == "user"][-max_messages:]
    assistant_idxs = [i for i, m in enumerate(messages) if m["role"] == "assistant"][-max_messages:]

    keep_idxs = sorted(set(user_idxs + assistant_idxs))
    trimmed = [messages[i] for i in keep_idxs]

    return system_msgs + trimmed

# retrieve only the most relevant chunks for this question
def get_relevant_context(query, collection, client, n_results=5):
    response = client.embeddings.create(
        input=query,
        model='text-embedding-3-small'
    )
    query_embedding = response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    docs = results['documents'][0]
    ids = results['ids'][0]

    return docs, ids

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    client = st.session_state.openai_client

    # retrieve just the top few relevant chunks, not everything
    docs, ids = get_relevant_context(prompt, collection, client, n_results=5)

    context_text = "\n\n---\n\n".join(
        f"[Source: {doc_id}]\n{doc_text}"
        for doc_id, doc_text in zip(ids, docs)
    )

    augmented_prompt = f"""Reference material from course documents:

{context_text}

User's question: {prompt}"""

    buffered_messages = get_buffered_messages(st.session_state.messages)

    api_messages = buffered_messages[:-1] + [{"role": "user", "content": augmented_prompt}]

    stream = client.chat.completions.create(
        model=openAI_model,
        messages=api_messages,
        stream=True
    )

    with st.chat_message("assistant"):
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})