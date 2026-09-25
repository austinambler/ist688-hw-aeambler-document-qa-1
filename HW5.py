import streamlit as st
from openai import OpenAI
import sys

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
import json

if 'openai_client' not in st.session_state:
    api_key = st.secrets["OPEN_AI_KEY"]
    st.session_state.openai_client = OpenAI(api_key=api_key)

# Connect to the existing ChromaDB collection (already populated elsewhere)
chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_HW')
collection = chroma_client.get_collection('HW4Collection')

st.title("My HW5 Chatbot Using RAG Enhanced")





openAI_model = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a helpful assistant. Follow this conversation pattern strictly:

- Use simple, everyday words and short sentences.
- Avoid jargon and technical terms; if you must use one, explain it simply right after.

You will be given "Reference material" pulled from HTML documents, along with the user's question.
- If the reference material is relevant, base your answer on it, and briefly mention that you're drawing 
from the reference documents (e.g., "Based on the reference materials...").
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

# --- Tool: retrieve relevant chunks from ChromaDB for a given query ---
def relevant_club_info(query, collection, client, n_results=5):
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

    if not docs:
        return "No relevant reference material was found."

    return "\n\n---\n\n".join(
        f"[Source: {doc_id}]\n{doc_text}"
        for doc_id, doc_text in zip(ids, docs)
    )


# JSON schema describing relevant_club_info to the OpenAI API
tools = [
    {
        "type": "function",
        "function": {
            "name": "relevant_club_info",
            "description": (
                "Search the club/organization reference documents and return the "
                "most relevant passages for a query. Use this when the user's "
                "question is about a specific student org or club."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up in the reference documents."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    client = st.session_state.openai_client

    buffered_messages = get_buffered_messages(st.session_state.messages)

    # --- Step 1: let the LLM decide whether it needs to call relevant_club_info ---
    first_response = client.chat.completions.create(
        model=openAI_model,
        messages=buffered_messages,
        tools=tools,
        tool_choice="auto"
    )

    first_msg = first_response.choices[0].message
    tool_calls = first_msg.tool_calls

    if tool_calls:
        messages_with_tool_result = buffered_messages + [
            {
                "role": "assistant",
                "content": first_msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in tool_calls
                ]
            }
        ]

        for tc in tool_calls:
            if tc.function.name == "relevant_club_info":
                args = json.loads(tc.function.arguments)
                query = args.get("query", prompt)
                tool_output = relevant_club_info(query, collection, client)
            else:
                tool_output = ""

            messages_with_tool_result.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_output
            })

        # --- Step 2: give the LLM the tool results and let it answer. 
        stream = client.chat.completions.create(
            model=openAI_model,
            messages=messages_with_tool_result,
            stream=True
        )
    else:
        # The model chose to answer without retrieving anything.
        stream = client.chat.completions.create(
            model=openAI_model,
            messages=buffered_messages,
            stream=True
        )

    with st.chat_message("assistant"):
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})