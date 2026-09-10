import streamlit as st
from openai import OpenAI, AuthenticationError
from bs4 import BeautifulSoup
import requests

st.title("My Homework 3 question answering chatbot")

llm = st.sidebar.radio("Choose a LLM:", ("OpenAI", "Gemini"))
url_count = st.sidebar.radio("How many URLs:", (1, 2))
 
openai_api_key = st.secrets.OPEN_AI_KEY
gemini_api_key = st.secrets.GEMINI_KEY

key_is_valid = False
client = None
 
# check to make sure api key is valid
if llm == "OpenAI":
    if openai_api_key:
        try:
            client = OpenAI(api_key=openai_api_key)
            # Lightweight call just to check if the key works
            client.models.list()
            st.success("OpenAI API key is valid ✅")
            key_is_valid = True
        except AuthenticationError:
            st.error("Invalid OpenAI API key ❌")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
    api_key_present = bool(openai_api_key)
else:
    if gemini_api_key:
        try:
            client = OpenAI(
                api_key=gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            # Lightweight call just to check if the key works
            client.models.list()
            st.success("Gemini API key is valid ✅")
            key_is_valid = True
        except AuthenticationError:
            st.error("Invalid Gemini API key ❌")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
    api_key_present = bool(gemini_api_key)

# reset client if the user switches models mid-session
if "current_llm" not in st.session_state or st.session_state.current_llm != llm:
    st.session_state.current_llm = llm

    if llm == "OpenAI":
        st.session_state.client = OpenAI(api_key=openai_api_key)
        st.session_state.model_name = "gpt-4.1-mini"

    elif llm == "Gemini":
        st.session_state.client = OpenAI(
            api_key=gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        st.session_state.model_name = "gemini-3.6-flash"

def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

SYSTEM_PROMPT = """You are a helpful assistant. Follow this conversation pattern strictly:

- Use simple, everyday words and short sentences.
- Avoid jargon and technical terms; if you must use one, explain it simply right after.
- Use relatable examples or comparisons (like toys, games, animals, or everyday situations) to make ideas easier to picture.
- Keep a friendly, encouraging tone.

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
max_messages = 2

# define function to get buffered messages
def get_buffered_messages(messages, max_messages = max_messages):

    system_msgs = [m for m in messages if m["role"] == "system"]
    user_idxs = [i for i, m in enumerate(messages) if m["role"] == "user"][-max_messages:]
    assistant_idxs = [i for i, m in enumerate(messages) if m["role"] == "assistant"][-max_messages:]

    keep_idxs = sorted(set(user_idxs + assistant_idxs))
    trimmed = [messages[i] for i in keep_idxs]

    return system_msgs + trimmed

# Let the user enter one or two URLs instead of uploading a file.
url_1 = st.text_input("Enter a URL", placeholder="https://example.com/article")

url_2 = None
if url_count == 2:
    url_2 = st.text_input("Enter a second URL", placeholder="https://example.com/another-article")

# only proceed once required URL fields are filled in
ready = url_1 and (url_count == 1 or (url_count == 2 and url_2))

if ready:
    document_1 = read_url_content(url_1)

    if not document_1:
        st.warning(f"Could not retrieve content from {url_1}. Try a different link.")
        st.stop()

    if url_count == 2:
        document_2 = read_url_content(url_2)

        if not document_2:
            st.warning(f"Could not retrieve content from {url_2}. Try a different link.")
            st.stop()

        combined_document = (
            f"Document 1 (from {url_1}):\n{document_1}\n\n"
            f"---\n\n"
            f"Document 2 (from {url_2}):\n{document_2}"
        )
        display_text = f"Summarize these URLs: {url_1} and {url_2}"
    else:
        combined_document = document_1
        display_text = f"Summarize this URL: {url_1}"

    user_message = {
        "role": "user",
        "content": (
            f"Here's the document content: {combined_document}"
        ),
    }

    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(display_text)

    client = st.session_state.client
    buffered_messages = get_buffered_messages(st.session_state.messages)

    stream = client.chat.completions.create(
        model=st.session_state.model_name,
        messages=buffered_messages,
        stream=True,
    )

    with st.chat_message("assistant"):
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})