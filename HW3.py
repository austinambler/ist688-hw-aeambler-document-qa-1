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
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

SYSTEM_PROMPT = """You are a helpful assistant that reads and explains web page content.


Conversation pattern:
1. When the user provides document content (from one or two URLs), read it and give a clear, simple summary of what it says. If two documents are provided, summarize each one and briefly note how they relate or differ.
2. After giving the summary, ask: "Do you want more info?"
3. If the user says yes (or anything affirmative):
   - Go deeper into the same document content — add more detail, examples, or explanation, still in simple language.
   - Then ask again: "Do you want more info?"
   - Repeat this loop for as long as the user keeps saying yes.
4. If the user says no (or anything negative):
   - Respond with something like "Sounds good!" and then ask: "What can I help you with? Enter a new URL anytime."
   - Wait for a new question or a new URL and start the pattern over from step 1.

Always keep track of which document(s) are currently being discussed so "more info" responses stay relevant, until the user submits a new URL or asks something unrelated.
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
max_messages = 3  # 6 total - 3 for each user and assistant

def get_buffered_messages(messages, max_messages=max_messages):
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

# only proceed once required URL fields are filled in, and only once per URL set
ready = url_1 and (url_count == 1 or (url_count == 2 and url_2))
current_urls = (url_1, url_2)

if ready and st.session_state.get("last_urls") != current_urls:
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
    st.session_state.last_urls = current_urls  # remember so we don't re-summarize on every rerun

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

# Chat input for follow-up messages (e.g. "yes"/"no" to "Do you want more info?")
if prompt := st.chat_input("Type your reply here (e.g. yes / no), or ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

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