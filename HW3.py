import streamlit as st
from openai import OpenAI, AuthenticationError
from bs4 import BeautifulSoup
import requests

st.title("My Homework 3 question answering chatbot")

st.write(
    """
    **How this app works:**
    1. Choose an LLM (OpenAI or Gemini) and how many URLs you want to provide (1 or 2) in the sidebar.
    2. Enter the URL(s) in the sidebar — the app will fetch and read the page content as reference material.
    3. Once loaded, ask a question about the URL(s) in the chat box below.
    4. The bot will answer, then ask "Do you want more info?" — say yes for more detail, or no to move to a new question.
    5. **Conversation memory:** to keep responses fast and within model limits, this app only remembers your **last 3 questions and 3 answers** at a time (a "buffer"). Older messages are dropped from what's sent to the model, but the reference document content is always included so the bot never forgets what page(s) you're asking about.
    """
)

llm = st.sidebar.radio("Choose a LLM:", ("OpenAI", "Gemini"))
url_count = st.sidebar.radio("How many URLs:", (1, 2))

openai_api_key = st.secrets.OPEN_AI_KEY
gemini_api_key = st.secrets.GEMINI_KEY

key_is_valid = False
client = None

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
        return soup.get_text(separator="\n", strip=True)
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about web page content the user has provided.

Style rules:
- Use simple, everyday words and short sentences.
- Avoid jargon and technical terms; if you must use one, explain it simply right after.
- Use relatable examples or comparisons (like toys, games, animals, or everyday situations) to make ideas easier to picture.
- Keep a friendly, encouraging tone.

Conversation pattern:
1. The user has provided the content of one or two web pages as reference material. Wait for the user to ask a question about that content.
2. Answer the question clearly and simply, using only the provided document content as your source. If the answer isn't in the document(s), say so honestly instead of guessing.
3. After answering, ask: "Do you want more info?"
4. If the user says yes (or anything affirmative):
   - Provide additional relevant detail from the document(s) on the same topic, still explained simply.
   - Then ask again: "Do you want more info?"
   - Repeat this loop for as long as the user keeps saying yes.
5. If the user says no (or anything negative):
   - Respond with something like "Sounds good!" and then ask: "What else do you want to know about the document(s)?"
   - Wait for a new question and start the pattern over from step 1.

Always keep track of the current topic so "more info" responses stay relevant, until the user asks something new.
"""

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": "Enter your URL(s) on the left, then ask me a question about them!"}
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

# Let the user enter one or two URLs as reference material.
url_1 = st.sidebar.text_input("Enter a URL", placeholder="https://example.com/article")

url_2 = None
if url_count == 2:
    url_2 = st.sidebar.text_input("Enter a second URL", placeholder="https://example.com/another-article")

ready = url_1 and (url_count == 1 or (url_count == 2 and url_2))
current_urls = (url_1, url_2)

# read the URL(s) once per unique URL combination, and store as context (not a chat turn)
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
    else:
        combined_document = f"Document (from {url_1}):\n{document_1}"

    st.session_state.document_context = combined_document
    st.session_state.last_urls = current_urls
    st.success("URL(s) loaded! Ask me a question about them below.")

# Chat input for asking questions about the loaded document(s)
if prompt := st.chat_input("Ask a question about the URL(s)..."):
    if "document_context" not in st.session_state:
        st.warning("Please enter and load a URL first before asking a question.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    client = st.session_state.client
    buffered_messages = get_buffered_messages(st.session_state.messages)

    # inject the document content as context right after the system prompt,
    # without permanently bloating the stored chat history on every turn
    context_message = {
        "role": "system",
        "content": f"Reference document content:\n\n{st.session_state.document_context}"
    }
    api_messages = [buffered_messages[0], context_message] + buffered_messages[1:]

    stream = client.chat.completions.create(
        model=st.session_state.model_name,
        messages=api_messages,
        stream=True,
    )

    with st.chat_message("assistant"):
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})