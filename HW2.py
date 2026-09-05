import streamlit as st
from openai import OpenAI, AuthenticationError
from bs4 import BeautifulSoup
import requests

def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

# Show title and description.
st.title("HW 2 URL question answering")
st.write(
    "Upload an URL below and choose a summary style – GPT or Gemini will answer! "
    "To use this app, you need to provide an OpenAI API key or Gemini Key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
llm = st.sidebar.radio("Choose a LLM:", ("OpenAI", "Gemini"))
 
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
 
if not api_key_present or not key_is_valid:
    st.info(f"Please add your {llm} API key to continue.", icon="🗝️")
else:
 
    # Summary style options (replaces the free-text question box).
    summary_option = st.sidebar.radio(
        "Choose a summary style:",
        (
            "Summarize the document in 100 words",
            "Summarize the document in 2 connecting paragraphs",
            "Summarize the document in 5 bullet points",
        ),
    )
 
    # Model selection: checkbox toggles between basic (default) and advanced model.
    # Model names depend on which LLM is selected.
    use_advanced_model = st.sidebar.checkbox("Use advanced model")
    if llm == "OpenAI":
        model = "gpt-4.1-mini" if use_advanced_model else "gpt-4.1-nano"
    else:
        model = "gemini-3.6-flash" if use_advanced_model else "gemini-3.5-flash-lite"
    st.sidebar.caption(f"Model in use: `{model}`")
 
    # Output language selection.
    language = st.sidebar.radio(
        "Choose output language:",
        ("English", "French", "Spanish"),
    )
 
    # Let the user enter a URL instead of uploading a file.
    url = st.text_input("Enter a URL", placeholder="https://example.com/article")
 
    if url:
 
        # Fetch and parse the page content.
        document = read_url_content(url)
 
        if not document:
            st.warning("Could not retrieve content from that URL. Try a different link.")
            st.stop()
 
        messages = [
            {
                "role": "user",
                "content": (
                    f"Here's a document: {document} \n\n---\n\n {summary_option}. "
                    f"Write your entire response in {language}, regardless of what "
                    f"language the document is written in."
                ),
            }
        ]
 
        # Generate an answer using the OpenAI API.
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        # Stream the response to the app using `st.write_stream`.
        st.write_stream(stream)