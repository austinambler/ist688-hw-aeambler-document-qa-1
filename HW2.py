import streamlit as st
from openai import OpenAI, AuthenticationError

# Show title and description.
st.title("HW 2 Document question answering")
st.write(
    "Upload a document below and choose a summary style – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.secrets.OPEN_AI_KEY

key_is_valid = False

# check to make sure api key is valid
if openai_api_key:
    try:
        client = OpenAI(api_key=openai_api_key)
        # Lightweight call just to check if the key works
        client.models.list()
        st.success("API key is valid ✅")
        key_is_valid = True
    except AuthenticationError:
        st.error("Invalid OpenAI API key ❌")
    except Exception as e:
        st.error(f"Something went wrong: {e}")

if not openai_api_key or not key_is_valid:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # --- Sidebar controls ---
    st.sidebar.header("Options")

    # Summary style options (replaces the free-text question box).
    summary_option = st.sidebar.radio(
        "Choose a summary style:",
        (
            "Summarize the document in 100 words",
            "Summarize the document in 2 connecting paragraphs",
            "Summarize the document in 5 bullet points",
        ),
    )

    # Model selection: checkbox toggles between nano (default) and mini (advanced).
    use_advanced_model = st.sidebar.checkbox("Use advanced model")
    model = "gpt-4.1-mini" if use_advanced_model else "gpt-4.1-nano"
    st.sidebar.caption(f"Model in use: `{model}`")

    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md)", type=("txt", "md")
    )

    if uploaded_file:

        # Process the uploaded file and the chosen summary option.
        document = uploaded_file.read().decode()
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {summary_option}",
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