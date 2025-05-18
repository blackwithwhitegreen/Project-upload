import streamlit as st
from langchain_community.chat_models import ChatDeepSeek

st.title("🦜🔗 DeepSeek Quickstart")

deepseek_api_key = st.sidebar.text_input("DeepSeek API Key", type="password")

def generate_response(input_text):
    model = ChatDeepSeek(
        temperature=0.7,
        deepseek_api_key=deepseek_api_key
    )
    st.info(model.invoke(input_text))

with st.form("my_form"):
    text = st.text_area("Enter text:", "What are three key pieces of advice for learning how to code?")
    submitted = st.form_submit_button("Submit")
    if not deepseek_api_key.startswith("ds-"):
        st.warning("Please enter your DeepSeek API key!", icon="⚠")
    if submitted and deepseek_api_key.startswith("ds-"):
        generate_response(text)

# import streamlit as st
# from langchain_community.llms import HuggingFaceHub

# st.title("🦜🔗 HF Model Quickstart")

# huggingfacehub_api_key = st.sidebar.text_input("HuggingFace Hub API Key", type="password")

# def generate_response(input_text):
#     # For different models: google/flan-t5-xxl, mistralai/Mistral-7B-v0.1, etc.
#     model = HuggingFaceHub(
#         repo_id="mistralai/Mistral-7B-v0.1",
#         model_kwargs={"temperature": 0.7, "max_length": 500},
#         huggingfacehub_api_token=huggingfacehub_api_key
#     )
#     st.info(model.invoke(input_text))

# with st.form("my_form"):
#     text = st.text_area("Enter text:", "What are three key pieces of advice for learning how to code?")
#     submitted = st.form_submit_button("Submit")
#     if not huggingfacehub_api_key.startswith("hf_"):
#         st.warning("Please enter your HuggingFace Hub token!", icon="⚠")
#     if submitted and huggingfacehub_api_key.startswith("hf_"):
#         generate_response(text)
