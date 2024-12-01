import streamlit as st
from llama_index.core import Document
from wikipedia_tool import fetch_wikipedia_content
from retriever import create_index
from query import get_response

content=''

st.title("Wikipedia Question Answering using LlamaIndex and Meltemi")
st.write("Enter a Wikipedia article title, and ask questions about it!")

article_title = st.text_input("Wikipedia Article Title", "Artificial Intelligence")

# Fetch and Display Wikipedia Content
if st.button("Fetch Article"):
    with st.spinner("Fetching content from Wikipedia..."):
        content = fetch_wikipedia_content(article_title)
        if content:
            st.success(f"Successfully fetched article: {article_title}")
            st.text_area("Article Content", content, height=300)
        else:
            st.error(f"Article '{article_title}' not found on Wikipedia.")

if content:
    with st.spinner("Building index..."):
        document = Document(text=content)
        index = create_index([document])
    
    # Input: User Question
    question = st.text_input("Ask a Question", "")
    
    if question:
        # Query the Index
        with st.spinner("Getting the answer..."):
            response = get_response(question, index)
            print(response)
            st.write("**Answer:**")
            st.success(response)