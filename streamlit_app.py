import streamlit as st 
import requests
import json 
import numpy as np

from newspaper import Article, ArticleException
from langchain.text_splitter import TokenTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate, LLMChain, OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_latest_results(query):

    
    parsed_texts = [] #list to store parsed text and corresponding URL
    article_texts = []  # list to store original article texts for similarity comparison

    # Initialize the text_splitter before using it
    text_splitter = TokenTextSplitter(chunk_size=3000, chunk_overlap=200)

    #iterate over each URL 
  
    try:
        #create an article object
        article = Article(query)

        #download the article 
        article.download()

        #parse the article 
        article.parse()

        #split text into chunks of 4k tokens 
        splitted_texts = text_splitter.split_text(article.text)
        if not splitted_texts:
            print(article.text)
              
        #Append tuple of splitted text and URL to the list
        parsed_texts.append((splitted_texts, query))
        article_texts.append(article.text)  # Add the text of the new unique article to the list

    except ArticleException: 
        print(f"Failed to download and parse article: {query}")

    return parsed_texts

#required by chain.run()
class Document:
    def __init__(self, title, text):
        self.title = title
        self.page_content = text
        self.metadata = {"stop": []} 

def summarize_text(to_summarize_texts, openai_api_key):
    summarized_texts_titles_urls = []

    llm = OpenAI(openai_api_key=openai_api_key, temperature=0.8)
    chain_summarize = load_summarize_chain(llm, chain_type="map_reduce")

    # Define prompt for generating titles
    title_prompt = PromptTemplate(
        input_variables=["text"], 
        template="Write a clickbaity title in Polish for this text: {text}. Please prepare 3 versions in a list."
    )

    # Define prompt for summarizing text
    text_prompt = PromptTemplate(
        input_variables=["text"], 
        template="Summarize the following text in Polish in a concise and informative manner: {text}"
    )

    for to_summarize_text, url in to_summarize_texts:
        if not to_summarize_text:
            print(f"No text to summarize for URL: {url}")
            continue

        # Convert text string to a Document object for summarization
        to_summarize_text_doc = Document('Dummy Title', to_summarize_text)

        # Create chain for text summarization
        chain_summarize_text = LLMChain(llm=llm, prompt=text_prompt)
        summarized_text = chain_summarize_text.run([to_summarize_text_doc])

        # Create chain for title generation
        chain_prompt_title = LLMChain(llm=llm, prompt=title_prompt)
        clickbait_title = chain_prompt_title.run([summarized_text])

        summarized_texts_titles_urls.append((clickbait_title, summarized_text, url))

    return summarized_texts_titles_urls




def main():
    #frontend
    st.title('AutoNewsletter')
    st.markdown("## Please input your API keys")

    #create text input field for API keys 
    openai_api_key = st.text_input("Insert your OpenAI api key: ", type="password")

    #create text input field for keyword 
    user_query = st.text_input("URL")

    if st.button('Submit'):
        st.session_state.user_query = user_query

        # Split the result of get_latest_results into two separate variables
        st.session_state.get_splitted_text = get_latest_results(user_query)
        if not st.session_state.get_splitted_text:
            st.write("No results found.")
        st.session_state.summarized_texts = summarize_text(st.session_state.get_splitted_text, openai_api_key)
        
        for title, summarized_text, url in st.session_state.summarized_texts:
          st.markdown("## Suggested titles") 
          st.title(title)
          # Add the emoji before the summarized text
          st.markdown("## Suggested articles") 
          st.write(f"❇️ {summarized_text}")
          st.write(f"🔗 {url}")
          # Create an empty line for a gap
          st.markdown("\n\n")

    return openai_api_key

if __name__ == "__main__":
    main()
