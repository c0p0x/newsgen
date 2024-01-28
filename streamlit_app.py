#Import and install necessary dependencies 

#serpapi, requests → Scrape google results
#sklearn → filter results based on how similar they are 
#Newspaper3K → extract text from articles 
#Langchain → split text/summarize it and prompt template in order to generate the title
#MailGun → send email 

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

# checks if the fetched newsarticles are identical, and filters out the ones that are too similar
def is_unique(new_article, articles):
    if not articles:  # if the list is empty
        return True

    # Create a new TfidfVectorizer and transform the article texts into vectors
    vectorizer = TfidfVectorizer().fit([new_article] + articles)
    vectors = vectorizer.transform([new_article] + articles)

    # Calculate the cosine similarity of the new article to each of the existing articles
    similarity_scores = cosine_similarity(vectors[0:1], vectors[1:])

    # If the highest similarity score is above a threshold (for example, 0.8), return False (not unique), keep at around 0.6
    if np.max(similarity_scores) > 0.6:
        return False

    # Otherwise, return True (unique)
    return True

# Scrapes google search results
def get_latest_results(query):

    url = query
    parsed_texts = [] #list to store parsed text and corresponding URL
    article_texts = []  # list to store original article texts for similarity comparison

    # Initialize the text_splitter before using it
    text_splitter = TokenTextSplitter(chunk_size=3000, chunk_overlap=200)

    #iterate over each URL 
  
    try:
        #create an article object
        article = Article(url)

        #download the article 
        article.download()

        #parse the article 
        article.parse()

        #split text into chunks of 4k tokens 
        splitted_texts = text_splitter.split_text(article.text)
        if not splitted_texts:
            print(article.text)
              
        #Append tuple of splitted text and URL to the list
        parsed_texts.append((splitted_texts, url))
        article_texts.append(article.text)  # Add the text of the new unique article to the list

    except ArticleException: 
        print(f"Failed to download and parse article: {url}")

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
    
    # Define prompt that generates titles for summarized text
    prompt = PromptTemplate(
            input_variables=["text"], 
            template="Write an appropriate, clickbaity news article title in less than 250 characters for this text: {text}"
        )
   
    for to_summarize_text, url in to_summarize_texts:
        # Convert each text string to a Document object
        to_summarize_text = [Document('Dummy Title', text) for text in to_summarize_text]
        if not to_summarize_text:  # Check if list is empty before running the chain
          print(f"No text to summarize for URL: {url}")
          continue
        
        # Summarize chunks here
        summarized_text = chain_summarize.run(to_summarize_text)

        # prompt template that generates unique titles
        chain_prompt = LLMChain(llm=llm, prompt=prompt)
        clickbait_title = chain_prompt.run(summarized_text)

        summarized_texts_titles_urls.append((clickbait_title, summarized_text, url))

    return summarized_texts_titles_urls



def main():
    test = "siema"
    test2 = "siema2"
    #frontend
    st.title('AutoNewsletter')
    st.markdown("## Please input your API keys")

    #create text input field for API keys 
    openai_api_key = st.text_input("Insert your OpenAI api key: ", type="password")

    #create text input field for keyword 
    user_query = st.text_input("URL")
    
    #Summarized article to be translated 
    st.markdown("## Summarized article") 
    st.text(test)


    st.markdown("## OpenAI Translation")
    st.text(test2)

    if st.button('Submit'):
        st.session_state.user_query = user_query

        # Split the result of get_latest_results into two separate variables
        st.session_state.get_splitted_text = get_latest_results(user_query)
        if not st.session_state.get_splitted_text:
            st.write("No results found.")
        st.session_state.summarized_texts = summarize_text(st.session_state.get_splitted_text, openai_api_key)
        
        for title, summarized_text, url in st.session_state.summarized_texts:
          st.title(title)
          # Add the emoji before the summarized text
          st.write(f"❇️ {summarized_text}")
          st.write(f"🔗 {url}")
          # Create an empty line for a gap
          st.markdown("\n\n")

    return openai_api_key

if __name__ == "__main__":
    main()
