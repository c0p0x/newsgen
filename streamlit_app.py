import streamlit as st 
import requests
import json 
import numpy as np
import os
import openai 


from newspaper import Article, ArticleException
from langchain.text_splitter import TokenTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate, LLMChain, OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chat_models import ChatOpenAI


model = -1

def set_model(model):
    os.environ['OPENAI_MODEL'] = model
def get_model():
    return os.getenv('OPENAI_MODEL')

def get_openai_api_key():
    # Retrieve the API key from an environment variable
    return os.getenv('OPENAI_API_KEY')

def clear_openai_api_key():
    # Clear the API key from the environment for security
    os.environ['OPENAI_API_KEY'] = ''

def set_openai_api_key(key):
    os.environ['OPENAI_API_KEY'] = key

def get_latest_results(query):

    
    parsed_texts = [] #list to store parsed text and corresponding URL
    article_texts = []  # list to store original article texts for similarity comparison

    # Initialize the text_splitter before using it
    text_splitter = TokenTextSplitter(chunk_size=2000, chunk_overlap=500)

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

    # set api key in env variable 
    set_openai_api_key(openai_api_key)

    llm = ChatOpenAI(model_name=get_model(), openai_api_key=openai_api_key, temperature=0.68, max_tokens = 3000)
    # Initialize the chain for summarization
    chain_summarize = load_summarize_chain(llm, chain_type="stuff")
    
    # Define prompt that generates titles for summarized text
    title_prompt = PromptTemplate(
        input_variables=["text"], 
        template="""Write an appropriate, clickbaity, but not judgemental news article title in Polish for less then approximatetly 200 characters for this text: {text}. Make sure it is in Polish and less then 100chars. Prepare list of 5 titles so I can choose. 
        
        PROPOSED TITLES IN POLISH:
        """
        )
    # define prompt that generates text translated 
    text_prompt = PromptTemplate(
        input_variables=["text"], 
        template="""Please provide engaging post of the following text in Polish, ensuring that it is 220 words approximate - SUPER IMPORTANT. The summary should be informative, neutral, and devoid of any judgmental tones focusing on and quoting facts from article. Remember, the post must be in Polish. {text}
        
        LONG SUMMARY IN POLISH:
        """
    )

    facts_prompt = PromptTemplate(
        input_variables=["text"], 
        template="""Please provide a list of 10-15 key facts in polish - ONLY KEY FACTS - such as statistics, numbers, prices etc from the {text}
        
        LIST OF KEY FACTS IN POLISH:
        """
    )
    full_prompt = PromptTemplate(
        input_variables=["text"], 
        template="""Make a fact-driven article in Polish from the following text: {text}
        
        LONG ARTICLE IN POLISH WITH FACTS AND QUOTES:
        """
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
        chain_prompt_title = LLMChain(llm=llm, prompt=title_prompt)
        clickbait_title = chain_prompt_title.run(summarized_text)

       # full article parser
        if to_summarize_texts and to_summarize_texts[0] and to_summarize_texts[0][0]:
            desired_text = to_summarize_texts[0][0][0]
            #st.write(desired_text)
        else:
             st.write("Text not found")

        
        chain_prompt_text = LLMChain(llm=llm, prompt=text_prompt)
        short_article = chain_prompt_text.run(summarized_text)

        chain_prompt_text = LLMChain(llm=llm, prompt=full_prompt)
        full_article = chain_prompt_text.run(desired_text)

        chain_prompt_text = LLMChain(llm=llm, prompt=facts_prompt)
        facts = chain_prompt_text.run(desired_text)

        summarized_texts_titles_urls.append((clickbait_title, short_article, full_article, facts, summarized_text, url))



    return summarized_texts_titles_urls

def summarize_text_raw_text(raw_text, openai_api_key):
  
    text_prompt = PromptTemplate(
        input_variables=["text"], 
        template="""Please provide engaging post of the following text in Polish, ensuring that it is 220 words approximate - SUPER IMPORTANT. The summary should be informative, neutral, and devoid of any judgmental tones focusing on and quoting facts from article. Remember, the post must be in Polish. {text}
        
        LONG SUMMARY IN POLISH:
        """
    )

    llm = ChatOpenAI(model_name=get_model(), openai_api_key=openai_api_key, temperature=0.68, max_tokens = 3000)

    chain_prompt_text = LLMChain(llm=llm, prompt=text_prompt)
    short_article = chain_prompt_text.run(raw_text)

    return short_article

def main():
    st.title('AutoNewsletter-DEV')

    # Create text input field for API keys 
    openai_api_key = st.text_input("Insert your OpenAI api key: ", type="password")

    selectbox = st.selectbox("GPT Model to be used", ("gpt-3.5-turbo-1106", "gpt-4-1106-preview", "gpt-4-0125-preview"))
    if selectbox == "gpt-3.5-turbo-1106":
        set_model("gpt-3.5-turbo-1106")
    elif selectbox == "gpt-4-1106-preview":
        set_model("gpt-4-1106-preview")
    elif selectbox == "gpt-4-0125-preview":
        set_model("gpt-4-0125-preview")   
    selectbox = st.selectbox("Raw text or URL source", ("URL", "Raw text"))

    if selectbox == "Raw text":
        raw_text = st.text_area(label="Text", height=300, max_chars=10000)
        if st.button("Submit Raw Text"):
            st.write(summarize_text_raw_text(raw_text, openai_api_key))
            
     

    elif selectbox == "URL":
        user_query = st.text_input(label="URL")
        if st.button("Submit URL"):
            st.session_state.user_query = user_query
            st.session_state.get_splitted_text = get_latest_results(user_query)
            st.session_state.summarized_texts = summarize_text(st.session_state.get_splitted_text, openai_api_key)
            if st.session_state.get_splitted_text:
                for title, short_article, full_article, facts, summarized_text, url in st.session_state.summarized_texts:
                    st.markdown("## Headline") 
                    st.write(title)
                    st.markdown("## Summary") 
                    st.write(f"❇️ {short_article}")
                    st.markdown("## Key facts") 
                    st.write(facts)
                    st.markdown("## Full article") 
                    st.write(full_article)
                    st.write(f"🔗 {url}")
                    st.markdown("\n\n")

    # Wipe API key 
    clear_openai_api_key()


if __name__ == "__main__":
    main()
