import pandas as pd
import joblib
import numpy as np
import re
import streamlit as st
import nltk

nltk.download('punkt')
nltk.download('stopwords')


# Loading models
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")
map = joblib.load("int_to_label.pkl")


st.set_page_config(
    page_title="Resume Screener",
    page_icon="📄",
    layout="centered"
    )

# Cleaning function

def cleanResume(text):
    # 1) # remove URLs - http://helloworld
    cleantxt = re.sub(r'http\S+','', text) # Remove anything from https: till the next whitespace
    # 2) remove emails [....@....]
    cleantxt = re.sub(r'\S+@\S+','', cleantxt) # Removes anthing before and after '@'
    # 3) remove hashtags [#....]
    cleantxt = re.sub(r'#\S+','', cleantxt) # Removes anthing after '#'
    # 4) remove special characters
    cleantxt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[/]^_'{|}~"""),'', cleantxt)
    # remove RT, cc safely
    cleantxt = re.sub(r'\bRT\b|\bcc\b', '', cleantxt)
    # 5) remove non-ASCII
    cleantxt = re.sub(r'[^\x00-\x7f]','', cleantxt)
    # 6) Removes any type of whitespaces like \t, \n etc with a single whitespace
    # normalize whitespace
    cleantxt = re.sub(r'\s+',' ', cleantxt)
    
    
    cleantxt = cleantxt.strip()  
    return cleantxt



# Web app
def main():
    st.title("Resume Screening App")
    
    
    
    
    uploaded_file = st.file_uploader("Upload Resume", type=['pdf', 'txt'])
    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 Decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')
            
        resume_text = resume_text.lower()
        resume_text = re.sub(r'\n+', ' ', resume_text)
        resume_text = re.sub(r'\s+', ' ', resume_text)
        
        st.write("Extracted text length:", len(resume_text))
        st.write(resume_text[:500])
        
        
        cleaned_text = cleanResume(resume_text)
        resume_vector = tfidf.transform([cleaned_text])
        prediction_id = model.predict(resume_vector)
        
        category = map[int(prediction_id)]
        
        
        


        st.write(category)
        





















# python main
if __name__ == "__main__":
    main()
    