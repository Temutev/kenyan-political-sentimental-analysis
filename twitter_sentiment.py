import streamlit as st
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from wordcloud import STOPWORDS
from collections import defaultdict
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import urllib
import requests
import altair as alt 
import io
import plotly.express as px
import joblib
import nltk 

nltk.download('stopwords')

#from transformers import pipeline
st.set_option('deprecation.showPyplotGlobalUse', False)

#sentiment_analysis = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
def remove_pattern(input_text, pattern):
    r = re.findall(pattern, input_text)
    for i in r:
        input_text = re.sub(i, '', input_text)
    return input_text


#function to extract hashtags
def Hashtags_Extract(x):
    hashtags=[]
    
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r'#(\w+)',i)
        hashtags.append(ht)
    
    return hashtags



def home_page():
    st.header("Welcome to Social Media Data Analysis App")
    st.write("This app allows you to analyze political sentiment on both Twitter and Facebook.")
    st.write("Please select a page from the sidebar on the left.")

    st.subheader("Importance of Political Sentiment Analysis on Twitter and Facebook in Kenya")
    st.write(
        "Twitter and Facebook have become significant platforms for political discourse and engagement in Kenya, "
        "with politicians, citizens, and journalists actively participating in discussions related to "
        "political issues. Political Sentiment Analysis on both Twitter and Facebook holds immense importance in the Kenyan sphere "
        "for the following reasons:"
    )

    st.markdown("1. **Real-time Insights:** Twitter and Facebook provide real-time updates on political opinions, allowing analysts to gauge public sentiment as it evolves.")

    st.markdown("2. **Election Monitoring:** During elections, sentiment analysis can help track public sentiment towards political candidates and parties on both platforms.")

    st.markdown("3. **Public Policy:** Understanding public sentiment on key policy issues can inform government decisions and policymaking, and this extends to both Twitter and Facebook.")

    st.markdown("4. **Media Monitoring:** Media outlets and journalists can use sentiment analysis to measure the impact of their political coverage on both platforms and adjust their strategies accordingly.")

    st.markdown("5. **Crisis Management:** Sentiment analysis can help identify emerging political crises or issues that require immediate attention, and this is applicable to both Twitter and Facebook.")

    st.markdown("6. **Campaign Strategies:** Political campaigns can adjust their strategies based on public sentiment on both Twitter and Facebook to improve their outreach and messaging.")

    st.markdown(
        "This app aims to provide tools for analyzing political sentiment on Twitter and Facebook in Kenya, helping users gain insights "
        "into the dynamic political landscape and its impact on society."
    )



def data_analysis_page():
    #global combined_df
    st.header("Data Analysis")
    st.subheader("Combine and analyze Twitter and Facebook data")

    # Load CSV files directly into a DataFrame
    combined_df = pd.read_csv('merged_df.csv')

    

    # Display your data analysis results using Streamlit widgets
    # Data Overview Section
    st.markdown("## Data Overview")
    st.write("Here's a sample of our Facebook data:")
    st.dataframe(combined_df.loc[205:210,:])

    st.write("Here's a sample of our Twitter data:")
    st.dataframe(combined_df.loc[3000:3005,:])
  # Sentiment Distribution Section
    # Sentiment Distribution Pie Chart
    st.subheader("Sentiment Distribution")
    sentiment_counts = combined_df['sentiment'].value_counts()
    fig = px.pie(sentiment_counts, values=sentiment_counts, names=sentiment_counts.index, title='Sentiment Distribution')
    st.plotly_chart(fig)



    # Text Preprocessing Section
    st.markdown("## Text Preprocessing")
    # Find the number of tweets per day and plot it


    # Remove Twitter handles (@user)
    combined_df['tidy_text'] = np.vectorize(remove_pattern)(combined_df['full_text'], "@[\w]*")

    # Step 1: Removing Twitter Handles
    st.write("1. **Removing Twitter Handles:** In text preprocessing, we often remove Twitter handles, which start with the '@' symbol, to ensure that they don't interfere with our analysis.")

    st.write(combined_df[['full_text', 'tidy_text']].head())

    
    # Step 2: Removing Special Characters, Numbers, Punctuations
    st.write("2. **Removing Special Characters, Numbers, Punctuations:** We eliminate special characters, numbers, and punctuation marks from the text. This helps in focusing on the actual words and their meaning.")
    combined_df['tidy_text'] = combined_df['tidy_text'].str.replace("[^a-zA-Z#]", " ")
    st.write(combined_df[['full_text', 'tidy_text']].head())

    # Step 3: Removing Short Words
    st.write("3. **Removing Short Words:** Short words like 'a,' 'an,' 'the,' etc., don't usually provide much information. Removing them can make text analysis more meaningful.")

    combined_df['tidy_text'] = combined_df['tidy_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 1]))
    st.write(combined_df[['full_text', 'tidy_text']].head())

    
    # Step 4: Tokenization
    st.write("4. **Tokenization:** Tokenization involves splitting text into individual words or 'tokens.' It's a crucial step for many natural language processing tasks as it breaks down text into manageable units.")

    tokenized_tweet = combined_df['tidy_text'].apply(lambda x: x.split())
    st.write(tokenized_tweet.head())

    # Step 5: Stemming
    st.write("5. **Stemming:** Stemming is a text normalization technique that reduces words to their root form. For example, 'running' and 'ran' are stemmed to 'run.' This helps in treating similar words as the same for analysis.")

    stemmer = PorterStemmer()
    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])
    st.write(tokenized_tweet.head())

    # Stitching tokens back together
    st.write("Stitching Tokens Back Together")
    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

    combined_df['tidy_text'] = tokenized_tweet
    st.write(combined_df[['full_text', 'tidy_text']].head())

    # Conclusion
    st.write("These preprocessing steps help clean and prepare text data for various text analysis tasks, like sentiment analysis, topic modeling, and more.")

    # Word Clouds Section
    st.markdown("## Word Clouds")
    st.write("Word clouds are important for sentiments as they visually highlight the most frequently occurring words, providing quick insights into the key themes and emotions in textual data.")

    col1,col2 = st.columns(2)
    

    with col1:
        st.write("Word Cloud for Positive Sentiment:")
        all_words_positive = ' '.join([text for text in combined_df['tidy_text'][combined_df['sentiment'] == 'POS']])
        Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))
        image_colors = ImageColorGenerator(Mask)
        wc_positive = WordCloud(background_color='black', height=1500, width=4000, mask=Mask).generate(all_words_positive)
        st.image(wc_positive.recolor(color_func=image_colors).to_array(), use_column_width=True, caption="Positive Sentiment")

    # Word Cloud for negative sentiment
    with col2:
        st.write("Word Cloud for Negative Sentiment:")
        all_words_negative = ' '.join([text for text in combined_df['tidy_text'][combined_df['sentiment'] == 'NEG']])
        Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))
        image_colors = ImageColorGenerator(Mask)
        wc_negative = WordCloud(background_color='black', height=1500, width=4000, mask=Mask).generate(all_words_negative)
        st.image(wc_negative.recolor(color_func=image_colors).to_array(), use_column_width=True, caption="Negative Sentiment")

    # Top 10 Hashtags Section
    st.markdown("## Top 10 Hashtags")
    st.subheader("Top 10 Hashtags for Sentiments")
    ht_positive = Hashtags_Extract(combined_df['tidy_text'][combined_df['sentiment'] == 'POS'])
    ht_negative = Hashtags_Extract(combined_df['tidy_text'][combined_df['sentiment'] == 'NEG'])

    # Unnesting list
    ht_positive_unnest = sum(ht_positive, [])
    ht_negative_unnest = sum(ht_negative, [])

    # Counting the frequency of words having positive sentiment
    word_freq_positive = nltk.FreqDist(ht_positive_unnest)
    df_positive = pd.DataFrame({'Hashtags': list(word_freq_positive.keys()), 'Count': list(word_freq_positive.values())})
    df_positive_plot = df_positive.nlargest(10, columns='Count')

    
    # Create a bar plot using Plotly Express
    fig = px.bar(df_positive_plot, x='Count', y='Hashtags', orientation='h', title='Top 10 Hashtags for Positive Sentiment')
        
    # Customize the layout if needed
    fig.update_layout(xaxis_title='Count', yaxis_title='Hashtags', xaxis_ticks="outside")
        
    # Display the plotly chart
    st.plotly_chart(fig)

    # Counting the frequency of words having negative sentiment
    word_freq_negative = nltk.FreqDist(ht_negative_unnest)
    df_negative = pd.DataFrame({'Hashtags': list(word_freq_negative.keys()), 'Count': list(word_freq_negative.values())})

  
    # Plotting the barplot for the most frequently used words in hashtags for negative sentiment
    #st.write("Top 10 Hashtags for Negative Sentiment:")
    df_negative_plot = df_negative.nlargest(10, columns='Count')
    # Create a bar plot using Plotly Express
    fig = px.bar(df_negative_plot, x='Count', y='Hashtags', orientation='h', title='Top 10 Hashtags for Negative Sentiment')
        
    # Customize the layout if needed
    fig.update_layout(xaxis_title='Count', yaxis_title='Hashtags', xaxis_ticks="outside")
        
    # Display the plotly chart
    st.plotly_chart(fig)


    # Define a mapping dictionary
    sentiment_mapping = {'POS': 1, 'NEG': 0, 'NEU': -1}

    # Map the 'sentiment' column to numerical labels
    combined_df['sentiment_label'] = combined_df['sentiment'].map(sentiment_mapping)

    #return combined_df
    # Store the combined_df in session_state for access on other pages
    st.session_state.combined_df = combined_df

def sentiment_analysis_page():


    #load our df
    bow_accuracy_df = pd.read_csv('bow_accuracy_df.csv')
    tfidf_accuracy_df = pd.read_csv('tfidf_accuracy_df.csv')

    st.header("Sentiment Analysis")
    st.subheader("Analyze sentiment of Twitter and Facebook Data")

    st.markdown("### Bag-of-Words (BoW) and TF-IDF for Sentiment Analysis")
    st.write("In sentiment analysis, I use techniques like Bag-of-Words (BoW) and TF-IDF (Term Frequency-Inverse Document Frequency) for the following reasons:")

    st.write("1. **Feature Extraction:** BoW and TF-IDF convert text data into numerical vectors, making it possible for machine learning models to understand and classify sentiment.")

    st.write("2. **Word Importance:** TF-IDF considers the importance of words in a document relative to a corpus, capturing valuable information for sentiment analysis.")

    st.write("3. **Dimension Reduction:** BoW and TF-IDF reduce the complexity of text data, which is useful when working with large datasets.")

    st.write("4. **Interpretability:** These techniques provide interpretable features, helping us understand which words contribute to sentiment classification.")

    st.write("5. **Efficiency:** BoW and TF-IDF are computationally efficient, making them suitable for tasks with limited resources.")

    st.write("6. **Widely Used:** BoW and TF-IDF are widely used in natural language processing, with available resources and libraries for easy implementation.")

    st.write("7. **Flexibility:** You can customize BoW and TF-IDF to incorporate domain-specific knowledge and adapt them to your specific task.")

    st.write("In summary, BoW and TF-IDF are essential tools for sentiment analysis, providing structured and interpretable features that help classify sentiment in text data.")


    # Bag-of-Words (BoW) Section
    st.markdown("## Bag-of-Words (BoW)")
    st.write("Bag-of-Words (BoW) is a technique for extracting features from text data. It involves creating a vocabulary of all the unique words in the data and then creating vectors of word counts for each document in the data.")

    st.write("Here are the results of the models trained on the BoW vectors:")
    st.write(bow_accuracy_df)

     # Plot the DataFrame using a bar chart
    st.write("Bar Chart  for Bag-of-Words:")
    # Create a bar chart using Plotly Express
    fig = px.bar(bow_accuracy_df, x='index', y='Accuracy',
                  title='Bar Chart for Bag-of-Words ')
    
    # Customize the layout if needed
    fig.update_xaxes(title_text='Category')
    fig.update_yaxes(title_text='Values')
    
    # Display the Plotly chart
    st.plotly_chart(fig)


    # TF-IDF Section
    st.markdown("## TF-IDF")
    st.write("TF-IDF (Term Frequency-Inverse Document Frequency) is a technique for extracting features from text data. It involves creating a vocabulary of all the unique words in the data and then creating vectors of word counts for each document in the data.")

    st.write("Here are the results of the models trained on the TF-IDF vectors:")
    st.write(tfidf_accuracy_df)


    # Plot the DataFrame using a bar chart
    st.write("Bar Chart for TF-IDF:")
    # Create a bar chart using Plotly Express
    fig = px.bar(tfidf_accuracy_df, x='index', y='Accuracy',
                  title='Bar Chart for TF-IDF')
    
    # Customize the layout if needed
    fig.update_xaxes(title_text='Category')
    fig.update_yaxes(title_text='Values')

    # Display the Plotly chart
    st.plotly_chart(fig)




# Create a sidebar navigation
app_pages = {
    "Home": home_page,
    "Data Analysis": data_analysis_page,
    "Modelling": sentiment_analysis_page,
}

# Add a Streamlit sidebar for navigation
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to", list(app_pages.keys()))

# Display the selected page
app_pages[selected_page]()



