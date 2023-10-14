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

#from transformers import pipeline
st.set_option('deprecation.showPyplotGlobalUse', False)

#sentiment_analysis = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
def remove_pattern(text, pattern):
    if isinstance(text, str):
        return re.sub(pattern, '', text)
    else:
        return text


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
    #folder_path = '/home/tevin/Desktop/Tems/twitter/data/'
    file_path = 'combined_data.csv'
    combined_df = pd.read_csv(file_path)

    #combined_df = combined_df[['full_text', 'reply_count', 'retweet_count', 'favorite_count', 'url', 'created_at', 'sentiment']]
    #combined_df['created_at'] = pd.to_datetime(combined_df['created_at'])

    #combined_df.sort_values(by=['created_at'], inplace=True, ascending=True)

    # Display your data analysis results using Streamlit widgets
    # Data Overview Section
    st.markdown("## Data Overview")
    st.write("Here's a sample of our Facebook data:")
    st.dataframe(combined_df.head())

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
    combined_df['Tidy_Tweets'] = np.vectorize(remove_pattern)(combined_df['full_text'], "@[\w]*")

    # Step 1: Removing Twitter Handles
    st.write("1. **Removing Twitter Handles:** In text preprocessing, we often remove Twitter handles, which start with the '@' symbol, to ensure that they don't interfere with our analysis.")

    st.write(combined_df[['full_text', 'Tidy_Tweets']].head())

    
    # Step 2: Removing Special Characters, Numbers, Punctuations
    st.write("2. **Removing Special Characters, Numbers, Punctuations:** We eliminate special characters, numbers, and punctuation marks from the text. This helps in focusing on the actual words and their meaning.")
    combined_df['Tidy_Tweets'] = combined_df['Tidy_Tweets'].str.replace("[^a-zA-Z#]", " ")
    st.write(combined_df[['full_text', 'Tidy_Tweets']].head())

    # Step 3: Removing Short Words
    st.write("3. **Removing Short Words:** Short words like 'a,' 'an,' 'the,' etc., don't usually provide much information. Removing them can make text analysis more meaningful.")

    combined_df['Tidy_Tweets'] = combined_df['Tidy_Tweets'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 1]))
    st.write(combined_df[['full_text', 'Tidy_Tweets']].head())

    
    # Step 4: Tokenization
    st.write("4. **Tokenization:** Tokenization involves splitting text into individual words or 'tokens.' It's a crucial step for many natural language processing tasks as it breaks down text into manageable units.")

    tokenized_tweet = combined_df['Tidy_Tweets'].apply(lambda x: x.split())
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

    combined_df['Tidy_Tweets'] = tokenized_tweet
    st.write(combined_df[['full_text', 'Tidy_Tweets']].head())

    # Conclusion
    st.write("These preprocessing steps help clean and prepare text data for various text analysis tasks, like sentiment analysis, topic modeling, and more.")

    # Word Clouds Section
    st.markdown("## Word Clouds")
    st.write("Word clouds are important for sentiments as they visually highlight the most frequently occurring words, providing quick insights into the key themes and emotions in textual data.")

    col1,col2 = st.columns(2)
    

    with col1:
        st.write("Word Cloud for Positive Sentiment:")
        all_words_positive = ' '.join([text for text in combined_df['Tidy_Tweets'][combined_df['sentiment'] == 'POS']])
        Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))
        image_colors = ImageColorGenerator(Mask)
        wc_positive = WordCloud(background_color='black', height=1500, width=4000, mask=Mask).generate(all_words_positive)
        st.image(wc_positive.recolor(color_func=image_colors).to_array(), use_column_width=True, caption="Positive Sentiment")

    # Word Cloud for negative sentiment
    with col2:
        st.write("Word Cloud for Negative Sentiment:")
        all_words_negative = ' '.join([text for text in combined_df['Tidy_Tweets'][combined_df['sentiment'] == 'NEG']])
        Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))
        image_colors = ImageColorGenerator(Mask)
        wc_negative = WordCloud(background_color='black', height=1500, width=4000, mask=Mask).generate(all_words_negative)
        st.image(wc_negative.recolor(color_func=image_colors).to_array(), use_column_width=True, caption="Negative Sentiment")

    # Top 10 Hashtags Section
    st.markdown("## Top 10 Hashtags")
    st.subheader("Top 10 Hashtags for Sentiments")
    ht_positive = Hashtags_Extract(combined_df['Tidy_Tweets'][combined_df['sentiment'] == 'POS'])
    ht_negative = Hashtags_Extract(combined_df['Tidy_Tweets'][combined_df['sentiment'] == 'NEG'])

    # Unnesting list
    ht_positive_unnest = sum(ht_positive, [])
    ht_negative_unnest = sum(ht_negative, [])

    # Counting the frequency of words having positive sentiment
    word_freq_positive = nltk.FreqDist(ht_positive_unnest)
    df_positive = pd.DataFrame({'Hashtags': list(word_freq_positive.keys()), 'Count': list(word_freq_positive.values())})
    df_positive_plot = df_positive.nlargest(10, columns='Count')

    col3,col4 = st.columns(2)
    with col3:
        # Plotting the barplot for the most frequently used words in hashtags for positive sentiment
        #st.write("Top 10 Hashtags for Positive Sentiment:")
        # Create a bar plot using Plotly Express
        fig = px.bar(df_positive_plot, x='Count', y='Hashtags', orientation='h', title='Top 10 Hashtags for Positive Sentiment')
        
        # Customize the layout if needed
        fig.update_layout(xaxis_title='Count', yaxis_title='Hashtags', xaxis_ticks="outside")
        
        # Display the plotly chart
        st.plotly_chart(fig)

    # Counting the frequency of words having negative sentiment
    word_freq_negative = nltk.FreqDist(ht_negative_unnest)
    df_negative = pd.DataFrame({'Hashtags': list(word_freq_negative.keys()), 'Count': list(word_freq_negative.values())})

    with col4:
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
    sentiment_mapping = {'POS': 0, 'NEG': 1, 'NEU': 2}

    # Map the 'sentiment' column to numerical labels
    combined_df['sentiment_label'] = combined_df['sentiment'].map(sentiment_mapping)

    #return combined_df
    # Store the combined_df in session_state for access on other pages
    st.session_state.combined_df = combined_df

def sentiment_analysis_page():
    # Access the combined_df from session_state
    combined_df = st.session_state.combined_df
    st.header("Sentiment Analysis")
    st.subheader("Analyze sentiment of Twitter data")

    st.markdown("### Bag-of-Words Model")
    st.write("In this section, we analyze sentiment using a Bag-of-Words model.")

    
    bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')


    # bag-of-words feature matrix
    bow = bow_vectorizer.fit_transform(combined_df['Tidy_Tweets'])
    df_bow = pd.DataFrame(bow.todense())
    tfidf = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(combined_df['Tidy_Tweets'])
    df_tfidf = pd.DataFrame(tfidf_matrix.todense())

    train_bow = bow[:8000]
    train_bow.todense() 

    train_tfidf_matrix = tfidf_matrix[:8000]
    train_tfidf_matrix.todense()

    x_train_bow, x_valid_bow, y_train_bow, y_valid_bow = train_test_split(
    train_bow, 
    combined_df['sentiment_label'][:8000], 
    test_size=0.3, 
    random_state=2)

    x_train_tfidf, x_valid_tfidf, y_train_tfidf, y_valid_tfidf = train_test_split(
    train_tfidf_matrix, 
    combined_df['sentiment_label'][:8000], 
    test_size=0.3, 
    random_state=17)

    y_train_bow.fillna(0, inplace=True)
    y_valid_bow.fillna(0, inplace=True)

    st.write("Training the Logistic Regression model...")
    # Logistic Regression model on BOW features
    logreg = LogisticRegression()
    logreg.fit(x_train_bow, y_train_bow) # training the model
    prediction_bow = logreg.predict_proba(x_valid_bow) # predicting on the validation set

    #calculating the f1-score
    prediction_int = prediction_bow[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0

    #converting the results to integer type
    prediction_int = prediction_int.astype(int)

    log_bow= f1_score(y_valid_bow, prediction_int, average='weighted') # calculating f1 score
    acc_log_bow = accuracy_score(y_valid_bow, prediction_int) # calculating accuracy score

    #print("F1-Score for logistic regression BOW is: ",log_bow)
    st.write("F1-Score for logistic regression BOW is: ",round(log_bow,2))

    #print("Accuracy for logistic regression BOW is: ",acc_log_bow)
    st.write("Accuracy for logistic regression BOW is: ",round(acc_log_bow,2))

    y_train_tfidf.fillna(0, inplace=True)
    y_valid_tfidf.fillna(0, inplace=True)

    #fitting the model with TFIDF features
    logreg.fit(x_train_tfidf, y_train_tfidf)

    #predicting on the validation set
    prediction_tfidf = logreg.predict_proba(x_valid_tfidf)

    #calculating the f1-score
    prediction_int = prediction_tfidf[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0

    prediction_int = prediction_int.astype(int)

    log_tfidf= f1_score(y_valid_tfidf, prediction_int, average='weighted') # calculating f1 score
    acc_log_tfidf = accuracy_score(y_valid_tfidf, prediction_int) # calculating accuracy score
    
    #print("F1-Score for logistic regression TFIDF is: ",log_tfidf)
    #st.write("F1-Score for logistic regression TFIDF is: ",log_tfidf)

    #print("Accuracy for logistic regression TFIDF is: ",acc_log_tfidf)
    #st.write("Accuracy for logistic regression TFIDF is: ",acc_log_tfidf)



    # SVM 
    svm = SVC(C=1.0, kernel='linear', degree=3, gamma='auto',probability=True)

    svm.fit(x_train_bow, y_train_bow) # training the model
    prediction_bow = svm.predict_proba(x_valid_bow) # predicting on the validation set

    #calculating the f1-score
    prediction_int = prediction_bow[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0

    #converting the results to integer type
    prediction_int = prediction_int.astype(int)

    svm_bow = f1_score(y_valid_bow, prediction_int, average='weighted') # calculating f1 score
    svm_acc_bow = accuracy_score(y_valid_bow, prediction_int) # calculating accuracy score

    #print("F1-Score for SVM BOW is: ",svm_bow)
    #st.write("F1-Score for SVM BOW is: ",svm_bow)

    #print("Accuracy for SVM BOW is: ",svm_acc_bow)
    #st.write("Accuracy for SVM BOW is: ",svm_acc_bow)



    svm.fit(x_train_tfidf, y_train_tfidf) # training the model
    prediction_tfidf = svm.predict_proba(x_valid_tfidf) # predicting on the validation set

    prediction_int = prediction_tfidf[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
    prediction_int = prediction_int.astype(int)

    svm_tfidf = f1_score(y_valid_tfidf, prediction_int, average='weighted') # calculating f1 score
    svm_acc_tfidf = accuracy_score(y_valid_tfidf, prediction_int) # calculating accuracy score

    #print("F1-Score for SVM TFIDF is: ",svm_tfidf)
    #st.write("F1-Score for SVM TFIDF is: ",svm_tfidf)

    #print("Accuracy for SVM TFIDF is: ",svm_acc_tfidf)
    #st.write("Accuracy for SVM TFIDF is: ",svm_acc_tfidf)



    #Naive Bayes
    from sklearn import naive_bayes
    nb = naive_bayes.MultinomialNB()
    nb.fit(x_train_bow, y_train_bow) # training the model

    prediction_bow = nb.predict_proba(x_valid_bow) # predicting on the validation set
    prediction_int = prediction_bow[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
    prediction_int = prediction_int.astype(int)

    nb_bow = f1_score(y_valid_bow, prediction_int, average='weighted') # calculating f1 score
    nb_acc_bow = accuracy_score(y_valid_bow, prediction_int) # calculating accuracy score

    #print("F1-Score for Naive Bayes BOW is: ",nb_bow)
    #st.write("F1-Score for Naive Bayes BOW is: ",nb_bow)

    #print("Accuracy for Naive Bayes BOW is: ",nb_acc_bow)
    #st.write("Accuracy for Naive Bayes BOW is: ",nb_acc_bow)


    nb.fit(x_train_tfidf, y_train_tfidf) # training the model
    prediction_tfidf = nb.predict_proba(x_valid_tfidf) # predicting on the validation set

    prediction_int = prediction_tfidf[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
    prediction_int = prediction_int.astype(int)

    nb_tfidf = f1_score(y_valid_tfidf, prediction_int, average='weighted') # calculating f1 score
    nb_acc_tfidf = accuracy_score(y_valid_tfidf, prediction_int) # calculating accuracy score

    #print("F1-Score for Naive Bayes TFIDF is: ",nb_tfidf)
    #st.write("F1-Score for Naive Bayes TFIDF is: ",nb_tfidf)

    #print("Accuracy for Naive Bayes TFIDF is: ",nb_acc_tfidf)
    #st.write("Accuracy for Naive Bayes TFIDF is: ",nb_acc_tfidf)

    

    dct = DecisionTreeClassifier(criterion='entropy', random_state=1)

    dct.fit(x_train_bow, y_train_bow) # training the model

    dct_bow = dct.predict_proba(x_valid_bow) # predicting on the validation set

    #calculating the f1-score
    dct_bow = dct_bow[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
    dct_int_bow = dct_bow.astype(int)

    dct_score_bow = f1_score(y_valid_bow, dct_int_bow, average='weighted') # calculating f1 score
    dct_acc_bow = accuracy_score(y_valid_bow, dct_int_bow) # calculating accuracy score

    #print("F1-Score for Decision Tree BOW is: ",dct_score_bow)
    #st.write("F1-Score for Decision Tree BOW is: ",dct_score_bow)

    #print("Accuracy for Decision Tree BOW is: ",dct_acc_bow)
    #st.write("Accuracy for Decision Tree BOW is: ",dct_acc_bow)


    dct.fit(x_train_tfidf, y_train_tfidf) # training the model
    dct_tfidf = dct.predict_proba(x_valid_tfidf) # predicting on the validation set
    #calculating the f1-score
    dct_tfidf = dct_tfidf[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
    dct_int_tfidf = dct_tfidf.astype(int)

    dct_score_tfidf = f1_score(y_valid_tfidf, dct_int_tfidf, average='weighted') # calculating f1 score
    dct_acc_tfidf = accuracy_score(y_valid_tfidf, dct_int_tfidf) # calculating accuracy score

    #print("F1-Score for Decision Tree TFIDF is: ",dct_score_tfidf)
    #st.write("F1-Score for Decision Tree TFIDF is: ",dct_score_tfidf)

    #print("Accuracy for Decision Tree TFIDF is: ",dct_acc_tfidf)
    #st.write("Accuracy for Decision Tree TFIDF is: ",dct_acc_tfidf)

    #Model comparison
    Algo_1_bow = ['LogisticRegression(Bag-of-Words)','SVM(Bag-of-Words)','Naive Bayes(Bag-of-Words)','Decision Tree(Bag-of-Words)']
    score_1_bow = [log_bow,svm_bow,nb_bow,dct_score_bow]

    compare_1_bow = pd.DataFrame({'Model':Algo_1_bow,'F1_Score':score_1_bow},index=[i for i in range(1,5)])
    
    Algo_1_tfidf = ['LogisticRegression(TF-IDF)','SVM(TF-IDF)','Naive Bayes(TF-IDF)','Decision Tree(TF-IDF)']
    score_1_tfidf = [log_tfidf,svm_tfidf,nb_tfidf,dct_score_tfidf]

    compare_1_tfidf = pd.DataFrame({'Model':Algo_1_tfidf,'F1_Score':score_1_tfidf},index=[i for i in range(1,5)])

    st.write("Comparison of Models using Bag-of-Words(F1-Score)")

    st.write(compare_1_bow)

    st.write("Comparison of Models using TF-IDF(F1-Score)")

    st.write(compare_1_tfidf)

    Algo_1_acc_bow = ['LogisticRegression','SVM','Naive Bayes','Decision Tree']
    score_1_acc_bow = [acc_log_bow,svm_acc_bow,nb_acc_bow,dct_acc_bow]

    compare_1_acc_bow = pd.DataFrame({'Model':Algo_1_acc_bow,'Accuracy':score_1_acc_bow},index=[i for i in range(1,5)])

    Algo_1_acc_tfidf = ['LogisticRegression','SVM','Naive Bayes','Decision Tree']
    score_1_acc_tfidf = [acc_log_tfidf,svm_acc_tfidf,nb_acc_tfidf,dct_acc_tfidf]

    compare_1_acc_tfidf = pd.DataFrame({'Model':Algo_1_acc_tfidf,'Accuracy':score_1_acc_tfidf},index=[i for i in range(1,5)])

    

    st.write("Comparison of Models using Bag-of-Words(Accuracy)")

    st.write(compare_1_acc_bow)
    

    st.write("Comparison of Models using TF-IDF(Accuracy)")

    st.write(compare_1_acc_tfidf)


    # Plot the DataFrame using a bar chart
    st.write("Bar Chart  for Bag-of-Words(F1_SCORE):")
    # Create a bar chart using Plotly Express
    fig = px.bar(compare_1_bow, x='Model', y='F1_Score',
                  title='Bar Chart for Bag-of-Words (F1 Score)')
    
    # Customize the layout if needed
    fig.update_xaxes(title_text='Category')
    fig.update_yaxes(title_text='Values')
    
    # Display the Plotly chart
    st.plotly_chart(fig)

    # Plot the DataFrame using a bar chart
    st.write("Bar Chart  for Bag-of-Words (Accuracy):")
    fig = px.bar(compare_1_acc_bow, x='Model', y='Accuracy',
                  title='Bar Chart for Bag-of-Words (Accuracy)')
    
    # Customize the layout if needed
    fig.update_xaxes(title_text='Category')
    fig.update_yaxes(title_text='Values')
    
    # Display the Plotly chart
    st.plotly_chart(fig)

    # Plot the DataFrame using a bar chart
    st.write("Bar Chart  for TFIDF(F1_SCORE):")
    fig = px.bar(compare_1_tfidf, x='Model', y='F1_Score',
                  title='Bar Chart for TFIDF (F1 Score)')
    
    # Customize the layout if needed
    fig.update_xaxes(title_text='Category')
    fig.update_yaxes(title_text='Values')
    
    # Display the Plotly chart
    st.plotly_chart(fig)


    # Plot the DataFrame using a bar chart
    st.write("Bar Chart  for TFIDF (Accuracy):")
    fig = px.bar(compare_1_acc_tfidf, x='Model', y='Accuracy',
                  title='Bar Chart for TFIDF (Accuracy)')
    
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
#if __name__ == "__main__":
#    combined_df = data_analysis_page()
#    sentiment_analysis_page(combined_df)



