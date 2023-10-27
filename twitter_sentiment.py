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
import re
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import joblib  # Import joblib to load the fitted CountVectorizer


nltk.download('stopwords')

#from transformers import pipeline
st.set_option('deprecation.showPyplotGlobalUse', False)

#sentiment_analysis = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
def remove_pattern(input_text, pattern):
    # Check if the input_text is a string, and if not, convert it to a string
    if not isinstance(input_text, str):
        input_text = str(input_text)
    
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
    st.header("Data Evaluation")
    st.subheader("Combine and analyze Twitter and Facebook data")

    # Load CSV files directly into a DataFrame
    combined_df = pd.read_csv('merged_df.csv')

    

    # Display your data analysis results using Streamlit widgets
    # Data Overview Section
    st.markdown("## Data Overview")
    st.write("Here's a sample of our Facebook data:")
    code = """
    combined_df.sample(5)
    """
    st.code(code,language='python')

    st.dataframe(combined_df.loc[205:210,:])

    st.write("Here's a sample of our Twitter data:")
    code = """
    combined_df.sample(5)
    """
    st.code(code,language='python')
    st.dataframe(combined_df.loc[3000:3005,:])
  # Sentiment Distribution Section
    # Sentiment Distribution Pie Chart
    st.subheader("Sentiment Distribution")

    code ="""
    sentiment_counts = combined_df['sentiment'].value_counts()
    fig = px.pie(sentiment_counts, values=sentiment_counts, 
                 names=sentiment_counts.index, title='Sentiment Distribution')

    """
    st.code(code,language="python")
    sentiment_counts = combined_df['sentiment'].value_counts()
    fig = px.pie(sentiment_counts, values=sentiment_counts, 
                 names=sentiment_counts.index, title='Sentiment Distribution')
    st.plotly_chart(fig)


    st.markdown("## Assessing Sentence Length")

    code ="""

        num_samples = len(combined_df)

    # Handle missing values (NaN) in the 'full_text' column by filling them with empty strings
    combined_df['full_text'].fillna('', inplace=True)

    # Calculate average sentence length
    combined_df['sentence_length'] = combined_df['full_text'].apply(lambda x: len(x.split()))
    average_sentence_length = combined_df['sentence_length'].mean()



        # Visualize data distribution using histograms or other plots
    plt.figure(figsize=(10, 5))

    # Plot a histogram of sentence lengths
    plt.subplot(1, 2, 1)
    sns.histplot(combined_df['sentence_length'], bins=20, kde=True)
    plt.title('Sentence Length Distribution')
    plt.xlabel('Sentence Length')
    plt.ylabel('Frequency')
    """

    st.code(code,language="python")
    #number of samples
    num_samples = len(combined_df)

    # Handle missing values (NaN) in the 'full_text' column by filling them with empty strings
    combined_df['full_text'].fillna('', inplace=True)

    # Calculate average sentence length
    combined_df['sentence_length'] = combined_df['full_text'].apply(lambda x: len(x.split()))
    average_sentence_length = combined_df['sentence_length'].mean()



        # Visualize data distribution using histograms or other plots
    plt.figure(figsize=(10, 5))

    # Plot a histogram of sentence lengths
    plt.subplot(1, 2, 1)
    sns.histplot(combined_df['sentence_length'], bins=20, kde=True)
    plt.title('Sentence Length Distribution')
    plt.xlabel('Sentence Length')
    plt.ylabel('Frequency')

    # Display the plot in Streamlit
    st.pyplot(plt)

    # Text Preprocessing Section
    st.markdown("## Text Preprocessing")
    # Find the number of tweets per day and plot it

    code ="""
    combined_df['tidy_text'] = np.vectorize(remove_pattern)(combined_df['full_text'], "@[\w]*")

    """

    st.code(code,language="python")
    # Remove Twitter handles (@user)
    combined_df['tidy_text'] = np.vectorize(remove_pattern)(combined_df['full_text'], "@[\w]*")

    # Step 1: Removing Twitter Handles
    st.write("1. **Removing Twitter Handles:** In text preprocessing, we often remove Twitter handles, which start with the '@' symbol, to ensure that they don't interfere with our analysis.")

    st.write(combined_df[['full_text', 'tidy_text']].head())

    
    # Step 2: Removing Special Characters, Numbers, Punctuations
    st.write("2. **Removing Special Characters, Numbers, Punctuations:** We eliminate special characters, numbers, and punctuation marks from the text. This helps in focusing on the actual words and their meaning.")
    
    code ="""
    combined_df['tidy_text'] = combined_df['tidy_text'].str.replace("[^a-zA-Z#]", " ")

    """

    st.code(code,language="python")
    combined_df['tidy_text'] = combined_df['tidy_text'].str.replace("[^a-zA-Z#]", " ")
    st.write(combined_df[['full_text', 'tidy_text']].head())

    # Step 3: Removing Short Words
    st.write("3. **Removing Short Words:** Short words like 'a,' 'an,' 'the,' etc., don't usually provide much information. Removing them can make text analysis more meaningful.")
    code ="""
    combined_df['tidy_text'] = combined_df['tidy_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 1]))
    """

    st.code(code,language="python")
    combined_df['tidy_text'] = combined_df['tidy_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 1]))
    st.write(combined_df[['full_text', 'tidy_text']].head())

    
    # Step 4: Tokenization
    st.write("4. **Tokenization:** Tokenization involves splitting text into individual words or 'tokens.' It's a crucial step for many natural language processing tasks as it breaks down text into manageable units.")
    code ="""
    tokenized_tweet = combined_df['tidy_text'].apply(lambda x: x.split())
    """

    st.code(code,language="python")
    tokenized_tweet = combined_df['tidy_text'].apply(lambda x: x.split())
    st.write(tokenized_tweet.head())

    # Step 5: Stemming
    st.write("5. **Stemming:** Stemming is a text normalization technique that reduces words to their root form. For example, 'running' and 'ran' are stemmed to 'run.' This helps in treating similar words as the same for analysis.")
    code ="""
    stemmer = PorterStemmer()
    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])

    """

    st.code(code,language="python")
    stemmer = PorterStemmer()
    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])
    st.write(tokenized_tweet.head())

    # Stitching tokens back together
    st.write("Stitching Tokens Back Together")
    code ="""
        for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

    combined_df['tidy_text'] = tokenized_tweet
    """

    st.code(code,language="python")
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


    bow_f1_score_df = pd.read_csv('bow_f1_score_df.csv')
    tfidf_f1_score_df = pd.read_csv('tfidf_f1_score_df.csv')

    st.header("Model Development")

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

    code ="""
    
        from  sklearn.naive_bayes import ComplementNB,BernoulliNB

        model_dir ="saved_models/"

        # Create dictionaries to store F1-Scores and Accuracy for each classifier
        bow_f1_score = {}
        bow_accuracy_score = {}
        saved_models ={}

        # Create a dictionary of classifiers
        classifiers = {
            'Mulitinomial Naive Bayes': MultinomialNB(),
            'SVM': SVC(probability=True),  # Set probability=True for SVC
        }


        # Iterate through each classifier
        for classifier_name, classifier in classifiers.items():
            # Fit the model on the training data
            classifier.fit(x_train_bow, y_train_bow)

            # Predict on the validation set
            if 'SVM' in classifier_name:
                prediction = classifier.predict(x_valid_bow)  # Use predict for SVM
            else:
                prediction = classifier.predict(x_valid_bow)

            # Calculate the F1-Score and Accuracy for each class
            f1 = np.round(f1_score(y_valid_bow, prediction, average=None), 2)
            accuracy = np.round(accuracy_score(y_valid_bow, prediction), 2)

            # Store the F1-Score and Accuracy in the dictionaries
            bow_f1_score[classifier_name] = f1
            bow_accuracy_score[classifier_name] = accuracy

            # Print the F1-Score and Accuracy for each class
            print(f"{classifier_name}:")
            for class_idx, f1_score_class in enumerate(f1):
                print(f"Class {class_idx}: F1-Score: {f1_score_class}")
            print(f"Accuracy: {accuracy}")
            print()

            # Save the model
            model_filename =f"{model_dir}{classifier_name}_bow_model.joblib"
            joblib.dump(classifier, model_filename)

            # Save the model in the dictionary
            saved_models[classifier_name] = model_filename
            print(f"{classifier_name} model saved to {model_filename}")


        # Print the dictionaries containing F1-Scores and Accuracy
        print("F1-Scores for BoW:")
        print(bow_f1_score)
        print("Accuracy Scores for BoW:")
        print(bow_accuracy_score)

        Mulitinomial Naive Bayes:
        Class 0: F1-Score: 0.79
        Class 1: F1-Score: 0.6
        Class 2: F1-Score: 0.67
        Accuracy: 0.71

        Mulitinomial Naive Bayes model saved to saved_models/Mulitinomial Naive Bayes_bow_model.joblib
        SVM:
        Class 0: F1-Score: 0.82
        Class 1: F1-Score: 0.67
        Class 2: F1-Score: 0.76
        Accuracy: 0.77

        SVM model saved to saved_models/SVM_bow_model.joblib
        F1-Scores for BoW:
        {'Mulitinomial Naive Bayes': array([0.79, 0.6 , 0.67]), 'SVM': array([0.82, 0.67, 0.76])}
        Accuracy Scores for BoW:
        {'Mulitinomial Naive Bayes': 0.71, 'SVM': 0.77}

    """
    st.code(code, language="python")

    st.write("Here are the results of the models trained on the BoW vectors:")
    st.write(bow_accuracy_df)

     # Plot the DataFrame using a bar chart
    # Create a bar chart using Plotly Express
    fig = px.bar(bow_accuracy_df, x='index', y='Accuracy',
                  title='Bar Chart for Bag-of-Words ')
    
    # Customize the layout if needed
    fig.update_xaxes(title_text='Category')
    fig.update_yaxes(title_text='Values')
    
    # Display the Plotly chart
    st.plotly_chart(fig)

    #f1-score
    st.write("F1-Score for Bag-of-Words:")
    st.write(bow_f1_score_df)

    st.write("Bar Chart for F1-Scores:")
    fig = px.bar(bow_f1_score_df, x='Models', y=['Negative', 'Neutral', 'Positive'], title='Bar Chart for F1-Scores')
    fig.update_xaxes(title_text='Models')
    fig.update_yaxes(title_text='F1-Score')

    st.plotly_chart(fig)





    # TF-IDF Section
    st.markdown("## TF-IDF")
    st.write("TF-IDF (Term Frequency-Inverse Document Frequency) is a technique for extracting features from text data. It involves creating a vocabulary of all the unique words in the data and then creating vectors of word counts for each document in the data.")

    code = """

        model_dir ="saved_models/"
        # Create dictionaries to store F1-Scores and Accuracy for each classifier
        tfidf_f1_score = {}
        tfidf_accuracy_score = {}
        saved_models={}

        # Create a dictionary of classifiers
        classifiers = {
            'SVM': SVC(probability=True),  # Set probability=True for SVC
            'Multinomial Naive Bayes': MultinomialNB(),
        }
        # Iterate through each classifier
        for classifier_name, classifier in classifiers.items():
            # Fit the model on the training data
            classifier.fit(x_train_bow, y_train_bow)

            # Predict on the validation set
            if 'SVM' in classifier_name:
                prediction = classifier.predict(x_valid_tfidf)  # Use predict for SVM
            else:
                prediction = classifier.predict(x_valid_tfidf)

            # Calculate the F1-Score and Accuracy for each class
            f1 = np.round(f1_score(y_valid_tfidf, prediction, average=None), 2)
            accuracy = np.round(accuracy_score(y_valid_tfidf, prediction), 2)

            # Store the F1-Score and Accuracy in the dictionaries
            tfidf_f1_score[classifier_name] = f1
            tfidf_accuracy_score[classifier_name] = accuracy

            # Print the F1-Score and Accuracy for each class
            print(f"{classifier_name}:")
            for class_idx, f1_score_class in enumerate(f1):
                print(f"Class {class_idx}: F1-Score: {f1_score_class}")
            print(f"Accuracy: {accuracy}")
            print()

            # Save the model
            model_filename =f"{model_dir}{classifier_name}_tfidf_model.joblib"
            joblib.dump(classifier, model_filename)

            # Save the model in the dictionary
            saved_models[classifier_name] = model_filename
            print(f"{classifier_name} model saved to {model_filename}")


        # Print the dictionaries containing F1-Scores and Accuracy
        print("F1-Scores for TF-IDF:")
        print(tfidf_f1_score)
        print("Accuracy Scores for TF-IDF:")
        print(tfidf_accuracy_score)

        SVM:
        Class 0: F1-Score: 0.47
        Class 1: F1-Score: 0.15
        Class 2: F1-Score: 0.63
        Accuracy: 0.54

        SVM model saved to saved_models/SVM_tfidf_model.joblib
        Multinomial Naive Bayes:
        Class 0: F1-Score: 0.79
        Class 1: F1-Score: 0.62
        Class 2: F1-Score: 0.69
        Accuracy: 0.73

        Multinomial Naive Bayes model saved to saved_models/Multinomial Naive Bayes_tfidf_model.joblib
        F1-Scores for TF-IDF:
        {'SVM': array([0.47, 0.15, 0.63]), 'Multinomial Naive Bayes': array([0.79, 0.62, 0.69])}
        Accuracy Scores for TF-IDF:
        {'SVM': 0.54, 'Multinomial Naive Bayes': 0.73}

    """
    st.code(code, language="python")
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

    #f1-score

    st.write("F1-Score for TF-IDF:")
    st.write(tfidf_f1_score_df)

    # Plot the DataFrame using a bar chart
    st.write("Bar Chart for TF-IDF:")
    fig = px.bar(tfidf_f1_score_df, x='Models', y=['Negative', 'Neutral', 'Positive'], title='Bar Chart for F1-Scores')
    fig.update_xaxes(title_text='Models')
    fig.update_yaxes(title_text='F1-Score')

    # Display the Plotly chart
    st.plotly_chart(fig)

    # Conclusion

    st.write("In summary, BoW and TF-IDF are essential tools for sentiment analysis, providing structured and interpretable features that help classify sentiment in text data.")


    st.write("Among the models trained, SVM has shown higher accuracy and F1-scores compared to Multinomial Naive Bayes. SVM is a powerful classifier that works well in high-dimensional spaces and is effective for text classification tasks.")

    st.write("To further improve the model's performance, we  used the StackingClassifier, which combines multiple classifiers to enhance predictive accuracy and reduce overfitting. ")

    ##st.markdown("## BaggingClassifier for Model Improvement")

    code = """
    from sklearn.ensemble import BaggingClassifier
    bgc = BaggingClassifier(base_estimator=svc_bow, n_estimators=250, random_state=42)
    bgc.fit(x_train_bow, y_train_bow)

    prediction = bgc.predict(x_valid_bow)
    print("F1-Score:", f1_score(y_valid_bow, prediction, average=None))
    print("Accuracy-score", round(accuracy_score(y_valid_bow, prediction), 2))

    F1-Score: [0.81372855 0.67669173 0.75587467]
    Accuracy-score 0.77
    """

    st.code(code, language="python")
        
    st.title("Why Stacking Helps Improve Predictions")

    st.write("Stacking is a machine learning technique that combines the predictions of different models to improve accuracy.")
    st.write("Here's how it can help:")

    st.markdown("1. **Diverse Models:** Stacking uses a mix of models with different strengths and weaknesses. Each model may find different patterns in the data.")

    st.markdown("2. **Reduced Overfitting:** By combining multiple models, stacking reduces the risk of overfitting. It generalizes better to new data.")

    st.markdown("3. **Improved Accuracy:** Stacking's final estimator optimally weighs the predictions, resulting in more accurate and reliable predictions.")

    st.markdown("4. **Robustness:** If one model struggles on specific data, others may compensate, making the model more robust.")

    st.title("Stacking Ensemble Results")

    st.write("In our specific implementation, the stacking ensemble achieved:")
    st.write("- An Accuracy score  of approximately 0.78")

    st.write("This improvement demonstrates that the stacking technique:")
    st.write("- Combines the strengths of different classifiers")
    st.write("- Produces better predictions for our problem")


    st.title("Cross Validation for Stacking Ensemble")

    code =("""
            #cross validation
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(clf, x_train_bow, y_train_bow, cv=5)

        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        Accuracy: 0.78 (+/- 0.01)
            """)

    st.code(code,language='python')

    # Your cross-validation results
    mean_accuracy = 0.78
    std_deviation = 0.01

    # Explanation
    st.write("Cross-validation is a technique that helps assess and improve the performance of a machine learning model.")
    st.write("In our case, the model's accuracy was evaluated using cross-validation with 5 folds.")
    st.write(f"The model achieved an average accuracy of {mean_accuracy:.2f}, which means it correctly predicted 78% of the instances on average.")
    st.write(f"The standard deviation of {std_deviation:.2f} indicates how much the accuracy scores varied across different subsets of the data.")
    st.write("A smaller standard deviation suggests more stable and consistent model performance, which is desirable.")
    st.write("Cross-validation is essential for reducing overfitting, optimizing hyperparameters, and gaining confidence in your model's performance.")
    
    
    st.title("Model Deployment -Try the Model")
        ##st.title("Try the Model")

    st.write("Now that we've trained our model, let's try it out!")

    # Collect user input
    user_input = st.text_input("Enter a text to predict sentiment")

    if st.button("Predict Sentiment"):
        if user_input:
            # Call the predict_sentiment function
            predicted_sentiment = predict_sentiment(user_input)

            # Display the predicted sentiment
            st.write(f"Predicted sentiment: **{predicted_sentiment}**")


def further_research():
    # Streamlit app section
    st.markdown("## Further Research and Actionable Insights")

    st.write("In our ongoing quest to improve sentiment analysis, there are several avenues for further research and actionable insights. Here are some next steps to consider:")

    # List of next steps and actionable insights
    next_steps = [
        "1. **Fine-Tuning Hyperparameters:** Experiment with different hyperparameters for the SVM and BaggingClassifier to optimize model performance.",
        "2. **Feature Engineering:** Explore advanced text preprocessing techniques, such as word embeddings or word2vec, to capture more nuanced text representations.",
        "3. **Ensemble Methods:** Try other ensemble methods like Random Forest, AdaBoost, or Stacking, and evaluate their impact on the model.",
        "4. **Text Augmentation:** Consider data augmentation techniques to increase the size and diversity of your training dataset, which can lead to improved generalization.",
        "5. **Model Interpretability:** Implement techniques for model interpretability, such as LIME or SHAP, to gain insights into why the model makes certain predictions.",
        "6. **Continuous Monitoring:** Continuously monitor model performance and retrain it with new data to adapt to evolving language and trends.",
        "7. **User Feedback:** Collect feedback from users of your sentiment analysis tool to understand their needs and refine the model accordingly.",
    ]

    # Display the next steps and insights
    for step in next_steps:
        st.write(step)


# Load the Bagging Classifier model
model = joblib.load('final_model.joblib')

# Load the fitted CountVectorizer from your training code
vectorizer = joblib.load('bow_vectorizer.joblib')



def preprocess_user_input(input_text):

    # Remove special characters, numbers, and punctuations
    cleaned_text = re.sub("[^a-zA-Z#]", " ", input_text)

    # Tokenization (split the text into words)
    tokenized_text = cleaned_text.split()

    # Stemming (reduce words to their root form)
    stemmer = PorterStemmer()
    stemmed_text = [stemmer.stem(word) for word in tokenized_text]

    # Combine the processed words back into a single string
    processed_input = ' '.join(stemmed_text)

    # Vectorize the processed input using the fitted CountVectorizer
    processed_input_vectorized = vectorizer.transform([processed_input])

    return processed_input_vectorized



def predict_sentiment(input_text):

    

    # Preprocess the user input (cleaning, tokenization, etc.) to match the format used during model training
    preprocessed_input = preprocess_user_input(input_text)

    # Use the loaded model to make a prediction
    sentiment_label = model.predict(preprocessed_input)[0]  # Extract the first element

    # Map the numeric label back to 'positive', 'negative', or 'neutral'
    sentiment_mapping = {0: 'positive', 1: 'negative', 2: 'neutral'}
    predicted_sentiment = sentiment_mapping[sentiment_label]

    return predicted_sentiment

# Create a sidebar navigation
app_pages = {
    "Home": home_page,
    "Data Evaluation": data_analysis_page,
    "Model Development": sentiment_analysis_page,
    "Further Research": further_research,
}

# Add a Streamlit sidebar for navigation
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to", list(app_pages.keys()))

# Display the selected page
app_pages[selected_page]()



