import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import plotly.express as px

import warnings
warnings.filterwarnings("ignore")
import spacy

from sklearn.metrics import accuracy_score

import pickle
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
#from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.colors as mcolors
from wordcloud import WordCloud, STOPWORDS
from gensim.models import LdaModel
from textblob import TextBlob
from sklearn import metrics
from sklearn.metrics import accuracy_score
import unidecode
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Data Science app')

image = Image.open('ds.jpg')
st.image(image, use_column_width=True)




def data_preprocessing(text):   
    from cleantext import clean
    clean_text = clean(text,
     fix_unicode=True, 
      to_ascii=True, 
      lower=True, 
      no_line_breaks=True,
      no_urls=True, 
      no_numbers=True, 
      no_digits=True, 
      no_currency_symbols=True, 
      no_punct=True, 
      replace_with_punct="", 
      replace_with_url="", 
      replace_with_number="", 
      replace_with_digit="", 
      replace_with_currency_symbol="",
      lang='en')
    return clean_text


#"""remove accented characters from text, e.g. café"""
def remove_accented_chars(text):
    text = unidecode.unidecode(text)
    return text


def load_prediction_models(model_file):
	loaded_model = pickle.load(open(model_file,'rb'))
	return loaded_model


def polarity(text): 
    return TextBlob(text).sentiment.polarity


def getAnalysis(score):
        if score < -0.13:
            return 'Negative'
        elif score > 0.13:
            return 'Positive'
        else:
            return 'Neutral'
    

def Sentiment_Prediction(data,model_name):
    
    data['Customer_Reviews'] = data['Customer_Reviews'].apply(lambda x: " ".join(x for x in str(x).split()))
    data['Customer_Reviews'] = data['Customer_Reviews'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
    data['Customer_Reviews'] = data['Customer_Reviews'].apply(data_preprocessing)
    data['Customer_Reviews']= data['Customer_Reviews'].apply(remove_accented_chars)
    vectorizer = pickle.load(open('tf_idf_vectorizer.pkl', 'rb'))
    Vectors = vectorizer.transform(data['Customer_Reviews'])
    predictor = load_prediction_models(model_name)
    if model_name != 'GNB.pkl':
        data['Sentiment'] = predictor.predict(Vectors)    
    else:
        #Vectors.toarray()
        data['Sentiment'] = predictor.predict(Vectors.toarray()) 
    return data

def main():
    
    activities = ['Exploratory Data Analysis', 'ML Models for Sentiment Polarity', 'LDA Topic Extraction', 'Descriptive Analysis', 'Our Models vs Textblob', 'Text Prediction', 'About Us']
    option = st.sidebar.selectbox('Select Options: ', activities)
    
    #EDA
    if option == 'Exploratory Data Analysis':
        
        st.subheader('Expolatory Data Analysis')
        data = st.file_uploader('Upoad Dataset: ',type = ['csv'])
        
        if data is not None:
            st.success('Data Successfully Loaded')
            df = pd.read_csv(data)
            st.dataframe(df.head(10))

            if st.checkbox('Display Shape'):
                st.write(df.shape)
			
            if st.checkbox('Display Columns'):
                st.write(df.columns)
			
            if st.checkbox('Select Multiple Columns'):
                selected_col = st.multiselect('Select Prefered Columns: ',df.columns)
                df1 = df[selected_col]
                st.dataframe(df1)

            if st.checkbox('Summary'):
                st.write(df.describe().T)

            if st.checkbox('Display Null Values'):
                st.write(df.isnull().sum())

        #if st.checkbox('Display Data Types'):
            #st.write(df.types)

        #if st.checkbox('Display Correlation of Data'):
            #st.write(df.corr())


	# VISUALIZATION

    elif option == 'ML Models for Sentiment Polarity':
		
        st.subheader('Machine Learning Models to find Sentiment Polarity of the Reviews')
        data = st.file_uploader('Upoad Dataset: ',type = ['csv'])

        if data is not None:
            
            st.success('Data Successfully Loaded')
            data = pd.read_csv(data)
            st.dataframe(data.head(5))
            
            
            activities1 = ['None','Support Vector Machine', 'Logistic Regression', 'Complement Naive Bayes', 'Multinomial Naive Bayes', 'Gaussian Naive Bayes', 'Bernoullis Naive Bayes']
            option1 = st.selectbox('Select Model:', activities1)
            
            if option1 == 'None':
                st.markdown('Chose a ML Model')
            
            if option1 == 'Support Vector Machine':
                model_name = "SVM.pkl"
                Sentiment_Prediction(data,model_name)
                
            if option1 == 'Logistic Regression':
                model_name = "LR.pkl"
                Sentiment_Prediction(data,model_name)
                
            if option1 == 'Complement Naive Bayes':
                model_name = "CNB.pkl"
                Sentiment_Prediction(data,model_name)
                
            if option1 == 'Multinomial Naive Bayes':
                model_name = "MNB.pkl"
                Sentiment_Prediction(data,model_name)
                
            if option1 == 'Gaussian Naive Bayes':
                model_name = "GNB.pkl"
                Sentiment_Prediction(data,model_name)
                
            if option1 == 'Bernoullis Naive Bayes':
                model_name = "BNB.pkl"
                Sentiment_Prediction(data,model_name)
                
                
            
            
            activities2 = ['None', 'Bar Plot', 'Pie Chart']
            option2 = st.selectbox('Select Graph:', activities2)
            
            if option2 == 'None':
                st.markdown("Choose a Graph Type")
            
            if(int(option2 == 'Bar Plot') & int(option1 != 'None')):
                st.markdown("### Number of Reviews by sentiment")
                sentiment_count = data['Sentiment'].value_counts()
                sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Reviews':sentiment_count.values})
                fig = px.bar(sentiment_count, x='Sentiment', y='Reviews', color='Reviews', height=500)
                st.plotly_chart(fig)
            #option2 == 'Pie Chart':    
            if (int(option2 == 'Pie Chart') & int(option1 != 'None')):
                st.markdown("### Number of Reviews by sentiment")
                sentiment_count = data['Sentiment'].value_counts()
                sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Reviews':sentiment_count.values})
                fig = px.pie(sentiment_count, values='Reviews', names='Sentiment')
                st.plotly_chart(fig)
                
                
            activities3 = ['None', 'Positive', 'Negative', 'Neutral']
            option3 = st.selectbox('Select Wordcloud:', activities3)
            
            if option3 == 'None':
                st.markdown('Choose WordCloud')
                
            if (int(option3 == 'Positive') &  int(option1 != 'None')):
                st.subheader('Word cloud for %s sentiment' % (option3))
                df = data[data['Sentiment']==option3]
                words = ' '.join(df['Customer_Reviews'])
                processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
                from wordcloud import WordCloud, STOPWORDS
                wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(processed_words)
                plt.imshow(wordcloud)
                plt.xticks([])
                plt.yticks([])
                st.pyplot()
                
            if (int(option3 == 'Negative') &  int(option1 != 'None')):
                st.subheader('Word cloud for %s sentiment' % (option3))
                df = data[data['Sentiment']==option3]
                words = ' '.join(df['Customer_Reviews'])
                processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT']) 
                from wordcloud import WordCloud, STOPWORDS
                wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(processed_words)
                plt.imshow(wordcloud)
                plt.xticks([])
                plt.yticks([])
                st.pyplot()
                
                
            if (int(option3 == 'Neutral') &  int(option1 != 'None')):
                st.subheader('Word cloud for %s sentiment' % (option3))
                df = data[data['Sentiment']==option3]
                words = ' '.join(df['Customer_Reviews'])
                processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
                from wordcloud import WordCloud, STOPWORDS
                wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(processed_words)
                plt.imshow(wordcloud)
                plt.xticks([])
                plt.yticks([])
                st.pyplot()
                
            #if (int(option1 == 'None') & int(option2 == 'Bar Plot') | int(option2 == 'Pie Chart') | int(option3 == 'Positive') | int(option3 == 'Negative') | int(option3 == 'Neutral')):
               # st.markdown("Select a Model First!!!")
                
            if (option1 == 'None'):
                st.markdown("Select a Model First!!")
                

        
            
				
			
            
            
    elif option == 'LDA Topic Extraction':
		
        st.subheader('Topic Extraction using LDA Allocation')
        data = st.file_uploader('Upoad Dataset: ',type = ['csv'])

        if data is not None:
            
            st.success('Data Successfully Loaded')
            data = pd.read_csv(data)
            st.dataframe(data.head(5))
            
            activities1 = ['None','Support Vector Machine', 'Logistic Regression', 'Complement Naive Bayes', 'Multinomial Naive Bayes', 'Gaussian Naive Bayes', 'Bernoullis Naive Bayes']
            option1 = st.selectbox('Select Model:', activities1)
            
            if option1 == 'None':
                st.markdown('Chose a ML Model')
            
            if option1 == 'Support Vector Machine':
                model_name = "SVM.pkl"
                Sentiment_Prediction(data,model_name)
                
            if option1 == 'Logistic Regression':
                model_name = "LR.pkl"
                Sentiment_Prediction(data,model_name)
                
            if option1 == 'Complement Naive Bayes':
                model_name = "CNB.pkl"
                Sentiment_Prediction(data,model_name)
                
            if option1 == 'Multinomial Naive Bayes':
                model_name = "MNB.pkl"
                Sentiment_Prediction(data,model_name)
                
            if option1 == 'Gaussian Naive Bayes':
                model_name = "GNB.pkl"
                Sentiment_Prediction(data,model_name)
                
            if option1 == 'Bernoullis Naive Bayes':
                model_name = "BNB.pkl"
                Sentiment_Prediction(data,model_name)
                
                
            activities2 = ['None','Positive', 'Negative', 'Neutral']
            option2 = st.selectbox('Select Polarity of Topic:', activities2)
            
            if option2 == 'None':
                st.markdown('Select Polarity of Topics')
                
            if (int(option2 != 'None') & int(option1 != 'None')):
                df = data[data['Sentiment']==option2]
                reviews = [d.split() for d in df['Customer_Reviews']]
                # Create Dictionary
                id2word = corpora.Dictionary(reviews)
                # Create Corpus: Term Document Frequency
                corpus = [id2word.doc2bow(text) for text in reviews]
                #gensim.models.ldamodel.LdaModel
                
                # Build LDA model
                lda_model = LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=4, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=10,
                                           passes=10,
                                           alpha='symmetric',
                                           iterations=100,
                                           per_word_topics=True)
                
            activities3 = ['None','WordCloud of Topic', 'Topic Importance']
            option3 = st.selectbox('Select Topic Extraction Method:', activities3)
                
            if(option3 == 'None'):
                st.markdown('Select Method for LDA Extraction')
                
            if(int(option3 == 'WordCloud of Topic') & int(option2 != 'NOne') & int(option1 != 'None')):
                cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'
                from wordcloud import WordCloud, STOPWORDS
                cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal= 1.0)

                topics = lda_model.show_topics(formatted=False)
                
                fig, axes = plt.subplots(2, 2, figsize=(15,15), sharex=True, sharey=True)

                for i, ax in enumerate(axes.flatten()):
                    fig.add_subplot(ax)
                    topic_words = dict(topics[i][1])
                    cloud.generate_from_frequencies(topic_words, max_font_size=300)
                    plt.gca().imshow(cloud)
                    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=20))
                    plt.gca().axis('off')


                    plt.subplots_adjust(wspace=0, hspace=0)
                    plt.axis('off')
                    plt.margins(x=0, y=0)
                    plt.tight_layout()
                #plt.show()
        
                st.pyplot(fig)
                
                
            if(int(option3 == 'Topic Importance') & int(option2 != 'NOne') & int(option1 != 'None')):
                    topics = lda_model.show_topics(formatted=False)
                    data_flat = [w for w_list in reviews  for w in w_list]
                    from collections import Counter
                    counter = Counter(data_flat)

                    out = []
                    for i, topic in topics:
                        for word, weight in topic:
                            out.append([word, i , weight, counter[word]])

                    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

                    # Plot Word Count and Weights of Topic Keywords
                    fig, axes = plt.subplots(4, 1, figsize=(16,25), sharey=True, dpi=160)
                    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
                    for i, ax in enumerate(axes.flatten()):
                        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
                        ax_twin = ax.twinx()
                        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
                        ax.set_ylabel('Word Count', color=cols[i])
                        ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 3500)
                        ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=20)
                        ax.tick_params(axis='y', left=False)
                        ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], Fontsize = 20, rotation=30, horizontalalignment= 'right')
                        ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')
                        #plt.xticks(fontsize= 15)

                        fig.tight_layout(w_pad=2)    
                        #fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
                        # plt.show()
                    st.pyplot(fig)
                    
                    
            if(int(option3 != 'None') & int(option2 == 'NOne') & int(option1 != 'None')): 
                st.markdown('Select Polarity First!!')
                
            if(int(option3 != 'None') & int(option2 != 'NOne') & int(option1 == 'None')): 
                st.markdown('Select Model First!!')
                
                
            
                
           
                    
                
           #Descriptive Analysis 
    elif option == 'Descriptive Analysis':
		
        st.subheader('Descriptive Analysis of the Reviews Dataset')
        data = st.file_uploader('Upoad Dataset: ',type = ['csv'])
        
        if data is not None:
            
            st.success('Data Successfully Loaded')
            data = pd.read_csv(data)
            st.dataframe(data.head(5))
            
            
            activities1 = ['None','Location of Reviews']
            option1 = st.selectbox('Select Distribution:', activities1)
            
            if option1 == 'None':
                st.markdown("Select Distribution!!!")
                
            elif option1 == 'Location of Reviews':
                st.subheader("Total number of Reviews from each Country")
                each_country = st.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='1')
                country_sentiment_count = data.groupby('Location')['Customer_Reviews'].count().sort_values(ascending=False)
                country_sentiment_count = pd.DataFrame({'Country':country_sentiment_count.index, 'Reviews':country_sentiment_count.values.flatten()})
                if not st.checkbox("Close", True, key='1'):
                    if each_country == 'Bar plot':
                        st.subheader("Total number of Reviews from each Country")
                        fig_1 = px.bar(country_sentiment_count, x='Country', y='Reviews', color='Reviews', height=500)
                        st.plotly_chart(fig_1)
                    if each_country == 'Pie chart':
                        st.subheader("Total number of Reviews from each Country")
                        fig_2 = px.pie(country_sentiment_count, values='Reviews', names='Country')
                        st.plotly_chart(fig_2)


   #oUR TRAINED MODELS VS TEXTBLOB COMPARISON
    elif option == 'Our Models vs Textblob':
        st.subheader('Trained ML Model and Textblob Comparison')
        data = st.file_uploader('Upoad Dataset: ',type = ['csv'])
        
        if data is not None:
            st.success('Data Successfully Loaded')
            data = pd.read_csv(data)
            st.dataframe(data.head(5))
            
            activities1 = ['None','Support Vector Machine', 'Logistic Regression', 'Complement Naive Bayes', 'Multinomial Naive Bayes', 'Gaussian Naive Bayes', 'Bernoullis Naive Bayes']
            option1 = st.selectbox('Select Model:', activities1)
            
            if option1 == 'None':
                st.markdown('Chose a ML Model')
            
            if option1 == 'Support Vector Machine':
                model_name = "SVM.pkl"
                Sentiment_Prediction(data,model_name)
                data['Polarity'] = data['Customer_Reviews'].apply(polarity)
                data['Polarity'] = data['Polarity'].apply(getAnalysis)
                accuracy = metrics.accuracy_score(data['Sentiment'],data['Polarity'])
                st.markdown('SVM accuracy = ' +str('{:4.2f}'.format(accuracy*100))+'%')
                
            if option1 == 'Logistic Regression':
                model_name = "LR.pkl"
                Sentiment_Prediction(data,model_name)
                data['Polarity'] = data['Customer_Reviews'].apply(polarity)
                data['Polarity'] = data['Polarity'].apply(getAnalysis)
                accuracy = metrics.accuracy_score(data['Sentiment'],data['Polarity'])
                st.markdown('LR accuracy = ' +str('{:4.2f}'.format(accuracy*100))+'%')
                
            if option1 == 'Complement Naive Bayes':
                model_name = "CNB.pkl"
                Sentiment_Prediction(data,model_name)
                data['Polarity'] = data['Customer_Reviews'].apply(polarity)
                data['Polarity'] = data['Polarity'].apply(getAnalysis)
                accuracy = metrics.accuracy_score(data['Sentiment'],data['Polarity'])
                st.markdown('CNB accuracy = ' +str('{:4.2f}'.format(accuracy*100))+'%')
                
            if option1 == 'Multinomial Naive Bayes':
                model_name = "MNB.pkl"
                Sentiment_Prediction(data,model_name)
                data['Polarity'] = data['Customer_Reviews'].apply(polarity)
                data['Polarity'] = data['Polarity'].apply(getAnalysis)
                accuracy = metrics.accuracy_score(data['Sentiment'],data['Polarity'])
                st.markdown('MNB accuracy = ' +str('{:4.2f}'.format(accuracy*100))+'%')
                
            if option1 == 'Gaussian Naive Bayes':
                model_name = "GNB.pkl"
                Sentiment_Prediction(data,model_name)
                data['Polarity'] = data['Customer_Reviews'].apply(polarity)
                data['Polarity'] = data['Polarity'].apply(getAnalysis)
                accuracy = metrics.accuracy_score(data['Sentiment'],data['Polarity'])
                st.markdown('GNB accuracy = ' +str('{:4.2f}'.format(accuracy*100))+'%')
                
            if option1 == 'Bernoullis Naive Bayes':
                model_name = "BNB.pkl"
                Sentiment_Prediction(data,model_name)
                data['Polarity'] = data['Customer_Reviews'].apply(polarity)
                data['Polarity'] = data['Polarity'].apply(getAnalysis)
                accuracy = metrics.accuracy_score(data['Sentiment'],data['Polarity'])
                st.markdown('BNB accuracy = ' +str('{:4.2f}'.format(accuracy*100))+'%')
                
                
                
            
    
        
                
    elif option == 'Text Prediction':
        st.subheader('Enter Text to Predict Sentiment')
        raw_docx = st.text_area('Your Text','')
        if st.button("Predict"):
            if raw_docx != '':
                
                Polarity = TextBlob(raw_docx).sentiment.polarity
                if Polarity < -0.13:
                    st.markdown( 'This is Negative Review' )
                elif Polarity > 0.13:
                    st.markdown( 'This is Positive Review' )
                else:
                    st.markdown( 'This is Neutral Review' )
            
            else:
                st.markdown("Enter Review First!!")
       
    
    
        
    else:
        st.markdown('FYP 2021')
        st.balloons()



if __name__ == '__main__':
	main()