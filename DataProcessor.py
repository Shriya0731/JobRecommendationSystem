import spacy
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import csv


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

class JobDataset:
    def __init__(self, filename, sample_size,required_cols):
        self.filename = filename
        self.sample_size = sample_size
        self.dataset = None
        self.required_cols = required_cols
        self.nlp = spacy.load("en_core_web_md")
        self.load_data()
        #self.train_word2vec_model()

    def load_data(self):
        # Open the CSV file and read its contents using the csv module
        with open(self.filename, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            self.dataset = list(reader)

        # Convert the list of rows into a DataFrame
        self.dataset = pd.DataFrame(self.dataset[1:], columns=self.dataset[0])
        columns_to_check = self.required_cols
        self.dataset.dropna(subset=columns_to_check, how='any', inplace=True)
        print(self.dataset.shape[0])
        self.dataset = self.dataset.sample(n=200, random_state=42)
        self.dataset['jd'] = self.dataset[columns_to_check].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

    # Function to preprocess text data
    def preprocess_text(self,text):
        # Remove special characters, punctuation, and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convert text to lowercase
        text = text.lower()
        # Tokenize text
        tokens = word_tokenize(text)
        # Remove stopwords
        additional_stopwords = set(['job', 'title', 'company', 'join', 'need', 'looking','work','utccategory'])
        # Combine standard English stop words with additional stop words
        stop_words = set(stopwords.words('english')) | additional_stopwords
        tokens = [token for token in tokens if token not in stop_words]
        # Join tokens back into a single string
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text

    # def get_preprocess_tokens(self,text):
    #     # Remove special characters, punctuation, and digits
    #     text = re.sub(r'[^a-zA-Z\s]', '', text)
    #     # Convert text to lowercase
    #     text = text.lower()
    #     # Tokenize text
    #     tokens = word_tokenize(text)
    #     # Remove stopwords
    #     additional_stopwords = set(['job', 'title', 'company', 'join', 'need', 'looking','work','utccategory'])
    #     # Combine standard English stop words with additional stop words
    #     stop_words = set(stopwords.words('english')) | additional_stopwords
    #     tokens = [token for token in tokens if token not in stop_words]
    #     # Join tokens back into a single string
    #     return tokens
    #
    # def get_sentence_embeddings_2(self, sentence):
    #     tokens = self.preprocess_text(sentence)
    #     # Only consider tokens that are present in the Word2Vec model's vocabulary
    #     tokens_in_vocab = [token for token in tokens if token in self.w2v_model.wv]
    #     if tokens_in_vocab:
    #         # Calculate the average vector for the tokens present in the vocabulary
    #         embeddings = [self.w2v_model.wv[token] for token in tokens_in_vocab]
    #         avg_embedding = sum(embeddings) / len(embeddings)
    #         return avg_embedding
    #     else:
    #         return None
    # # If no tokens
    #
    # def train_word2vec_model(self):
    #     # Tokenize job descriptions
    #     tokenized_descriptions = self.dataset['jd'].apply(self.get_preprocess_tokens)
    #
    #     # Train Word2Vec model
    #     self.w2v_model = Word2Vec(sentences=tokenized_descriptions, vector_size=100, window=5, min_count=1, workers=4)
    #
    #     # Save the trained Word2Vec model
    #     self.w2v_model.save("word2vec_model.bin")

    def get_sentence_embeddings(self,sentence):
        doc = self.nlp(sentence)
        return doc.vector

    def store_embeddings(self,column_to_embed, new_col_name):
        old_col = self.dataset[column_to_embed]
        sentence_embeddings = [self.get_sentence_embeddings(column_to_embed) for column_to_embed in old_col]
        print(sentence_embeddings[0].shape)
        # Create a new column in the dataset to store the embeddings
        self.dataset[new_col_name] = sentence_embeddings
        # Save the modified dataset to a new CSV file
        self.dataset.to_csv('dataset_with_embeddings.csv', index=False)

    def get_processed_dataset(self):
        # Apply preprocessing to the 'description' column
        self.dataset['jd'] = self.dataset['jd'].apply(self.preprocess_text)
        print(self.dataset[['jd']].head())
        self.store_embeddings('jd', 'embeddings')
        # self.tfidf_vectorizer = TfidfVectorizer()
        # self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.dataset['jd'])
        print(self.dataset.head())
        return self.dataset
