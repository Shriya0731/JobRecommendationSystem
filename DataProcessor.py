import spacy
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import csv


# Download NLTK resources
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
        # Join tokens back into a single string
        tokens = [token for token in tokens if token not in stop_words]
        # Join tokens back into a single string
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text

    def get_sentence_embeddings(self,sentence):
        doc = self.nlp(sentence)
        return doc.vector

    def store_embeddings(self,column_to_embed, new_col_name):
        old_col = self.dataset[column_to_embed]
        sentence_embeddings = [self.get_sentence_embeddings(column_to_embed) for column_to_embed in old_col]
        # Create a new column in the dataset to store the embeddings
        self.dataset[new_col_name] = sentence_embeddings
        # Save the modified dataset to a new CSV file
        self.dataset.to_csv('dataset_with_embeddings.csv', index=False)

    def get_processed_dataset(self):
        # Apply preprocessing to the 'description' column
        self.dataset['jd'] = self.dataset['jd'].apply(self.preprocess_text)
        print(self.dataset[['jd']].head())
        self.store_embeddings('jd', 'embeddings')
        print(self.dataset.head())
        return self.dataset
