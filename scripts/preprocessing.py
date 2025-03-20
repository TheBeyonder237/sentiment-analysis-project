import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import os

class DataPreprocessor:
    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

    def preprocess_text(self, text):
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(tokens)

    def load_and_preprocess_data(self):
        df = pd.read_csv(self.data_path)
        df = df[["text","sentiment"]]
        df.rename(columns={"sentiment":"label"}, inplace=True)
        df['text'] = df['text'].apply(self.preprocess_text)
        return df

    def split_and_save_data(self, test_size=0.2, random_state=42):
        df = self.load_and_preprocess_data()
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        train_df.to_csv(os.path.join(self.output_path, 'train_data.csv'), index=False)
        test_df.to_csv(os.path.join(self.output_path, 'test_data.csv'), index=False)
        print("Data preprocessed, split, and saved!")