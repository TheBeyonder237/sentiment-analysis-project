import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
import torch
import os
from evaluate import ModelEvaluator
from preprocessing import DataPreprocessor

class ModelTrainer:
    def __init__(self, model_path, data_path):
        self.model_path = model_path
        self.data_path = data_path
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        self.label_encoder = LabelEncoder()

    def load_data(self):
        train_df = pd.read_csv(os.path.join(self.data_path, 'train_data.csv'))
        train_df = train_df.dropna(subset=['text'])
        train_texts = train_df['text'].tolist()
        train_labels = train_df['label'].tolist()
        train_labels = self.label_encoder.fit_transform(train_labels)
        return train_texts, train_labels

    def tokenize_data(self, texts):
        return self.tokenizer(texts, truncation=True, padding=True)

    def create_dataset(self, encodings, labels):
        class SentimentDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        return SentimentDataset(encodings, labels)

    def train_model(self):
        train_texts, train_labels = self.load_data()
        train_encodings = self.tokenize_data(train_texts)
        train_dataset = self.create_dataset(train_encodings, train_labels)

        print(train_encodings)
        print(train_labels)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
        )
       
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )

        trainer.train()
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        print('Modèle entraîné et sauvegardé !')

def main():
    # Prétraitement
    preprocessor = DataPreprocessor('../data/sentiment_analysis.csv', '../data/processed')
    preprocessor.split_and_save_data()

    # Entraînement
    trainer = ModelTrainer('../models', '../data/processed')
    trainer.train_model()

    # Évaluation
    evaluator = ModelEvaluator('models', '../data/processed')
    evaluator.evaluate_model()

if __name__ == "__main__":
    main()