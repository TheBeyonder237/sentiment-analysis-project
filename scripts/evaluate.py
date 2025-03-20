import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import os

class ModelEvaluator:
    def __init__(self, model_path, data_path):
        self.model_path = model_path
        self.data_path = data_path
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)

    def load_data(self):
        test_df = pd.read_csv(os.path.join(self.data_path, 'test_data.csv'))
        test_texts = test_df['text'].tolist()
        test_labels = test_df['label'].tolist()
        return test_texts, test_labels

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

    def evaluate_model(self):
        test_texts, test_labels = self.load_data()
        test_encodings = self.tokenize_data(test_texts)
        test_dataset = self.create_dataset(test_encodings, test_labels)

        training_args = TrainingArguments(
            output_dir='./results',
            per_device_eval_batch_size=64,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            eval_dataset=test_dataset
        )

        predictions = trainer.predict(test_dataset)
        predicted_labels = predictions.predictions.argmax(-1)

        accuracy = accuracy_score(test_labels, predicted_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predicted_labels, average='binary')

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")
