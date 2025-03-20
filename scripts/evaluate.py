import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
import argparse
import torch
import os

class ModelEvaluator:
    def __init__(self, model_dir, test_path):
        """
        Initialise l'évaluateur avec le répertoire du modèle et le chemin des données de test.

        Args:
            model_dir (str): Répertoire contenant le modèle entraîné.
            test_path (str): Chemin vers le fichier CSV des données de test.
        """
        self.model_dir = model_dir
        self.test_path = test_path
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)

    def load_dataset(self):
        """
        Charge l'ensemble de test à partir du fichier CSV.

        Returns:
            Dataset: Ensemble de test formaté pour le modèle.
        """
        df = pd.read_csv(self.test_path)

        # Vérifier et remplacer les valeurs manquantes dans la colonne "text"
        df['text'] = df['text'].fillna("").astype(str)

        print("Colonnes détectées :", df.columns)
        print("Types de données des colonnes :\n", df.dtypes)
        print("Aperçu des premières lignes :\n", df.head())

        dataset = Dataset.from_pandas(df)

        def tokenize_function(examples):
            texts = [str(text) for text in examples['text']]
            return self.tokenizer(texts, padding='max_length', truncation=True, max_length=512)

        dataset = dataset.map(tokenize_function, batched=True)
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        print("Données de test chargées avec succès.")
        return dataset

    def evaluate(self, test_dataset):
        """
        Évalue le modèle sur l'ensemble de test.

        Args:
            test_dataset (Dataset): Ensemble de test prêt pour l'évaluation.
        """
        predictions = []
        labels = []
        self.model.eval()
        with torch.no_grad():
            for batch in test_dataset:
                inputs = {key: batch[key].unsqueeze(0) for key in ['input_ids', 'attention_mask']}
                output = self.model(**inputs)
                prediction = torch.argmax(output.logits, dim=1).item()
                predictions.append(prediction)
                labels.append(batch['label'])
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-score: {f1:.4f}")
        with open(os.path.join(self.model_dir, "evaluation.txt"), "w") as f:
            f.write(f"Accuracy: {accuracy:.4f}\nF1-score: {f1:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation du modèle de sentiment analysis.")
    parser.add_argument("--test_path", type=str, required=True, help="Chemin vers le fichier de test.")
    parser.add_argument("--model_dir", type=str, required=True, help="Répertoire du modèle entraîné.")
    args = parser.parse_args()

    evaluator = ModelEvaluator(model_dir=args.model_dir, test_path=args.test_path)
    test_dataset = evaluator.load_dataset()
    evaluator.evaluate(test_dataset)
