# Script d'Entraînement du Modèle (train.py)

"""
Ce script est responsable de l'entraînement du modèle de classification de sentiment en utilisant DistilBERT.
Il prend en entrée les données prétraitées et entraîne le modèle avec des hyperparamètres définis.
Le modèle entraîné est ensuite sauvegardé dans un répertoire spécifié.

Fonctionnalités :
1. Chargement des ensembles d'entraînement.
2. Configuration du modèle DistilBERT.
3. Entraînement du modèle avec suivi des métriques.
4. Sauvegarde du modèle après l'entraînement.

Exemple d'utilisation :
    python train.py --train_path data/processed/train.csv --output_dir models/distilbert
"""

import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import argparse
import os

class ModelTrainer:
    def __init__(self, train_path, text_column="text", output_dir="models/distilbert"):
        """
        Initialise l'entraîneur de modèle avec les données et le répertoire de sortie.

        Args:
            train_path (str): Chemin vers le fichier CSV d'entraînement.
            text_column (str): Nom de la colonne contenant les textes.
            output_dir (str): Répertoire de sauvegarde du modèle.
        """
        self.train_path = train_path
        self.text_column = text_column
        self.output_dir = output_dir

        # Charger les données pour déterminer le nombre de labels
        df = pd.read_csv(self.train_path)
        num_labels = df['label'].nunique()
        print(f"Nombre de classes détectées : {num_labels}")

        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=num_labels  # Définir le nombre de classes
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def load_dataset(self):
        """
        Charge et prépare l'ensemble d'entraînement à partir du fichier CSV.

        Returns:
            Dataset: Ensemble d'entraînement formaté pour le modèle.
        """
        df = pd.read_csv(self.train_path)

        if "text" not in df.columns:
            raise ValueError("La colonne 'text' est introuvable dans le fichier d'entraînement.")
        
        print("Colonnes détectées :", df.columns)
        print("Types de données des colonnes :\n", df.dtypes)
        print("Aperçu des premières lignes :\n", df.head())

         # Diviser en ensembles d'entraînement et de validation
        train_df = df.sample(frac=0.8, random_state=42)  
        val_df = df.drop(train_df.index) 


        def tokenize_function(examples):
            texts = [str(text) for text in examples['text']]
            return self.tokenizer(texts, padding='max_length', truncation=True, max_length=512)

        train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
        val_dataset = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)

        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        print("Données d'entraînement et de validation chargées et préparées.")
        return train_dataset, val_dataset

    def train(self, train_dataset):
        """
        Entraîne le modèle DistilBERT en utilisant les données d'entraînement fournies.

        Args:
            train_dataset (Dataset): Ensemble d'entraînement prêt pour le modèle.
        """
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="epoch",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=10,
            logging_dir="./logs",
            save_steps=10,
             eval_steps=10,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print("Modèle entraîné et sauvegardé avec succès.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement du modèle de sentiment analysis avec DistilBERT.")
    parser.add_argument("--train_path", type=str, required=True, help="Chemin vers le fichier d'entraînement.")
    parser.add_argument("--output_dir", type=str, default="models/distilbert", help="Répertoire de sortie du modèle.")
    args = parser.parse_args()

    trainer = ModelTrainer(train_path=args.train_path, output_dir=args.output_dir)
    train_dataset, val_dataset = trainer.load_dataset()
    trainer.train(train_dataset)
