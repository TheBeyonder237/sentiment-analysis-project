import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import argparse
import os
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

class Preprocessing:
    def __init__(self, data_path, output_dir="datas/processed"):
        """
        Initialise le prétraitement avec le chemin des données et le répertoire de sortie.

        Args:
            data_path (str): Chemin du fichier CSV contenant les données.
            output_dir (str): Répertoire où enregistrer les ensembles prétraités.
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.data = None
        os.makedirs(self.output_dir, exist_ok=True)

    def clean_text(self, text):
        """
        Nettoie le texte en supprimant les caractères spéciaux, les stop words et en appliquant la lemmatisation.

        Args:
            text (str): Texte à nettoyer.

        Returns:
            str: Texte nettoyé.
        """
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        
        # Convertir en minuscule
        text = text.lower()
        # Retirer les caractères spéciaux et les chiffres
        text = re.sub(f"[{string.punctuation}]", " ", text)
        text = re.sub(r"\d+", "", text)
        # Supprimer les mots vides
        words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
        return " ".join(words)

    def load_data(self):
        """
        Charge les données depuis le fichier CSV.
        
        Raises:
            FileNotFoundError: Si le fichier n'existe pas.
        """
        try:
            self.data = pd.read_csv(self.data_path)
            print("Données chargées avec succès.")
        except FileNotFoundError:
            print(f"Erreur : Le fichier {self.data_path} est introuvable.")
            raise

    def preprocess(self):
        """
        Effectue le prétraitement des données :
        1. Supprime les valeurs manquantes.
        2. Nettoie les textes.
        3. Encode les sentiments en entiers.
        4. Sépare les données en ensembles d'entraînement et de test.
        """
        self.data.dropna(inplace=True)
        self.data['text'] = self.data['text'].apply(self.clean_text)

        encoder = LabelEncoder()
        self.data['label'] = encoder.fit_transform(self.data['sentiment'])
        
        train, test = train_test_split(self.data, test_size=0.2, random_state=42, stratify=self.data['label'])
        train.to_csv(os.path.join(self.output_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.output_dir, "test.csv"), index=False)
        print("Prétraitement terminé et ensembles enregistrés.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prétraitement des données de sentiment analysis.")
    parser.add_argument("--data_path", type=str, required=True, help="Chemin vers le fichier CSV des données.")
    args = parser.parse_args()

    preprocessor = Preprocessing(data_path=args.data_path)
    preprocessor.load_data()
    preprocessor.preprocess()
