"""
Ce script est chargé de déployer le modèle entraîné sur la plateforme Hugging Face.
Il utilise l'API Hugging Face pour pousser les fichiers du modèle dans un dépôt distant.

Fonctionnalités :
1. Chargement du modèle depuis le répertoire local.
2. Connexion à l'API Hugging Face via un token d'authentification.
3. Envoi du modèle sur le dépôt distant.
4. Vérification du déploiement réussi.

Exemple d'utilisation :
    python deploy.py --model_dir models/distilbert --repo_name username/sentiment-analysis --api_token HF_API_KEY
"""

from huggingface_hub import HfApi, HfFolder
import argparse
import os

class ModelDeployer:
    def __init__(self, model_dir, repo_name, api_token):
        """
        Initialise le déploiement avec le répertoire du modèle, le nom du dépôt et le token API.

        Args:
            model_dir (str): Répertoire contenant le modèle entraîné.
            repo_name (str): Nom du dépôt sur Hugging Face.
            api_token (str): Token d'authentification API Hugging Face.
        """
        self.model_dir = model_dir
        self.repo_name = repo_name
        self.api_token = api_token

    def deploy(self):
        """
        Déploie le modèle sur Hugging Face en utilisant l'API.
        """
        api = HfApi()
        try:
            api.upload_folder(
                repo_id=self.repo_name,
                folder_path=self.model_dir,
                token=self.api_token
            )
            print(f"Modèle déployé avec succès sur Hugging Face dans le dépôt '{self.repo_name}'.")
        except Exception as e:
            print(f"Échec du déploiement : {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Déploiement du modèle sur Hugging Face.")
    parser.add_argument("--model_dir", type=str, required=True, help="Répertoire contenant le modèle.")
    parser.add_argument("--repo_name", type=str, required=True, help="Nom du dépôt Hugging Face.")
    parser.add_argument("--api_token", type=str, required=True, help="Token API Hugging Face.")
    args = parser.parse_args()

    deployer = ModelDeployer(model_dir=args.model_dir, repo_name=args.repo_name, api_token=args.api_token)
    deployer.deploy()
