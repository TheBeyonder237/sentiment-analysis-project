from huggingface_hub import HfApi, HfFolder
import argparse
import os

class ModelDeployer:
    def __init__(self, model_dir, repo_name, api_token):
        """
        Initialise le déploiement avec le répertoire du modèle, le nom du dépôt et le token API.

        Args:
            model_dir (str): Répertoire contenant le modèle entraîné.
            repo_name (str): Nom du dépôt sur Hugging Face (ex: username/model-name).
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
            # Créer le dépôt si non existant
            print(f"Vérification ou création du dépôt : {self.repo_name}")
            api.create_repo(repo_id=self.repo_name, token=self.api_token, exist_ok=True)

            # Déploiement du modèle
            print("Déploiement en cours...")
            api.upload_folder(
                repo_id=self.repo_name,
                folder_path=self.model_dir,
                token=self.api_token
            )
            print(f"✅ Modèle déployé avec succès sur Hugging Face : https://huggingface.co/{self.repo_name}")
        except Exception as e:
            print(f"❌ Échec du déploiement : {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Déploiement du modèle sur Hugging Face.")
    parser.add_argument("--model_dir", type=str, required=True, help="Répertoire contenant le modèle.")
    parser.add_argument("--repo_name", type=str, required=True, help="Nom du dépôt Hugging Face (ex: DavidNgoue/sentiment-analysis-project).")
    parser.add_argument("--api_token", type=str, required=True, help="Token API Hugging Face.")
    args = parser.parse_args()

    deployer = ModelDeployer(model_dir=args.model_dir, repo_name=args.repo_name, api_token=args.api_token)
    deployer.deploy()
