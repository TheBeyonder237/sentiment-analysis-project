# **Projet d'Analyse de Sentiment avec CI/CD Automatisé**

Ce projet utilise **DistilBERT** pour l'analyse de sentiment sur des données textuelles.  
Il intègre un pipeline **CI/CD** automatisé avec **GitHub Actions**, permettant de :  
- Entraîner et évaluer un modèle de sentiment.  
- Déployer le modèle sur **Hugging Face** si les performances sont satisfaisantes.  
- Générer et envoyer la documentation complète par e-mail aux contributeurs.  

---

##  **Objectifs du Projet**
1. Effectuer une analyse de sentiment sur des données textuelles en utilisant **DistilBERT**.  
2. Automatiser le pipeline CI/CD pour garantir la robustesse du modèle.  
3. Déployer le modèle sur **Hugging Face** si le score est satisfaisant.  
4. Générer la documentation automatiquement et l'envoyer par e-mail.  

---

##  **Technologies Utilisées**
- **Python 3.9**  
- **Transformers (Hugging Face)**  
- **PyTorch**  
- **Scikit-learn**  
- **GitHub Actions**  
- **Git LFS**  
- **SMTP (envoi d'e-mail)**  
- **Bash (génération de documentation)**  

---

##  **Structure du Projet**
```
sentiment-analysis-project/
├── data/
│   └── processed/
├── models/
│   └── distilbert/
├── scripts/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── deploy.py
│   └── send_email.py
├── docs/
│   └── documentation.html
├── .gitignore
├── .gitattributes
├── requirements.txt
├── ci-cd.yml
└── README.md
```

---

##  **Configuration des Secrets GitHub Actions**

Les secrets suivants doivent être configurés dans les **GitHub Secrets** :  
| Secret Name       | Description                              |
|------------------|------------------------------------------|
| `HF_API_KEY`      | Clé API pour accéder à Hugging Face.       |
| `THRESHOLD_SCORE` | Score minimum requis pour le déploiement. |
| `REPO_NAME`       | Nom du dépôt sur Hugging Face.             |
| `SMTP_USER`       | Adresse e-mail pour l'envoi de notifications. |
| `SMTP_PASS`       | Mot de passe ou token d'application pour l'e-mail. |

---

## 🔥 **Installation**
1. **Cloner le projet**  
   ```bash
   git clone https://github.com/TheBeyonder237/sentiment-analysis-project.git
   cd sentiment-analysis-project
   ```
2. **Installer les dépendances**  
   ```bash
   pip install -r requirements.txt
   ```
3. **Installer Git LFS**  
   ```bash
   git lfs install
   git lfs track "*.bin" "*.pt" "*.h5" "*.model" "*.csv"
   ```
4. **Configurer les variables d'environnement (si nécessaire)**  

---

## 📝 **Utilisation**
1. **Prétraitement des données**  
   ```bash
   python scripts/preprocessing.py --data_path data/sentiment_analysis.csv
   ```
2. **Entraînement du modèle**  
   ```bash
   python scripts/train.py --train_path data/processed/train.csv
   ```
3. **Évaluation du modèle**  
   ```bash
   python scripts/evaluate.py --test_path data/processed/test.csv --model_dir models/distilbert
   ```
4. **Déploiement sur Hugging Face**  
   ```bash
   python scripts/deploy.py --model_dir models/distilbert --repo_name username/sentiment-analysis --api_token hf_abcdef123456
   ```
5. **Envoi de la documentation par e-mail**  
   ```bash
   python scripts/send_email.py --smtp_user "user@gmail.com" --smtp_pass "password" --recipient "danielledjofang2003@gmail.com,nchourupouom04@gmail.com,ngouedavidrogeryannick@gmail.com" --subject "CI/CD Status" --message "Pipeline terminé." --attachment docs.zip
   ```
---

## ⚙️ **Pipeline CI/CD Automatisé**

### 💼 **Fonctionnalités du CI/CD**
- Vérification des dépendances  
- Prétraitement des données  
- Entraînement du modèle  
- Évaluation des performances  
- Vérification du score de précision  
- Déploiement sur Hugging Face si le score est satisfaisant  
- Génération de la documentation  
- Envoi d'un e-mail de notification avec la documentation en pièce jointe  

### 📝 **Lancement du CI/CD**  
Le pipeline CI/CD se déclenche automatiquement lors de chaque **push** ou mise à jour sur les branches :  
- **main**  
- **dev**  

---

## ✉️ **Contributeurs**

| Nom                          | E-mail                                    |
|------------------------------|--------------------------------------------|
| **Danielle Djofang**            | danielledjofang2003@gmail.com              |
| **Nchourupouo Mohamed**             | nchourupouom04@gmail.com                   |
| **Ngoue David Roger Yannick** | ngouedavidrogeryannick@gmail.com            |

---

## 📧 **Contact**
Pour toute question ou contribution, veuillez contacter un des contributeurs par e-mail.  

---

## 📜 **Licence**
Ce projet est sous licence MIT. Voir le fichier [LICENSE](./LICENSE) pour plus de détails.  

---

N'hésite pas à me dire si tu veux ajouter d'autres sections ou des détails supplémentaires ! 💪