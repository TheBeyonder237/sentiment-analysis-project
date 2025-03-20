"""
Ce script permet d'envoyer un e-mail de notification suite à l'entraînement, l'évaluation ou le déploiement du modèle.
Il utilise le protocole SMTP pour envoyer un message à une ou plusieurs adresses e-mail.

Fonctionnalités :
1. Configuration de l'e-mail expéditeur et des destinataires.
2. Création du contenu de l'e-mail avec un objet et un message personnalisé.
3. Connexion sécurisée au serveur SMTP et envoi de l'e-mail.

Exemple d'utilisation :
    python send_email.py --smtp_user user@gmail.com --smtp_pass password --recipient notify@example.com \
        --subject "Déploiement réussi" --message "Le modèle a été déployé avec succès sur Hugging Face."
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import argparse

class EmailNotifier:
    def __init__(self, smtp_user, smtp_pass, recipient, subject, message):
        """
        Initialise l'envoi d'e-mail avec les informations nécessaires.

        Args:
            smtp_user (str): Adresse e-mail de l'expéditeur.
            smtp_pass (str): Mot de passe de l'expéditeur.
            recipient (str): Adresse du destinataire.
            subject (str): Objet de l'e-mail.
            message (str): Contenu du message.
        """
        self.smtp_user = smtp_user
        self.smtp_pass = smtp_pass
        self.recipient = recipient
        self.subject = subject
        self.message = message

    def send(self):
        """
        Envoie l'e-mail de notification en utilisant le serveur SMTP de Gmail.
        """
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_user
            msg['To'] = self.recipient
            msg['Subject'] = self.subject
            msg.attach(MIMEText(self.message, 'plain'))

            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)
                print(f"E-mail envoyé avec succès à {self.recipient}.")
        except Exception as e:
            print(f"Erreur lors de l'envoi de l'e-mail : {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Envoi d'un e-mail de notification.")
    parser.add_argument("--smtp_user", type=str, required=True, help="Adresse e-mail de l'expéditeur.")
    parser.add_argument("--smtp_pass", type=str, required=True, help="Mot de passe de l'expéditeur.")
    parser.add_argument("--recipient", type=str, required=True, help="Adresse e-mail du destinataire.")
    parser.add_argument("--subject", type=str, required=True, help="Objet de l'e-mail.")
    parser.add_argument("--message", type=str, required=True, help="Contenu du message.")
    args = parser.parse_args()

    notifier = EmailNotifier(smtp_user=args.smtp_user, smtp_pass=args.smtp_pass, recipient=args.recipient, subject=args.subject, message=args.message)
    notifier.send()
