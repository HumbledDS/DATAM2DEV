"""
train_model.py - Entraînement du modèle de détection de spam

Script d'entraînement d'un classificateur spam/ham utilisant:
- TF-IDF pour la vectorisation du texte
- Logistic Regression pour la classification

Le modèle et le vectorizer sont sauvegardés sous forme de fichiers pickle
"""

import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def create_training_data():
    """
    Crée un dataset d'entraînement d'exemples d'emails spam et ham.

    Returns:
        tuple: (emails, labels) où labels = [1 pour spam, 0 pour ham]
    """

    # Emails SPAM
    spam_emails = [
        "URGENT! Click here to claim your FREE money now!!! Limited time offer expires in 1 hour. Do not share!",
        "Congratulations you have won a FREE IPHONE! Claim your prize at www.fakesite.ru/claim now!!!",
        "Dear valued customer, verify your account IMMEDIATELY or it will be closed. Click link below.",
        "ATTENTION: Your bank account has suspicious activity! Verify password here: fake-bank.com/verify",
        "Work from home and earn $5000 per week! No experience needed. Send money to get started!",
        "Buy cheap medications ONLINE with NO PRESCRIPTION! Viagra, cialis, more. Act now!",
        "Nigerian Prince offers you $10 MILLION dollars! Reply with banking details to claim inheritance.",
        "Sexy singles in your area want to chat with YOU! Click here for HOT encounters RIGHT NOW!",
        "Get FREE CASINO CHIPS worth $1000! No deposit required. Jackpot waiting for you!",
        "This email may be in VIOLATION of federal law! Delete immediately or face legal action!",
        "Your package delivery failed! Click to reschedule: tracking.scam-site.com/update?id=xyz",
        "AMAZING OFFER! Buy 1 get 3 FREE! Use code SPAM99 for instant savings! HURRY LIMITED STOCK!",
        "You have been SELECTED for a special tax refund! Claim $3,500 instantly at irs-fake.net",
        "WARNING: Your computer has VIRUS! Download antivirus NOW or your data will be DELETED!",
        "Earn Bitcoin FAST! Mining software makes money while you sleep. Download now FREE!",
        "SHOCKING: Celebrity uses this one weird trick doctors HATE! Click to see what it is!",
        "Lose 30 pounds in 30 days GUARANTEED! Revolutionary diet pill works or money back!",
        "Cheap luxury watches and bags! Rolex, Gucci, Prada AUTHENTIC prices! Shop now!",
        "ALERT: PayPal account compromised! Verify identity IMMEDIATELY to avoid suspension!",
        "Congratulations! Your email has been randomly selected to receive $50 Amazon gift card!",
    ]

    # Emails HAM (Légitime)
    ham_emails = [
        "Hi John, I wanted to follow up on our meeting last Thursday. Could you please send me the updated timeline?",
        "Dear Sarah, Thank you for your email. I will review the proposal and get back to you by Friday.",
        "Good morning team, Please find attached the Q2 financial report. Let me know if you have any questions.",
        "Hi Alex, I hope this email finds you well. I wanted to schedule a meeting to discuss the new project.",
        "Dear Client, Thank you for choosing our services. Your invoice for Invoice #2024-001 is attached.",
        "Hi everyone, The team meeting has been rescheduled to Tuesday at 2 PM. See you there!",
        "Good afternoon, I wanted to confirm our appointment tomorrow at 10 AM. Please let me know if you need to reschedule.",
        "Dear Professor, I am writing to inquire about the assignment submission deadline for next week.",
        "Hi team, Great work on completing the sprint! I have reviewed all pull requests and they look good.",
        "Hello, I wanted to thank you personally for attending my presentation yesterday. Your feedback was invaluable.",
        "Dear Customer, Your order #ORD-2024-1234 has been shipped. Tracking number: 1Z999AA10123456784",
        "Hi Michael, I hope you are doing well. I wanted to catch up and see how the new role is treating you.",
        "Good morning, The office will be closed on Monday for the holiday. Please plan your work accordingly.",
        "Dear colleague, I wanted to share some interesting research articles on machine learning for you to review.",
        "Hi Emma, Thank you for your help with the project. Your contributions made a real difference.",
        "Dear applicant, Thank you for submitting your resume. We will be in touch if you move forward in the process.",
        "Hi David, I wanted to introduce you to James who will be joining our team next month.",
        "Good afternoon, The lunch meeting tomorrow will be in the conference room instead of the cafeteria.",
        "Dear Dr. Smith, I would like to request a meeting to discuss my research findings with you.",
        "Hi Sarah, The slides for tomorrow presentation are ready. I am attaching them to this email.",
    ]

    # Combiner les données
    emails = spam_emails + ham_emails
    labels = [1] * len(spam_emails) + [0] * len(ham_emails)

    return emails, labels


def train_spam_classifier():
    """
    Entraîne un classificateur de spam et sauvegarde le modèle.

    Étapes:
    1. Crée les données d'entraînement
    2. Vectorise le texte avec TF-IDF
    3. Divise en train/test
    4. Entraîne Logistic Regression
    5. Évalue les performances
    6. Sauvegarde le vectorizer et le modèle
    """

    print("=" * 70)
    print("ENTRAÎNEMENT DU MODÈLE DE DÉTECTION DE SPAM")
    print("=" * 70)

    # Étape 1: Créer les données
    print("\n[1/5] Création du dataset d'entraînement...")
    emails, labels = create_training_data()
    print(f"    ✓ {len(emails)} emails créés ({sum(labels)} spam, {len(labels)-sum(labels)} ham)")

    # Étape 2: Vectoriser avec TF-IDF
    print("\n[2/5] Vectorisation du texte avec TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=1000,           # Maximum de features
        min_df=1,                    # Ignorer les tokens qui apparaissent dans moins de 1 doc
        max_df=0.9,                  # Ignorer les tokens qui apparaissent dans plus de 90% des docs
        ngram_range=(1, 2),          # Utiliser unigrammes et bigrammes
        lowercase=True,              # Convertir en minuscules
        stop_words='english'         # Supprimer les stop words anglais
    )
    X = vectorizer.fit_transform(emails)
    print(f"    ✓ {X.shape[0]} documents vectorisés")
    print(f"    ✓ {X.shape[1]} features créées")
    print(f"    ✓ Densité de la matrice: {X.nnz / (X.shape[0] * X.shape[1]) * 100:.2f}%")

    # Étape 3: Diviser en train/test
    print("\n[3/5] Division train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"    ✓ Train: {X_train.shape[0]} exemples")
    print(f"    ✓ Test: {X_test.shape[0]} exemples")

    # Étape 4: Entraîner le modèle
    print("\n[4/5] Entraînement du modèle Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs',
        verbose=0
    )
    model.fit(X_train, y_train)
    print("    ✓ Modèle entraîné avec succès")

    # Étape 5: Évaluer les performances
    print("\n[5/5] Évaluation des performances...")

    # Prédictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Probabilités
    y_train_proba = model.predict_proba(X_train)
    y_test_proba = model.predict_proba(X_test)

    # Métriques d'entraînement
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    train_recall = recall_score(y_train, y_train_pred, zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)

    # Métriques de test
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

    print("\n    📊 RÉSULTATS D'ENTRAÎNEMENT:")
    print(f"    Train Accuracy:  {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"    Train Precision: {train_precision:.4f}")
    print(f"    Train Recall:    {train_recall:.4f}")
    print(f"    Train F1-Score:  {train_f1:.4f}")

    print("\n    📊 RÉSULTATS DE TEST:")
    print(f"    Test Accuracy:   {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"    Test Precision:  {test_precision:.4f}")
    print(f"    Test Recall:     {test_recall:.4f}")
    print(f"    Test F1-Score:   {test_f1:.4f}")

    # Matrice de confusion
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    print(f"\n    📋 MATRICE DE CONFUSION (Test):")
    print(f"    True Negatives:  {tn}  (emails ham correctement identifiés)")
    print(f"    False Positives: {fp}  (emails ham classés comme spam)")
    print(f"    False Negatives: {fn}  (emails spam non détectés)")
    print(f"    True Positives:  {tp}  (emails spam correctement identifiés)")

    # Créer le répertoire models s'il n'existe pas
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Sauvegarder le vectorizer
    vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"\n    ✓ Vectorizer sauvegardé: {vectorizer_path}")

    # Sauvegarder le modèle
    model_path = os.path.join(models_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"    ✓ Modèle sauvegardé: {model_path}")

    print("\n" + "=" * 70)
    print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print("=" * 70)

    return vectorizer, model


if __name__ == "__main__":
    train_spam_classifier()
