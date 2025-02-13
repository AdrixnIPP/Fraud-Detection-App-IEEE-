import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve, classification_report

# Titre de l'application
st.title("📊 Détection de Fraude avec XGBoost")

# Upload des fichiers CSV
train_transaction_file = st.file_uploader("10gZEztQMNJ323_wTrjBl7duYeKvF06Wg", type=["csv"])
train_identity_file = st.file_uploader("1qVwgfh7795K42MLkSVFxggQd0ga4NUqG", type=["csv"])

# 🔗 URL pour récupérer les fichiers depuis Google Drive
url_transaction = f"https://drive.google.com/uc?export=download&id={"10gZEztQMNJ323_wTrjBl7duYeKvF06Wg"}"
url_identity = f"https://drive.google.com/uc?export=download&id={"1qVwgfh7795K42MLkSVFxggQd0ga4NUqG"}"

# 📥 Fonction pour charger les fichiers depuis Google Drive
@st.cache_data
def load_data(url):
    response = requests.get(url)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))

# Chargement des fichiers
train_transaction = load_data(url_transaction)
train_identity = load_data(url_identity)

# Fusion des fichiers
train_df = pd.merge(train_transaction, train_identity, how='left', on='TransactionID')

# Affichage des premières lignes
st.subheader("Aperçu des données fusionnées :")
st.dataframe(train_df.head())

if train_transaction_file and train_identity_file:
    train_transaction = pd.read_csv(train_transaction_file)
    train_identity = pd.read_csv(train_identity_file)

    # Fusion des fichiers
    train_df = pd.merge(train_transaction, train_identity, how='left', on='TransactionID')
    
    # Affichage des premières lignes
    st.subheader("Aperçu des données fusionnées :")
    st.dataframe(train_df.head())
    
    # Sélection du type d'analyse
    analysis_type = st.sidebar.selectbox("Sélectionnez une analyse :", [
        "Répartition des Fraudes", "Distribution des Transactions", "ECDF Transactions",
        "Entraînement du Modèle XGBoost", "Évaluation du Modèle"
    ])
    
    if analysis_type == "Répartition des Fraudes":
        fraud_count = train_df['isFraud'].value_counts()
        fig = px.pie(values=fraud_count.values, names=['Non frauduleuses', 'Frauduleuses'], 
                     title="Répartition des transactions", color_discrete_sequence=px.colors.qualitative.Set1)
        st.plotly_chart(fig)
    
    elif analysis_type == "Distribution des Transactions":
        fig = px.histogram(train_df['TransactionAmt'], nbins=100, title="Distribution des Montants des Transactions", 
                           color_discrete_sequence=['blue'])
        st.plotly_chart(fig)
    
    elif analysis_type == "ECDF Transactions":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(len(train_df[train_df['isFraud'] == 0])), 
            y=np.sort(train_df[train_df['isFraud'] == 0]['TransactionAmt']),
            mode='markers', name='Non Frauduleuses', opacity=0.5, marker=dict(color='lightgreen')))
        fig.add_trace(go.Scatter(
            x=np.arange(len(train_df[train_df['isFraud'] == 1])), 
            y=np.sort(train_df[train_df['isFraud'] == 1]['TransactionAmt']),
            mode='markers', name='Frauduleuses', opacity=0.5, marker=dict(color='lightpink')))
        fig.update_layout(title="ECDF - Transactions frauduleuses et non frauduleuses", 
                          xaxis_title="Index", yaxis_title="Montant")
        st.plotly_chart(fig)
    
    elif analysis_type == "Entraînement du Modèle XGBoost":
        st.subheader("🚀 Entraînement du modèle XGBoost")
        
        # Pré-traitement des données
        train_df.fillna(-999, inplace=True)
        for col in train_df.select_dtypes(include=['object']).columns:
            train_df[col] = train_df[col].astype('category').cat.codes
        train_df['uid'] = train_df['card1'].astype(str) + "_" + train_df['addr1'].astype(str) + "_" + train_df['P_emaildomain'].astype(str)
        train_df['uid'] = train_df['uid'].astype('category').cat.codes
        
        y = train_df['isFraud']
        X = train_df.drop(['isFraud', 'TransactionID'], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entraînement du modèle
        clf = xgb.XGBClassifier(
            n_estimators=500, max_depth=9, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, missing=-999,
            eval_metric='auc', use_label_encoder=False)
        
        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50, early_stopping_rounds=100)
        
        # Sauvegarde du modèle
        st.success("Modèle XGBoost entraîné avec succès !")
    
    elif analysis_type == "Évaluation du Modèle":
        st.subheader("📊 Évaluation du Modèle")
        
        # Prédictions
        y_pred = clf.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred)
        st.write(f"### Score AUC : {auc_score:.4f}")
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, (y_pred > 0.5).astype(int))
        fig_cm = ff.create_annotated_heatmap(
            z=cm, x=['Non-Fraude', 'Fraude'], y=['Non-Fraude', 'Fraude'], 
            colorscale='Blues', annotation_text=[[str(x) for x in row] for row in cm])
        fig_cm.update_layout(title="Matrice de Confusion", xaxis_title="Prédictions", yaxis_title="Valeurs Réelles")
        st.plotly_chart(fig_cm)
        
        # Courbe ROC
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve', line=dict(color='darkorange')))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Model', line=dict(dash='dash', color='gray')))
        fig_roc.update_layout(title="Courbe ROC", xaxis_title="Taux de Faux Positifs (FPR)", yaxis_title="Taux de Vrais Positifs (TPR)")
        st.plotly_chart(fig_roc)
        
        # Rapport de classification
        report = classification_report(y_test, (y_pred > 0.5).astype(int), output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.write("### Rapport de Classification :")
        st.dataframe(df_report)
