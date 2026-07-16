# ============================================
# CODE PFE - COMPLET // sk-ab9d198078fb432fa035e4f842bcead0 // https://darknet-api.onrender.com // https://darknet-detection.pages.dev
# C:\Users\vctr_on\Desktop\vscodePFE\Darknet.csv
# ============================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import time
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Configuration matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 10
os.environ['JOBLIB_MULTIPROCESSING'] = '0'

# Variables pour stocker les temps d'exécution
execution_times = {
    'total_start': time.time(),
    'steps': {}
}

def log_step_time(step_name, start_time):
    """Enregistre le temps d'exécution d'une étape"""
    elapsed = time.time() - start_time
    execution_times['steps'][step_name] = elapsed
    print(f"   ⏱️  {step_name}: {elapsed:.2f} secondes")
    return elapsed

print("="*60)
print("CODE PFE - VERSION COMPLÈTE (TOUS LES ALGORITHMES)")
print("SMOTE + FEATURE SELECTION sur TOUS les modèles")
print("="*60)

# ============================================
# PHASE 1 : PRÉPARATION DES DONNÉES
# ============================================

phase_start = time.time()
print("\n📂 Veuillez indiquer le chemin de votre fichier CSV :")
filepath = input("\n📁 Chemin du fichier: ").strip()

step_start = time.time()
if not os.path.exists(filepath) and not os.path.isabs(filepath):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filepath)
    
if not os.path.exists(filepath):
    print(f"\n❌ Fichier non trouvé: {filepath}")
    exit()

print(f"✅ Fichier trouvé : {filepath}")
log_step_time("Vérification fichier", step_start)

print("\n📥 Chargement...")
step_start = time.time()
df = pd.read_csv(filepath, low_memory=False)
print(f"✅ {df.shape[0]} lignes, {df.shape[1]} colonnes")
log_step_time("Chargement CSV", step_start)

# Nettoyage
print("\n🧹 Nettoyage...")
step_start = time.time()
initial_rows = len(df)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print(f"✅ {len(df)} lignes conservées")
log_step_time("Nettoyage données", step_start)

# Séparation features/labels
step_start = time.time()
colonnes_a_supprimer = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port',
                        'Protocol', 'Timestamp', 'Label']
if 'Label.1' in df.columns:
    colonnes_a_supprimer.append('Label.1')

y = df['Label'].copy()
X = df.drop(columns=colonnes_a_supprimer, errors='ignore')
print(f"✅ {X.shape[1]} features")
log_step_time("Séparation features/labels", step_start)

# Encodage
step_start = time.time()
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
log_step_time("Encodage labels", step_start)

# Split 70/30
step_start = time.time()
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)
log_step_time("Split train/test", step_start)

# Normalisation
step_start = time.time()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
log_step_time("Normalisation des données", step_start)

print(f"✅ Train: {X_train_scaled.shape[0]}, Test: {X_test_scaled.shape[0]}")

# Distribution des classes
step_start = time.time()
class_dist = df['Label'].value_counts()
log_step_time("Calcul distribution classes", step_start)

# ============================================
# GRAPHIQUE 1: Distribution des classes
# ============================================
graph_start = time.time()
plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(class_dist)), class_dist.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(class_dist)])
plt.title('Distribution des classes dans le dataset', fontsize=14, fontweight='bold')
plt.xlabel('Types de trafic', fontsize=12)
plt.ylabel("Nombre d'échantillons", fontsize=12)
plt.xticks(range(len(class_dist)), class_dist.index, rotation=45, ha='right')
for bar, val in zip(bars, class_dist.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(class_dist.values)*0.01,
             f'{val}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('1_distribution_classes.png', dpi=150, bbox_inches='tight')
plt.close()
log_step_time("Graphique 1: distribution_classes", graph_start)
print("✅ Graphique 1: 1_distribution_classes.png")

log_step_time("PHASE 1: Préparation des données", phase_start)

# ============================================
# PHASE 2 : ENTRAÎNEMENT DES MODÈLES ML
# ============================================

phase_start = time.time()
print("\n" + "="*60)
print("ENTRAÎNEMENT DES MODÈLES ML")
print("="*60)

results = {}
models = {}

def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test):
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    y_pred = model.predict(X_test)
    return {
        'model': model,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1-Score': f1_score(y_test, y_pred, average='weighted'),
        'Train Time': train_time,
        'predictions': y_pred
    }

# 1. Decision Tree
print("\n🌀 Decision Tree...")
step_start = time.time()
dt = DecisionTreeClassifier(random_state=42)
results['Decision Tree'] = train_and_evaluate(dt, 'DT', X_train_scaled, X_test_scaled, y_train, y_test)
models['Decision Tree'] = results['Decision Tree']['model']
log_step_time("Decision Tree", step_start)
print(f"   ✅ F1={results['Decision Tree']['F1-Score']:.4f}")

# 2. Random Forest
print("\n🌀 Random Forest (n_estimators=100)...")
step_start = time.time()
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
results['Random Forest'] = train_and_evaluate(rf, 'RF', X_train_scaled, X_test_scaled, y_train, y_test)
models['Random Forest'] = results['Random Forest']['model']
log_step_time("Random Forest", step_start)
print(f"   ✅ F1={results['Random Forest']['F1-Score']:.4f}")

# 3. XGBoost
print("\n🌀 XGBoost (n_estimators=100)...")
step_start = time.time()
xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss', verbosity=0, n_jobs=-1)
results['XGBoost'] = train_and_evaluate(xgb, 'XGB', X_train_scaled, X_test_scaled, y_train, y_test)
models['XGBoost'] = results['XGBoost']['model']
log_step_time("XGBoost", step_start)
print(f"   ✅ F1={results['XGBoost']['F1-Score']:.4f}")

# 4. SVM
print("\n🌀 SVM...")
step_start = time.time()
svm = SVC(kernel='rbf', probability=True, random_state=42, cache_size=1000)
results['SVM'] = train_and_evaluate(svm, 'SVM', X_train_scaled, X_test_scaled, y_train, y_test)
models['SVM'] = results['SVM']['model']
log_step_time("SVM", step_start)
print(f"   ✅ F1={results['SVM']['F1-Score']:.4f}")

# 5. KNN
print("\n🌀 KNN...")
step_start = time.time()
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
results['KNN'] = train_and_evaluate(knn, 'KNN', X_train_scaled, X_test_scaled, y_train, y_test)
models['KNN'] = results['KNN']['model']
log_step_time("KNN", step_start)
print(f"   ✅ F1={results['KNN']['F1-Score']:.4f}")

# 6. VotingClassifier
print("\n🗳️ VotingClassifier (soft)...")
step_start = time.time()
voting_clf = VotingClassifier(
    estimators=[('dt', models['Decision Tree']), ('rf', models['Random Forest']),
                ('xgb', models['XGBoost']), ('svm', models['SVM']), ('knn', models['KNN'])],
    voting='soft', n_jobs=-1
)
start = time.time()
voting_clf.fit(X_train_scaled, y_train)
train_time = time.time() - start
y_pred_voting = voting_clf.predict(X_test_scaled)
results['Voting (soft)'] = {
    'model': voting_clf,
    'Accuracy': accuracy_score(y_test, y_pred_voting),
    'Precision': precision_score(y_test, y_pred_voting, average='weighted'),
    'Recall': recall_score(y_test, y_pred_voting, average='weighted'),
    'F1-Score': f1_score(y_test, y_pred_voting, average='weighted'),
    'Train Time': train_time,
    'predictions': y_pred_voting
}
models['Voting (soft)'] = voting_clf
log_step_time("Voting Classifier", step_start)
print(f"   ✅ F1={results['Voting (soft)']['F1-Score']:.4f}")

log_step_time("PHASE 2: Entraînement modèles ML", phase_start)

# ============================================
# GRAPHIQUE 2: Comparaison F1-Score des modèles
# ============================================
graph_start = time.time()
plt.figure(figsize=(12, 6))
model_names = list(results.keys())
f1_scores = [results[m]['F1-Score'] for m in model_names]
colors_bar = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
bars = plt.bar(range(len(model_names)), f1_scores, color=colors_bar)
plt.title('Comparaison des performances des modèles (F1-Score)', fontsize=14, fontweight='bold')
plt.xlabel('Modèles', fontsize=12)
plt.ylabel('F1-Score', fontsize=12)
plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
plt.ylim(0, 1.05)
for bar, score in zip(bars, f1_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{score:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('2_comparaison_modeles_f1.png', dpi=150, bbox_inches='tight')
plt.close()
log_step_time("Graphique 2: comparaison_modeles_f1", graph_start)
print("✅ Graphique 2: 2_comparaison_modeles_f1.png")

# ============================================
# GRAPHIQUE 3: Temps d'entraînement
# ============================================
graph_start = time.time()
plt.figure(figsize=(12, 6))
train_times = [results[m]['Train Time'] for m in model_names]
bars = plt.bar(range(len(model_names)), train_times, color='lightcoral')
plt.title("Temps d'entraînement des modèles", fontsize=14, fontweight='bold')
plt.xlabel('Modèles', fontsize=12)
plt.ylabel('Temps (secondes)', fontsize=12)
plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
for bar, time_val in zip(bars, train_times):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(train_times)*0.01,
             f'{time_val:.1f}s', ha='center', va='bottom', fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('3_temps_entrainement.png', dpi=150, bbox_inches='tight')
plt.close()
log_step_time("Graphique 3: temps_entrainement", graph_start)
print("✅ Graphique 3: 3_temps_entrainement.png")

# ============================================
# PHASE 3 : DNN
# ============================================

phase_start = time.time()
print("\n" + "="*60)
print("DNN (epochs=50)")
print("="*60)

step_start = time.time()
y_train_dnn = to_categorical(y_train, num_classes=4)
y_test_dnn = to_categorical(y_test, num_classes=4)

model_dnn = Sequential([
    Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')
])

model_dnn.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
log_step_time("Construction DNN", step_start)

print("🚀 Entraînement DNN (50 epochs)...")
step_start = time.time()
history = model_dnn.fit(
    X_train_scaled, y_train_dnn,
    epochs=50,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)
dnn_time = time.time() - step_start
log_step_time("Entraînement DNN", step_start)

step_start = time.time()
y_pred_dnn_proba = model_dnn.predict(X_test_scaled, verbose=0)
y_pred_dnn = np.argmax(y_pred_dnn_proba, axis=1)
log_step_time("Prédictions DNN", step_start)

results['DNN'] = {
    'model': model_dnn,
    'Accuracy': accuracy_score(y_test, y_pred_dnn),
    'Precision': precision_score(y_test, y_pred_dnn, average='weighted'),
    'Recall': recall_score(y_test, y_pred_dnn, average='weighted'),
    'F1-Score': f1_score(y_test, y_pred_dnn, average='weighted'),
    'Train Time': dnn_time,
    'predictions': y_pred_dnn
}
print(f"✅ DNN terminé: F1={results['DNN']['F1-Score']:.4f}")
log_step_time("PHASE 3: DNN", phase_start)

# ============================================
# GRAPHIQUE 4: Courbes d'apprentissage DNN
# ============================================
graph_start = time.time()
plt.figure(figsize=(14, 5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2, color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='red')
plt.title("Courbe d'apprentissage - Accuracy", fontsize=12, fontweight='bold')
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2, color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='red')
plt.title("Courbe d'apprentissage - Loss", fontsize=12, fontweight='bold')
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.legend()
plt.grid(True, alpha=0.3)

plt.suptitle('Apprentissage du DNN (50 epochs)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('4_courbes_apprentissage_dnn.png', dpi=150, bbox_inches='tight')
plt.close()
log_step_time("Graphique 4: courbes_apprentissage_dnn", graph_start)
print("✅ Graphique 4: 4_courbes_apprentissage_dnn.png")

# ============================================
# GRAPHIQUE 5: Matrice de confusion - Voting Classifier
# ============================================
graph_start = time.time()
cm_voting = confusion_matrix(y_test, y_pred_voting)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_voting, annot=True, fmt='d', cmap='Greens', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.title('Matrice de confusion - Voting Classifier', fontsize=14, fontweight='bold')
plt.xlabel('Prédictions', fontsize=12)
plt.ylabel('Vérités terrain', fontsize=12)
plt.tight_layout()
plt.savefig('5_matrice_confusion_voting.png', dpi=150, bbox_inches='tight')
plt.close()
log_step_time("Graphique 5: matrice_confusion_voting", graph_start)
print("✅ Graphique 5: 5_matrice_confusion_voting.png")

# ============================================
# PHASE 4 : FEATURE SELECTION (Importance des features)
# ============================================

phase_start = time.time()
print("\n" + "="*60)
print("FEATURE SELECTION - IMPORTANCE DES FEATURES")
print("="*60)

step_start = time.time()
feature_names = X.columns
importances = models['Random Forest'].feature_importances_
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
log_step_time("Calcul importance features", step_start)

print("\n🏆 TOP 20 FEATURES:")
for i, row in feature_importance_df.head(20).iterrows():
    print(f"   {i+1:2}. {row['Feature'][:50]:<50} {row['Importance']:.4f}")

log_step_time("PHASE 4: Feature Selection", phase_start)

# ============================================
# GRAPHIQUE 6: Top 20 features importantes
# ============================================
graph_start = time.time()
plt.figure(figsize=(12, 8))
top20_plot = feature_importance_df.head(20)
plt.barh(range(len(top20_plot)), top20_plot['Importance'].values, color='steelblue')
plt.yticks(range(len(top20_plot)), top20_plot['Feature'].values)
plt.xlabel('Importance', fontsize=12)
plt.title('Top 20 features les plus importantes (Random Forest)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('6_top20_features.png', dpi=150, bbox_inches='tight')
plt.close()
log_step_time("Graphique 6: top20_features", graph_start)
print("✅ Graphique 6: 6_top20_features.png")

# ============================================
# PHASE 5 : SMOTE (TOUS les modèles)
# ============================================

phase_start = time.time()
print("\n" + "="*60)
print("SMOTE ANALYSIS - TOUS LES MODÈLES")
print("="*60)

print("\nDistribution avant SMOTE:")
for classe in range(4):
    count = np.sum(y_train == classe)
    nom = label_encoder.inverse_transform([classe])[0]
    print(f"   {nom}: {count} ({count/len(y_train)*100:.1f}%)")

print("\n🌀 Application SMOTE...")
step_start = time.time()
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
log_step_time("Application SMOTE", step_start)
print(f"✅ SMOTE terminé")

log_step_time("PHASE 5: SMOTE Analysis", phase_start)

# ============================================
# GRAPHIQUE 7: Distribution AVANT vs APRÈS SMOTE
# ============================================
graph_start = time.time()
before_smote_counts = Counter(y_train)
after_smote_counts = Counter(y_train_smote)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

classes_names = label_encoder.inverse_transform(sorted(before_smote_counts.keys()))
before_values = [before_smote_counts[i] for i in sorted(before_smote_counts.keys())]
bars1 = axes[0].bar(range(len(before_values)), before_values, color='#FF6B6B')
axes[0].set_title('Distribution AVANT SMOTE', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Classes', fontsize=10)
axes[0].set_ylabel("Nombre d'échantillons", fontsize=10)
axes[0].set_xticks(range(len(classes_names)))
axes[0].set_xticklabels(classes_names, rotation=45, ha='right')
for bar, val in zip(bars1, before_values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val}', 
                 ha='center', va='bottom', fontsize=9)

after_values = [after_smote_counts[i] for i in sorted(after_smote_counts.keys())]
bars2 = axes[1].bar(range(len(after_values)), after_values, color='#4ECDC4')
axes[1].set_title('Distribution APRÈS SMOTE', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Classes', fontsize=10)
axes[1].set_ylabel("Nombre d'échantillons", fontsize=10)
axes[1].set_xticks(range(len(classes_names)))
axes[1].set_xticklabels(classes_names, rotation=45, ha='right')
for bar, val in zip(bars2, after_values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val}', 
                 ha='center', va='bottom', fontsize=9)

plt.suptitle("Impact de SMOTE sur l'équilibrage des classes", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('7_comparaison_smote.png', dpi=150, bbox_inches='tight')
plt.close()
log_step_time("Graphique 7: comparaison_smote", graph_start)
print("✅ Graphique 7: 7_comparaison_smote.png")

# ============================================
# ENTRAÎNEMENT AVEC SMOTE - TOUS LES MODÈLES
# ============================================

step_start = time.time()
print("\n🚀 Entraînement avec SMOTE (TOUS les modèles)...")

# Tous les modèles pour SMOTE (comme demandé par l'encadreur)
models_smote = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss', verbosity=0, n_jobs=-1),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42, cache_size=1000),
    'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
}

results_smote = {}
print("\n📊 Résultats avec SMOTE:")
for name, model in models_smote.items():
    print(f"   {name}...", end=" ", flush=True)
    start = time.time()
    model.fit(X_train_smote, y_train_smote)
    train_time = time.time() - start
    y_pred = model.predict(X_test_scaled)
    f1_smote = f1_score(y_test, y_pred, average='weighted')
    results_smote[name] = {
        'F1-Score': f1_smote,
        'Train Time': train_time,
        'predictions': y_pred
    }
    print(f"F1={f1_smote:.4f} ({train_time:.1f}s)")

# Voting avec SMOTE
print(f"   Voting (soft)...", end=" ", flush=True)
voting_smote = VotingClassifier(
    estimators=[('dt', models_smote['Decision Tree']), 
                ('rf', models_smote['Random Forest']),
                ('xgb', models_smote['XGBoost']), 
                ('svm', models_smote['SVM']), 
                ('knn', models_smote['KNN'])],
    voting='soft', n_jobs=-1
)
start = time.time()
voting_smote.fit(X_train_smote, y_train_smote)
train_time = time.time() - start
y_pred_voting_smote = voting_smote.predict(X_test_scaled)
f1_voting_smote = f1_score(y_test, y_pred_voting_smote, average='weighted')
results_smote['Voting (soft)'] = {
    'F1-Score': f1_voting_smote,
    'Train Time': train_time,
    'predictions': y_pred_voting_smote
}
print(f"F1={f1_voting_smote:.4f} ({train_time:.1f}s)")

# DNN avec SMOTE
print(f"   DNN...", end=" ", flush=True)
y_train_dnn_smote = to_categorical(y_train_smote, num_classes=4)
model_dnn_smote = Sequential([
    Dense(128, input_dim=X_train_smote.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')
])
model_dnn_smote.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
start = time.time()
model_dnn_smote.fit(X_train_smote, y_train_dnn_smote, epochs=50, batch_size=128, validation_split=0.2, verbose=0)
train_time = time.time() - start
y_pred_dnn_smote_proba = model_dnn_smote.predict(X_test_scaled, verbose=0)
y_pred_dnn_smote = np.argmax(y_pred_dnn_smote_proba, axis=1)
f1_dnn_smote = f1_score(y_test, y_pred_dnn_smote, average='weighted')
results_smote['DNN'] = {
    'F1-Score': f1_dnn_smote,
    'Train Time': train_time,
    'predictions': y_pred_dnn_smote
}
print(f"F1={f1_dnn_smote:.4f} ({train_time:.1f}s)")

log_step_time("Entraînement avec SMOTE (tous modèles)", step_start)

# ============================================
# GRAPHIQUE 8: Comparaison performances avec/sans SMOTE (TOUS modèles)
# ============================================
graph_start = time.time()
models_comp = ['Decision Tree', 'Random Forest', 'XGBoost', 'SVM', 'KNN', 'Voting (soft)', 'DNN']
f1_without = [results[m]['F1-Score'] for m in models_comp if m in results]
f1_with = [results_smote[m]['F1-Score'] for m in models_comp if m in results_smote]

x = np.arange(len(models_comp))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 7))
bars1 = ax.bar(x - width/2, f1_without, width, label='Sans SMOTE', color='#FF6B6B')
bars2 = ax.bar(x + width/2, f1_with, width, label='Avec SMOTE', color='#4ECDC4')

ax.set_xlabel('Modèles', fontsize=12)
ax.set_ylabel('F1-Score', fontsize=12)
ax.set_title('Comparaison des performances AVANT et APRÈS SMOTE (tous modèles)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_comp, rotation=45, ha='right')
ax.legend()
ax.set_ylim(0, 1.05)
ax.grid(axis='y', alpha=0.3)

for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.4f}', 
            ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.4f}', 
            ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('8_comparaison_smote_performances.png', dpi=150, bbox_inches='tight')
plt.close()
log_step_time("Graphique 8: comparaison_smote_performances", graph_start)
print("✅ Graphique 8: 8_comparaison_smote_performances.png")

# ============================================
# PHASE 6 : FEATURE SELECTION (Top 20 et Top 30) - TOUS LES MODÈLES
# ============================================

phase_start = time.time()
print("\n" + "="*60)
print("FEATURE SELECTION - TOP 20 ET TOP 30 (TOUS LES MODÈLES)")
print("="*60)

step_start = time.time()
top20_features = feature_importance_df.head(20)['Feature'].tolist()
top30_features = feature_importance_df.head(30)['Feature'].tolist()

def filter_features(X_data, features_list, original_columns):
    indices = [list(original_columns).index(f) for f in features_list]
    return X_data[:, indices]

X_train_top20 = filter_features(X_train_scaled, top20_features, feature_names)
X_test_top20 = filter_features(X_test_scaled, top20_features, feature_names)
X_train_top30 = filter_features(X_train_scaled, top30_features, feature_names)
X_test_top30 = filter_features(X_test_scaled, top30_features, feature_names)
log_step_time("Préparation features Top20/Top30", step_start)

# Tous les modèles pour feature selection (comme demandé par l'encadreur)
models_fs = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss', verbosity=0, n_jobs=-1),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42, cache_size=1000),
    'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
}

results_top20 = {}
results_top30 = {}

print("\n📊 Évaluation TOP 20 features (tous modèles):")
step_start_fs = time.time()
for name, model in models_fs.items():
    print(f"   {name}...", end=" ", flush=True)
    start = time.time()
    model.fit(X_train_top20, y_train)
    train_time = time.time() - start
    y_pred = model.predict(X_test_top20)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results_top20[name] = {'F1-Score': f1, 'Train Time': train_time}
    print(f"F1={f1:.4f} ({train_time:.1f}s)")
log_step_time("Top20 features (tous modèles)", step_start_fs)

print("\n📊 Évaluation TOP 30 features (tous modèles):")
step_start_fs = time.time()
for name, model in models_fs.items():
    print(f"   {name}...", end=" ", flush=True)
    start = time.time()
    model.fit(X_train_top30, y_train)
    train_time = time.time() - start
    y_pred = model.predict(X_test_top30)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results_top30[name] = {'F1-Score': f1, 'Train Time': train_time}
    print(f"F1={f1:.4f} ({train_time:.1f}s)")
log_step_time("Top30 features (tous modèles)", step_start_fs)

# Voting avec Top 20
print(f"\n🗳️ Voting avec Top 20...")
voting_top20 = VotingClassifier(
    estimators=[('dt', models_fs['Decision Tree']), 
                ('rf', models_fs['Random Forest']),
                ('xgb', models_fs['XGBoost']), 
                ('svm', models_fs['SVM']), 
                ('knn', models_fs['KNN'])],
    voting='soft', n_jobs=-1
)
voting_top20.fit(X_train_top20, y_train)
y_pred_voting_top20 = voting_top20.predict(X_test_top20)
f1_voting_top20 = f1_score(y_test, y_pred_voting_top20, average='weighted')
results_top20['Voting (soft)'] = {'F1-Score': f1_voting_top20, 'Train Time': 0}
print(f"   Voting F1={f1_voting_top20:.4f}")

# Voting avec Top 30
print(f"\n🗳️ Voting avec Top 30...")
voting_top30 = VotingClassifier(
    estimators=[('dt', models_fs['Decision Tree']), 
                ('rf', models_fs['Random Forest']),
                ('xgb', models_fs['XGBoost']), 
                ('svm', models_fs['SVM']), 
                ('knn', models_fs['KNN'])],
    voting='soft', n_jobs=-1
)
voting_top30.fit(X_train_top30, y_train)
y_pred_voting_top30 = voting_top30.predict(X_test_top30)
f1_voting_top30 = f1_score(y_test, y_pred_voting_top30, average='weighted')
results_top30['Voting (soft)'] = {'F1-Score': f1_voting_top30, 'Train Time': 0}
print(f"   Voting F1={f1_voting_top30:.4f}")

# DNN avec Top 20
print(f"\n🧠 DNN avec Top 20...")
y_train_dnn_top20 = to_categorical(y_train, num_classes=4)
model_dnn_top20 = Sequential([
    Dense(128, input_dim=X_train_top20.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')
])
model_dnn_top20.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model_dnn_top20.fit(X_train_top20, y_train_dnn_top20, epochs=50, batch_size=128, validation_split=0.2, verbose=0)
y_pred = np.argmax(model_dnn_top20.predict(X_test_top20, verbose=0), axis=1)
f1_dnn_top20 = f1_score(y_test, y_pred, average='weighted')
results_top20['DNN'] = {'F1-Score': f1_dnn_top20, 'Train Time': 0}
print(f"   DNN F1={f1_dnn_top20:.4f}")

# DNN avec Top 30
print(f"\n🧠 DNN avec Top 30...")
y_train_dnn_top30 = to_categorical(y_train, num_classes=4)
model_dnn_top30 = Sequential([
    Dense(128, input_dim=X_train_top30.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')
])
model_dnn_top30.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model_dnn_top30.fit(X_train_top30, y_train_dnn_top30, epochs=50, batch_size=128, validation_split=0.2, verbose=0)
y_pred = np.argmax(model_dnn_top30.predict(X_test_top30, verbose=0), axis=1)
f1_dnn_top30 = f1_score(y_test, y_pred, average='weighted')
results_top30['DNN'] = {'F1-Score': f1_dnn_top30, 'Train Time': 0}
print(f"   DNN F1={f1_dnn_top30:.4f}")

log_step_time("PHASE 6: Feature Selection Top20/Top30 (tous modèles)", phase_start)

# ============================================
# GRAPHIQUE 9: Comparaison Top 20 vs Top 30 vs Complet (tous modèles)
# ============================================
graph_start = time.time()
all_models = ['Decision Tree', 'Random Forest', 'XGBoost', 'SVM', 'KNN', 'Voting (soft)', 'DNN']

x = np.arange(len(all_models))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 8))
f1_complet_list = [results[m]['F1-Score'] for m in all_models if m in results]
f1_20_list = [results_top20[m]['F1-Score'] for m in all_models if m in results_top20]
f1_30_list = [results_top30[m]['F1-Score'] for m in all_models if m in results_top30]

bars1 = ax.bar(x - width, f1_complet_list, width, label='Toutes features', color='#2E86AB')
bars2 = ax.bar(x, f1_20_list, width, label='Top 20 features', color='#A23B72')
bars3 = ax.bar(x + width, f1_30_list, width, label='Top 30 features', color='#F18F01')

ax.set_xlabel('Modèles', fontsize=12)
ax.set_ylabel('F1-Score', fontsize=12)
ax.set_title('Impact de la sélection des features sur les performances (tous modèles)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(all_models, rotation=45, ha='right')
ax.legend()
ax.set_ylim(0, 1.05)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('9_comparaison_feature_selection.png', dpi=150, bbox_inches='tight')
plt.close()
log_step_time("Graphique 9: comparaison_feature_selection", graph_start)
print("✅ Graphique 9: 9_comparaison_feature_selection.png")

# ============================================
# GRAPHIQUE 10: Courbes ROC
# ============================================
graph_start = time.time()
print("\n📊 Graphique 10: Courbes ROC...")

y_test_bin = label_binarize(y_test, classes=[0,1,2,3])
models_proba = {
    'Random Forest': rf.predict_proba(X_test_scaled),
    'XGBoost': xgb.predict_proba(X_test_scaled),
    'DNN': y_pred_dnn_proba
}

plt.figure(figsize=(12, 8))
colors_roc = ['blue', 'red', 'green', 'orange']

for model_name, y_score in models_proba.items():
    for i in range(4):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors_roc[i], linestyle='-', 
                label=f'{model_name} - {label_encoder.classes_[i]} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs (FPR)', fontsize=12)
plt.ylabel('Taux de vrais positifs (TPR)', fontsize=12)
plt.title('Courbes ROC - Comparaison des modèles', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('10_courbes_roc.png', dpi=150, bbox_inches='tight')
plt.close()
log_step_time("Graphique 10: courbes_roc", graph_start)
print("✅ Graphique 10: 10_courbes_roc.png")

# ============================================
# SYSTÈME DE DÉCISION FINALE
# ============================================

phase_start = time.time()
print("\n" + "="*60)
print("SYSTÈME DE DÉCISION FINALE")
print("="*60)

def calculer_score_risque(pred_vote, pred_dnn):
    score = 0
    if pred_vote in [0, 1]:
        score += 40
    if pred_dnn in [0, 1]:
        score += 30
    return score

scores_risque = [calculer_score_risque(y_pred_voting[i], y_pred_dnn[i]) for i in range(len(y_test))]
log_step_time("Calcul scores risque", phase_start)

# ============================================
# GRAPHIQUE 11: Distribution des scores de risque
# ============================================
graph_start = time.time()
plt.figure(figsize=(10, 6))
plt.hist(scores_risque, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(x=70, color='red', linestyle='--', linewidth=2, label='Seuil risque élevé (70)')
plt.axvline(x=40, color='orange', linestyle='--', linewidth=2, label='Seuil risque moyen (40)')
plt.title('Distribution des scores de risque', fontsize=14, fontweight='bold')
plt.xlabel('Score de risque', fontsize=12)
plt.ylabel("Nombre d'échantillons", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('11_distribution_risques.png', dpi=150, bbox_inches='tight')
plt.close()
log_step_time("Graphique 11: distribution_risques", graph_start)
print("✅ Graphique 11: 11_distribution_risques.png")

# ============================================
# GRAPHIQUE 12: Métriques par classe
# ============================================
graph_start = time.time()
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_voting, average=None)

x = np.arange(len(label_encoder.classes_))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 7))
bars1 = ax.bar(x - width, precision, width, label='Précision', color='#2E86AB')
bars2 = ax.bar(x, recall, width, label='Rappel', color='#A23B72')
bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#F18F01')

ax.set_xlabel('Classes', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Métriques détaillées par classe - Voting Classifier', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(label_encoder.classes_, rotation=45, ha='right')
ax.legend()
ax.set_ylim(0, 1.05)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('12_metriques_par_classe.png', dpi=150, bbox_inches='tight')
plt.close()
log_step_time("Graphique 12: metriques_par_classe", graph_start)
print("✅ Graphique 12: 12_metriques_par_classe.png")

# ============================================
# TABLEAU COMPARATIF FINAL
# ============================================

print("\n" + "="*60)
print("TABLEAU COMPARATIF FINAL")
print("="*80)

comparison_df = pd.DataFrame({
    'Modèle': list(results.keys()),
    'Accuracy': [results[m]['Accuracy'] for m in results.keys()],
    'Precision': [results[m]['Precision'] for m in results.keys()],
    'Recall': [results[m]['Recall'] for m in results.keys()],
    'F1-Score': [results[m]['F1-Score'] for m in results.keys()],
    'Temps (s)': [results[m]['Train Time'] for m in results.keys()]
})
print(comparison_df.to_string(index=False))

# ============================================
# TABLEAU SMOTE COMPARATIF
# ============================================

print("\n" + "="*60)
print("TABLEAU SMOTE - COMPARAISON AVANT/APRÈS")
print("="*80)

smote_comp_df = pd.DataFrame({
    'Modèle': list(results_smote.keys()),
    'F1-Score (sans SMOTE)': [results[m]['F1-Score'] for m in results_smote.keys() if m in results],
    'F1-Score (avec SMOTE)': [results_smote[m]['F1-Score'] for m in results_smote.keys()],
    'Différence': [results_smote[m]['F1-Score'] - results[m]['F1-Score'] if m in results else 0 for m in results_smote.keys()]
})
print(smote_comp_df.to_string(index=False))

# ============================================
# TABLEAU FEATURE SELECTION COMPARATIF
# ============================================

print("\n" + "="*60)
print("TABLEAU FEATURE SELECTION - COMPARAISON")
print("="*80)

fs_comp_data = []
for m in all_models:
    if m in results and m in results_top20 and m in results_top30:
        fs_comp_data.append({
            'Modèle': m,
            'Complet': results[m]['F1-Score'],
            'Top 20': results_top20[m]['F1-Score'],
            'Top 30': results_top30[m]['F1-Score']
        })

fs_comp_df = pd.DataFrame(fs_comp_data)
print(fs_comp_df.to_string(index=False))

# ============================================
# EXPORT DES MODÈLES POUR PYTHONANYWHERE (AJOUT)
# ============================================

print("\n" + "="*60)
print("EXPORT DES MODÈLES POUR DÉPLOIEMENT (PythonAnywhere)")
print("="*60)

import joblib

# Exporter le modèle XGBoost (le meilleur avec Voting)
joblib.dump(xgb, 'modele_xgboost.pkl')
print("✅ modele_xgboost.pkl exporté")

# Exporter le scaler (normalisation)
joblib.dump(scaler, 'scaler.pkl')
print("✅ scaler.pkl exporté")

# Exporter le label encoder (noms des classes)
joblib.dump(label_encoder, 'label_encoder.pkl')
print("✅ label_encoder.pkl exporté")

print("\n✅ Export terminé ! Les fichiers suivants ont été créés :")
print("   - modele_xgboost.pkl")
print("   - scaler.pkl")
print("   - label_encoder.pkl")
print("\n📌 Ces fichiers sont prêts à être uploadés sur PythonAnywhere.")

# ============================================
# RÉCAPITULATIF DES TEMPS D'EXÉCUTION
# ============================================

total_time = time.time() - execution_times['total_start']

print("\n" + "="*60)
print("⏱️  RÉCAPITULATIF DES TEMPS D'EXÉCUTION")
print("="*60)

print("\n📊 DÉTAIL PAR ÉTAPE:")
for step_name, step_time in execution_times['steps'].items():
    print(f"   • {step_name:<45} : {step_time:>8.2f} secondes")

print(f"\n⏰ TEMPS TOTAL D'EXÉCUTION: {total_time:.2f} secondes ({total_time/60:.2f} minutes)")

# ============================================
# RÉCAPITULATIF FINAL
# ============================================

print("\n" + "="*60)
print("📊 RÉCAPITULATIF DES 12 GRAPHIQUES GÉNÉRÉS")
print("="*60)
print("\n1.  distribution_classes.png          - Distribution des classes")
print("2.  comparaison_modeles_f1.png         - Comparaison F1-Score des modèles")
print("3.  temps_entrainement.png             - Temps d'entraînement")
print("4.  courbes_apprentissage_dnn.png      - Courbes d'apprentissage DNN")
print("5.  matrice_confusion_voting.png       - Matrice de confusion Voting")
print("6.  top20_features.png                 - Top 20 features importantes")
print("7.  comparaison_smote.png              - Distribution AVANT/APRÈS SMOTE")
print("8.  comparaison_smote_performances.png - Performance avec/sans SMOTE (tous modèles)")
print("9.  comparaison_feature_selection.png  - Top 20 vs Top 30 vs Complet (tous modèles)")
print("10. courbes_roc.png                    - Courbes ROC + AUC")
print("11. distribution_risques.png           - Distribution scores risque")
print("12. metriques_par_classe.png           - Métriques détaillées par classe")
print("="*60)

print("\n🏆 MEILLEUR MODÈLE:")
best_model = max(results.items(), key=lambda x: x[1]['F1-Score'] if x[0] != 'SVM' else 0)
print(f"   {best_model[0]} = F1-Score: {best_model[1]['F1-Score']:.4f}")

print("\n" + "="*60)
print("   • RF n_estimators=100")
print("   • XGB n_estimators=100")
print("   • DNN epochs=50")
print("   • Split 70/30")
print("   • SMOTE sur TOUS les modèles (DT, RF, XGB, SVM, KNN, Voting, DNN)")
print("   • Feature Selection sur TOUS les modèles (Top 20 et Top 30)")
print("   • 12 graphiques générés avec succès")
print("   • Modèles exportés pour PythonAnywhere (XGBoost, scaler, label_encoder)")
print(f"   • Temps total d'exécution: {total_time:.2f} secondes ({total_time/60:.2f} minutes)")
print("="*60)