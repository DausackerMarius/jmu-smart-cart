"""
=========================================================================================
JMU SMART SUPERMARKET: EVALUATION SÄULE A (MACHINE LEARNING SENSORIK)
=========================================================================================
Dieses Skript ist komplett vom laufenden System isoliert. Es dient ausschließlich 
der wissenschaftlichen Auswertung unseres XGBoost-Modells für die Dokumentation.

WICHTIGER VERTEIDIGUNGS-ASPEKT (Data Leakage Prevention):
Wir nutzen hier strikt die 'test_data_holdout.csv'. Diese Daten hat das Modell im 
Training NIEMALS gesehen. Da das Testset zufällig aus dem ganzen Jahr gezogen wurde 
(Random Split), verzichten wir bewusst auf durchgehende Linien-Plots (Time-Series) 
und nutzen stattdessen wissenschaftlich korrekte Korrelations- und Verteilungs-Plots.
=========================================================================================
"""

# WICHTIG: Backend für Headless-Server setzen (Verhindert TclErrors ohne Monitor)
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix

# --- KONFIGURATION ---
TEST_DATA_FILE = "test_data_holdout.csv"
MODEL_FILE = "traffic_model_xgboost.pkl"
OUTPUT_DIR = "eval_plots"

# Akademisches Plot-Design für eine saubere Darstellung in Papern (Read The Docs)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

def ensure_environment():
    """Prüft, ob alle benötigten Dateien existieren und erstellt den Output-Ordner."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    if not os.path.exists(TEST_DATA_FILE) or not os.path.exists(MODEL_FILE):
        raise FileNotFoundError("❌ Testdaten oder Modell fehlen! Bitte erst train_model_optuna.py ausführen.")

def load_and_predict():
    """Lädt das isolierte Testset und generiert die Vorhersagen."""
    print("1. Lade isoliertes Hold-Out Testset und Modell-Artefakte...")
    
    df_test = pd.read_csv(TEST_DATA_FILE)
    df_test['timestamp_dt'] = pd.to_datetime(df_test['timestamp_dt'])
    
    with open(MODEL_FILE, 'rb') as f:
        artifacts = pickle.load(f)
        
    model = artifacts['model']
    features = artifacts['features']

    print("2. Generiere Vorhersagen auf ungesehenen Daten...")
    # Sichert die exakte Spalten-Reihenfolge ab, die das XGBoost-Modell erwartet
    X_test = df_test[features]
    
    # Im Training haben wir den Logarithmus vorhergesagt (log1p). 
    # Für eine interpretierbare Auswertung in der realen Welt rechnen wir dies mit expm1 zurück.
    y_test_log = df_test['true_target']
    
    # np.clip verhindert extrem unwahrscheinliche, aber mögliche negative Vorhersagen des Boosters
    preds_log = np.clip(model.predict(X_test), 0, None)
    
    df_test['pred_load_real'] = np.expm1(preds_log)
    df_test['true_load_real'] = np.expm1(y_test_log)
    
    # Absoluten Fehler pro Zeile berechnen (Wichtig für die spätere Stunden-Analyse)
    df_test['abs_error'] = np.abs(df_test['true_load_real'] - df_test['pred_load_real'])
    
    return df_test, model, features

def plot_feature_importance(model, features):
    """
    Explainable AI (XAI): Zeigt, welche Variablen das Modell am stärksten nutzt.
    Für Enterprise-KI extrem wichtig, da "Blackbox"-Modelle in der Industrie 
    nicht akzeptiert werden. Wir müssen beweisen, dass das Modell logisch entscheidet.
    """
    print("3. Extrahiere Feature Importances (Explainability)...")
    importances = model.feature_importances_
    df_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
    
    # Wir zeigen nur die Top 12 Features, um das Diagramm übersichtlich zu halten
    df_imp = df_imp.sort_values('Importance', ascending=False).head(12)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_imp, x='Importance', y='Feature', palette='viridis')
    plt.title('Feature Importance (Was beeinflusst den Stau am meisten?)', fontweight='bold')
    plt.xlabel('Relativer Einfluss (Gain)')
    plt.ylabel('Modell-Variablen')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ml_feature_importance.png'), dpi=300)
    plt.close()

def plot_residuals(df):
    """
    Zeichnet ein Histogramm der Vorhersagefehler.
    Beweist, dass das Modell keinen systematischen Bias hat (Glockenkurve um die Null).
    """
    print("4. Erstelle Residual-Analyse (Fehlerverteilung)...")
    residuals = df['true_load_real'] - df['pred_load_real']
    
    mae = mean_absolute_error(df['true_load_real'], df['pred_load_real'])
    rmse = np.sqrt(mean_squared_error(df['true_load_real'], df['pred_load_real']))
    r2 = r2_score(df['true_load_real'], df['pred_load_real'])
    
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=60, kde=True, color='#3498db', edgecolor='black')
    plt.axvline(x=0, color='#e74c3c', linestyle='--', linewidth=2, label='Perfekte Vorhersage (0 Fehler)')
    
    # Textbox für wissenschaftliche Metriken (Verhindert, dass Zahlen im Text der Doku untergehen)
    stats_text = f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.3f}"
    plt.gca().text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.title('Fehlerverteilung der Stauvorhersage (Residuals)', fontweight='bold')
    plt.xlabel('Abweichung (Reale Personenanzahl minus KI Vorhersage)')
    plt.ylabel('Häufigkeit')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ml_residuals.png'), dpi=300)
    plt.close()

def plot_actual_vs_predicted(df):
    """
    Goldstandard der Regressions-Auswertung: Ein Scatter/Hexbin-Plot.
    Da wir ein zufällig gesampeltes Testset haben, ist dies der perfekte Weg, 
    die Treffsicherheit zu beweisen, ohne Zeitverläufe (Time-Series) zu verfälschen.
    """
    print("5. Erstelle Actual vs. Predicted Korrelations-Plot...")
    plt.figure(figsize=(8, 8))
    
    # Hexbin ist für große Datensätze performanter und lesbarer als normale Scatterplots
    plt.hexbin(df['true_load_real'], df['pred_load_real'], gridsize=30, cmap='Blues', mincnt=1)
    
    # Die perfekte Diagonale einzeichnen (Target = Prediction)
    max_val = max(df['true_load_real'].max(), df['pred_load_real'].max())
    plt.plot([0, max_val], [0, max_val], color='#e74c3c', linestyle='--', linewidth=2, label='Perfekte Diagonale')
    
    plt.title('Korrelation: Realität vs. KI Vorhersage', fontweight='bold')
    plt.xlabel('Tatsächlicher Stau (Ground Truth in Personen)')
    plt.ylabel('Vorhersage der KI (in Personen)')
    plt.colorbar(label='Dichte (Anzahl der Messungen)')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ml_actual_vs_predicted.png'), dpi=300)
    plt.close()

def plot_business_confusion_matrix(df):
    """
    Wandelt die stufenlose Regression in harte Business-Kategorien um.
    Beantwortet dem Management die Frage: "Wenn echter Stau herrscht, 
    erkennt die KI das auch zuverlässig als Stau?"
    """
    print("6. Erstelle Business Confusion Matrix (Klassifikation)...")
    
    def categorize_traffic(val):
        if val <= 2.5: return "Frei (0-2)"
        elif val <= 5.5: return "Mittel (3-5)"
        else: return "Stau (>5)"
        
    y_true_class = df['true_load_real'].apply(categorize_traffic)
    y_pred_class = df['pred_load_real'].apply(categorize_traffic)
    
    labels = ["Frei (0-2)", "Mittel (3-5)", "Stau (>5)"]
    cm = confusion_matrix(y_true_class, y_pred_class, labels=labels)
    
    # Sicherung gegen Division durch Null (Falls eine Kategorie im Testset fehlt)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_perc = np.nan_to_num(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_perc, annot=True, fmt='.1f', cmap='Blues', xticklabels=labels, yticklabels=labels, 
                cbar_kws={'label': 'Erkennungsrate (%)'})
    
    plt.title('Business Matrix: Zuverlässigkeit der Stau-Erkennung', fontweight='bold')
    plt.xlabel('Vorhersage der KI')
    plt.ylabel('Tatsächlicher Zustand (Ground Truth)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ml_confusion_matrix.png'), dpi=300)
    plt.close()

def plot_error_by_hour(df):
    """
    Zeigt, zu welcher Tageszeit sich das Modell wie stark irrt. 
    Sehr wertvoll für den realen Enterprise-Betrieb (Sicherstellen, dass die 
    KI in der Rush-Hour nicht komplett zusammenbricht).
    """
    print("7. Erstelle Fehler-Analyse nach Tageszeit...")
    
    hourly_mae = df.groupby('hour')['abs_error'].mean().reset_index()
    
    plt.figure(figsize=(10, 5))
    sns.barplot(data=hourly_mae, x='hour', y='abs_error', palette='magma')
    
    plt.title('Modellgüte im Tagesverlauf (Wann irrt sich die KI am ehesten?)', fontweight='bold')
    plt.xlabel('Uhrzeit (Stunde)')
    plt.ylabel('Mittlerer Fehler (MAE in Personen)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ml_error_by_hour.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    print("\n🚀 STARTE ML-EVALUIERUNGS-PIPELINE (SÄULE A)")
    print("-" * 60)
    
    try:
        ensure_environment()
        df_results, xgboost_model, feature_list = load_and_predict()
        
        plot_feature_importance(xgboost_model, feature_list)
        plot_residuals(df_results)
        plot_actual_vs_predicted(df_results)
        plot_business_confusion_matrix(df_results)
        plot_error_by_hour(df_results)
        
        print("-" * 60)
        print(f"✅ PERFEKT! Alle 5 Grafiken wurden erfolgreich im Ordner '{OUTPUT_DIR}' gespeichert.")
    
    except Exception as e:
        print(f"\n❌ Fehler während der Auswertung: {e}")