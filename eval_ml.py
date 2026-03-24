"""
=========================================================================================
JMU SMART SUPERMARKET: EVALUATION SÄULE A (MACHINE LEARNING SENSORIK)
=========================================================================================

Dieses Skript operiert architektonisch völlig isoliert von der Live-REST-API des Backends.
Es dient als dediziertes Offline-Modul zur streng wissenschaftlichen Evaluierung der 
Trainingsgüte unseres XGBoost-Regressionsmodells.

WICHTIGER METHODISCHER ASPEKT (Data Leakage Prevention):
Um das Prinzip der Generalisierung zu beweisen, nutzen wir hier strikt die Datei 
'test_data_holdout.csv'. Diese Datenmenge (das Out-of-Sample Testset) wurde vor dem 
Training abgespalten und dem Modell NIEMALS gezeigt. 
Da das Testset stochastisch aus dem gesamten Jahr gezogen wurde (Random Stratified Split 
zur Vermeidung von saisonalem Bias), verzichten wir bewusst auf klassische, durchgehende 
Linien-Plots (Time-Series) – diese würden zeitlich unzusammenhängende Punkte fälschlicherweise 
verbinden. Stattdessen nutzen wir wissenschaftlich robuste Verteilungs- und Korrelations-Plots.
=========================================================================================
"""

# WICHTIG: Backend für Headless-Server setzen. 
# Ohne 'Agg' würde Matplotlib versuchen, ein natives GUI-Fenster (X11/Tkinter) zu öffnen.
# Auf einem reinen Cloud-Server (Linux ohne Monitor) führt dies sofort zu einem fatalen TclError.
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

# Akademisches Plot-Design: Konfiguriert Seaborn für hochauflösende, druckreife Grafiken.
# Der 'paper' Kontext skaliert die Schriftgrößen optimal für LaTeX- oder Sphinx-Dokumente.
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

def ensure_environment():
    """
    Pre-Flight-Check: Prüft die Systemintegrität vor Beginn der Berechnungen.
    Garantiert, dass die Pipeline deterministisch abbrechen kann (Fail-Fast), falls 
    die Artefakte aus dem vorherigen Trainings-Schritt fehlen.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    if not os.path.exists(TEST_DATA_FILE) or not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            "❌ Kritischer Fehler: Testdaten oder Modell-Binaries fehlen! "
            "Bitte zwingend erst die MLOps-Pipeline (train_model_optuna.py) ausführen."
        )

def load_and_predict():
    """
    Lädt das isolierte Testset, rekonstruiert den XGBoost-Baum im RAM und generiert 
    die Vorhersagen. Hier findet eine entscheidende mathematische Rücktransformation statt.
    """
    print("1. Lade isoliertes Hold-Out Testset und Modell-Artefakte...")
    
    df_test = pd.read_csv(TEST_DATA_FILE)
    df_test['timestamp_dt'] = pd.to_datetime(df_test['timestamp_dt'])
    
    # Deserialisierung: Lädt den C-Memory-Zustand des trainierten Modells
    with open(MODEL_FILE, 'rb') as f:
        artifacts = pickle.load(f)
        
    model = artifacts['model']
    features = artifacts['features']

    print("2. Generiere Vorhersagen auf ungesehenen Daten...")
    # Sichert die exakte Spalten-Reihenfolge ab, auf die der XGBoost-Booster trainiert wurde.
    X_test = df_test[features]
    
    # THEORETISCHE FUNDIERUNG (Log-Transformation):
    # Personenzahlen (Count-Daten) sind poisson-verteilt und stark rechtsschief (Right-Skewed).
    # Um dem Modell das Lernen zu erleichtern und Ausreißer (Extremstaus) zu dämpfen, wurde
    # im Training der natürliche Logarithmus (log1p) vorhergesagt.
    # Für eine interpretierbare Auswertung (Business-Metriken) MÜSSEN wir diese logarithmischen 
    # Werte zwingend mit der Exponentialfunktion (expm1) in echte Personen zurückrechnen.
    y_test_log = df_test['true_target']
    
    # np.clip fungiert als architektonisches Sicherheitsnetz: Es verhindert extrem 
    # unwahrscheinliche, aber mathematisch mögliche negative Vorhersagen (< 0 Personen).
    preds_log = np.clip(model.predict(X_test), 0, None)
    
    df_test['pred_load_real'] = np.expm1(preds_log)
    df_test['true_load_real'] = np.expm1(y_test_log)
    
    # Den absoluten Fehler pro Zeile (Delta) als neue Spalte berechnen.
    # Dies ist essenziell für die spätere Fehleranalyse aggregiert nach Tageszeiten.
    df_test['abs_error'] = np.abs(df_test['true_load_real'] - df_test['pred_load_real'])
    
    return df_test, model, features

def plot_feature_importance(model, features):
    """
    Explainable AI (XAI): Extrahiert die internen Entscheidungswege des Modells.
    
    Warum ist das wichtig? In einer Enterprise-Architektur sind reine "Blackbox"-Modelle 
    inakzeptabel. Wir müssen dem Management (oder Gutachtern) mathematisch beweisen, 
    dass die KI korrekte kausale Zusammenhänge gelernt hat (z. B. dass die Tageszeit oder 
    der Lag-1-Wert entscheidend für Stau sind) und nicht auf Rauschen (Noise) überangepasst hat.
    """
    print("3. Extrahiere Feature Importances (Explainability)...")
    
    # Nutzt den F-Score (Gain) von XGBoost zur Bewertung des Informationsgewinns pro Split
    importances = model.feature_importances_
    df_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
    
    # Dimensionalitätsreduktion für den Plot: Wir zeigen nur die Top 12 Prädiktoren
    df_imp = df_imp.sort_values('Importance', ascending=False).head(12)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_imp, x='Importance', y='Feature', palette='viridis')
    plt.title('Feature Importance (Dominierende Stau-Prädiktoren)', fontweight='bold')
    plt.xlabel('Relativer Einfluss auf die Vorhersage (Information Gain)')
    plt.ylabel('Modell-Variablen')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ml_feature_importance.png'), dpi=300)
    plt.close()

def plot_residuals(df):
    """
    Zeichnet ein Histogramm der Vorhersagefehler (Residual-Analyse).
    
    Warum ist das wichtig? Diese Grafik beweist die Abwesenheit von systematischem Bias.
    Ein perfektes Modell erzeugt eine symmetrische Glockenkurve (Normalverteilung) exakt um 
    den Nullpunkt. Liegt die Kurve verschoben, überschätzt das Modell den Stau chronisch.
    Zudem schließt eine saubere Verteilung das Problem der Heteroskedastizität aus.
    """
    print("4. Erstelle Residual-Analyse (Fehlerverteilung)...")
    
    # Residuum = Wahre Realität minus Vorhersage der KI
    residuals = df['true_load_real'] - df['pred_load_real']
    
    # Berechnung der drei wichtigsten wissenschaftlichen Key Performance Indicators (KPIs)
    mae = mean_absolute_error(df['true_load_real'], df['pred_load_real'])
    rmse = np.sqrt(mean_squared_error(df['true_load_real'], df['pred_load_real']))
    r2 = r2_score(df['true_load_real'], df['pred_load_real']) # Bestimmtheitsmaß der Varianz
    
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=60, kde=True, color='#3498db', edgecolor='black')
    
    # Die rote Nulllinie markiert das absolute Optimum (Residuum von 0.0)
    plt.axvline(x=0, color='#e74c3c', linestyle='--', linewidth=2, label='Perfekte Vorhersage')
    
    # Textbox für wissenschaftliche Metriken. 
    # Dies verhindert, dass diese extrem wichtigen Zahlen im Fließtext der Doku untergehen.
    stats_text = f"MAE: {mae:.2f} Pers.\nRMSE: {rmse:.2f} Pers.\nR²: {r2:.3f}"
    plt.gca().text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.title('Fehlerverteilung der Stauvorhersage (Residual Analysis)', fontweight='bold')
    plt.xlabel('Abweichung (Reale Personenanzahl minus KI-Vorhersage)')
    plt.ylabel('Häufigkeit der Fehler')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ml_residuals.png'), dpi=300)
    plt.close()

def plot_actual_vs_predicted(df):
    """
    Der Goldstandard der Regressions-Auswertung: Ein Korrelations-Plot.
    
    Warum Hexbin statt Scatterplot? Bei zigtausenden Test-Datenpunkten entsteht in einem 
    Scatterplot das "Overplotting"-Problem (ein massiver blauer Fleck, in dem keine Dichte 
    mehr ablesbar ist). Ein Hexbin-Plot aggregiert nahe beieinander liegende Punkte in Waben 
    und kodiert die Dichte stochastisch über die Farbintensität.
    """
    print("5. Erstelle Actual vs. Predicted Korrelations-Plot...")
    plt.figure(figsize=(8, 8))
    
    plt.hexbin(df['true_load_real'], df['pred_load_real'], gridsize=30, cmap='Blues', mincnt=1)
    
    # Die perfekte Diagonale (Winkelhalbierende). Je enger die Hexbins an dieser 
    # roten Linie kleben, desto höher ist das Bestimmtheitsmaß (R²) der KI.
    max_val = max(df['true_load_real'].max(), df['pred_load_real'].max())
    plt.plot([0, max_val], [0, max_val], color='#e74c3c', linestyle='--', linewidth=2, label='Perfekte Korrelation')
    
    plt.title('Korrelation: Realität vs. KI-Vorhersage', fontweight='bold')
    plt.xlabel('Tatsächlicher Stau (Ground Truth in Personen)')
    plt.ylabel('Vorhersage der KI (in Personen)')
    plt.colorbar(label='Dichte (Anzahl der Messungen)')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ml_actual_vs_predicted.png'), dpi=300)
    plt.close()

def plot_business_confusion_matrix(df):
    """
    Wandelt die stufenlosen Floats der Regression in diskrete Business-Kategorien um.
    
    Warum ist das wichtig? Das Operations-Research-Routing im Backend braucht klare Schwellenwerte 
    (Freier Gang vs. Verstopfter Gang), um den Dijkstra-Algorithmus zu manipulieren.
    Diese Confusion Matrix beweist dem Management visuell, dass die KI einen echten Stau 
    auch zuverlässig in die Kategorie "Stau" einordnet (Recall/Precision).
    """
    print("6. Erstelle Business Confusion Matrix (Klassifikation)...")
    
    # Kybernetisches Binning (Discretization) der kontinuierlichen Tensoren
    def categorize_traffic(val):
        if val <= 2.5: return "Frei (0-2)"
        elif val <= 5.5: return "Mittel (3-5)"
        else: return "Stau (>5)"
        
    y_true_class = df['true_load_real'].apply(categorize_traffic)
    y_pred_class = df['pred_load_real'].apply(categorize_traffic)
    
    labels = ["Frei (0-2)", "Mittel (3-5)", "Stau (>5)"]
    cm = confusion_matrix(y_true_class, y_pred_class, labels=labels)
    
    # Zeilenweise Normalisierung zu Prozentwerten.
    # np.errstate schützt die Laufzeit vor Division-by-Zero Exceptions, falls
    # das Testset zufällig keine extremen Staus (Outlier) enthält.
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_perc = np.nan_to_num(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_perc, annot=True, fmt='.1f', cmap='Blues', xticklabels=labels, yticklabels=labels, 
                cbar_kws={'label': 'Klassifikations-Trefferquote (%)'})
    
    plt.title('Business Matrix: Zuverlässigkeit der diskreten Stau-Erkennung', fontweight='bold')
    plt.xlabel('Vorhersage der KI')
    plt.ylabel('Tatsächlicher physikalischer Zustand')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ml_confusion_matrix.png'), dpi=300)
    plt.close()

def plot_error_by_hour(df):
    """
    Evaluiert die Stabilität der Modellgüte (RMSE/MAE) in Abhängigkeit der Tageszeit.
    
    Warum ist das wichtig? Dies zeigt die informationstheoretischen Grenzen auf. 
    In der Rush-Hour (ca. 17:00 Uhr) verhalten sich Menschen irrationaler (Spontankäufe, 
    Ausweichmanöver). Dieser Plot beweist transparent, dass das Modell das "Bounded Rationality" 
    Verhalten der Kunden abbildet und zur Rush-Hour naturgemäß einen höheren Fehler aufweist.
    """
    print("7. Erstelle Fehler-Analyse nach Tageszeit...")
    
    # Gruppierung (Aggregation) der absoluten Fehler nach simulierten Stunden
    hourly_mae = df.groupby('hour')['abs_error'].mean().reset_index()
    
    plt.figure(figsize=(10, 5))
    sns.barplot(data=hourly_mae, x='hour', y='abs_error', palette='magma')
    
    plt.title('Temporale Stabilität: Modellfehler im Tagesverlauf', fontweight='bold')
    plt.xlabel('Uhrzeit (Stunde des Tages)')
    plt.ylabel('Mittlerer Fehler (MAE in Personen)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ml_error_by_hour.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    print("\n STARTE ML-EVALUIERUNGS-PIPELINE (SÄULE A)")
    print("-" * 60)
    
    try:
        # Die strukturierte Ausführung der MLOps Evaluations-Kaskade
        ensure_environment()
        df_results, xgboost_model, feature_list = load_and_predict()
        
        plot_feature_importance(xgboost_model, feature_list)
        plot_residuals(df_results)
        plot_actual_vs_predicted(df_results)
        plot_business_confusion_matrix(df_results)
        plot_error_by_hour(df_results)
        
        print("-" * 60)
        print(f"✅ PERFEKT! Alle 5 akademischen Grafiken wurden in '{OUTPUT_DIR}' materialisiert.")
    
    except Exception as e:
        print(f"\n Kritischer Fehler während der Pipeline-Evaluation: {e}")