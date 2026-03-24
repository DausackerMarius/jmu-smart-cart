"""
=========================================================================================
JMU SMART SUPERMARKET: MLOps NLP-BENCHMARK PIPELINE (SÄULE C)
=========================================================================================
Dieses Skript fungiert als isoliertes, wissenschaftliches Audit-Tool für die 
Natural Language Processing (NLP) Architektur des Backends. 

Während das Modell in der Produktion reaktiv auf Kundenanfragen antwortet, 
unterzieht dieses Skript die 'MLOpsEngine' (den Hybrid aus Damerau-Levenshtein-Heuristik 
und Logistischer Regression) einem proaktiven, quantitativen Stresstest.

WICHTIGE WISSENSCHAFTLICHE METRIKEN, DIE HIER BEWIESEN WERDEN:
1. Inference Latency (P95): Beweist, dass das System unter Last nicht kollabiert und 
   die O(1) LRU-Cache-Mechanismen greifen.
2. Robustheit (Data Augmentation): Durch "Deep-Noise-Injection" simulieren wir das 
   reale "Fat-Finger-Syndrom" auf Touchscreens, um den Sim2Real-Gap zu evaluieren.
3. Semantische Klassifikationsgüte: Eine Confusion-Matrix legt systematische 
   Fehlzuordnungen (Bias) des Modells gnadenlos offen.
=========================================================================================
"""

import os
import time
import random
import logging
import pandas as pd
import numpy as np

# WICHTIG: Backend für Headless-Server setzen. 
# Ohne 'Agg' würde Matplotlib versuchen, ein natives GUI-Fenster (X11/Tkinter) zu öffnen.
# Auf einem reinen Cloud-Server (Linux ohne Monitor) führt dies sofort zu einem fatalen TclError.
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# --- LOGGING SETUP ---
# Ein professionelles MLOps-Logging-Format zur Nachvollziehbarkeit auf Server-Ebene
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | [%(name)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("NLP-Audit")

# =============================================================================
# 1. SYSTEM-ADAPTER (Schnittstelle zur produktiven Geschäftslogik)
# =============================================================================
try:
    # Wir importieren absichtlich direkt das produktive Modell aus der Backend-Architektur.
    # Dadurch testen wir exakt die Code-Pfade, die auch der Endkunde im Markt durchläuft.
    from model import ml_predictor
    
    def get_category_prediction(product_name: str) -> str:
        """
        Kapselt den Aufruf an die ML-Pipeline und implementiert Fallbacks.
        Ein Ausfall der KI darf niemals zum Absturz des Evaluators führen.
        """
        try:
            # model.py liefert standardmäßig ein Tuple: (Kategorie, Knoten-ID, Konfidenz)
            prediction_tuple = ml_predictor.predict(product_name)
            return str(prediction_tuple[0])
        except Exception as e:
            logger.error(f"Inferenz-Fehler bei der Vorhersage von '{product_name}': {e}")
            # Graceful Degradation: Im Fehlerfall wird der Kunde zur Kasse geroutet
            return "Sonstiges (Kasse)"
            
except ImportError:
    logger.critical("Konnte 'model.py' nicht importieren. System-Integrität verletzt.")
    raise

# =============================================================================
# 2. BENCHMARK KONFIGURATION & GROUND TRUTH
# =============================================================================
OUTPUT_DIR = "eval_plots"

# Die Ground-Truth: Ein repräsentativer Querschnitt der Supermarkt-Ontologie.
# Dient als saubere (clean) Basis für die spätere Noise-Injection.
BASE_CATALOG = {
    "Süßwaren & Snacks": ["Haribo Goldbären", "Milka Schokolade", "Snickers", "Kartoffelchips"],
    "Obst & Gemüse": ["Bananen", "Rispentomaten", "Äpfel", "Eisbergsalat"],
    "Kühlregal (Molkerei)": ["H-Milch 3,5%", "Naturjoghurt", "Butter", "Speisequark"],
    "Kühlregal (Vegan & Käse)": ["Gouda Jung", "Feta", "Veganer Tofu", "Camembert"],
    "Fleischtheke": ["Rinderhackfleisch", "Schweinesteak", "Hähnchenbrust"],
    "Fisch & Wurstwaren": ["Salami", "Kochschinken", "Lachsfilet", "Forelle"],
    "Drogerie & Haushalt": ["Toilettenpapier", "Zahnbürste", "Duschgel", "Waschmittel"],
    "Getränke Alkoholfrei": ["Coca Cola", "Mineralwasser", "Apfelsaft", "Orangenlimonade"],
    "Bier & Wein": ["Pilsner Bier", "Weißwein", "Sekt", "Hefeweizen"],
    "Tiefkühlware": ["Tiefkühlpizza", "Fischstäbchen", "TK Spinat", "Pommes Frites"],
    "Backwaren": ["Vollkornbrot", "Weizenbrötchen", "Buttercroissant", "Schokomuffin"],
    "Kaffee & Tee": ["Kaffeebohnen", "Kamillentee", "Espresso Pulver"],
    "Trockensortiment & Konserven": ["Spaghetti", "Basmatireis", "Tomatenmark", "Dosensuppe"],
    "Gewürze & Backzutaten": ["Weizenmehl", "Speisesalz", "Schwarzer Pfeffer", "Olivenöl"],
    "Fitness & Sport": ["Proteinriegel", "Whey Pulver", "Energy Drink"]
}

class MLOpsEvaluator:
    """
    Orchestriert die Pipeline zur Data Augmentation, Inferenz-Messung und Visualisierung.
    """
    def __init__(self):
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            
        # Akademisches Plot-Design für eine saubere Darstellung im Sphinx-HTML-Build
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("paper", font_scale=1.1)

    def _inject_deep_noise(self, text: str) -> str:
        """
        Data Augmentation via Deep Noise Injection.
        
        Warum tun wir das? Wenn wir das Modell nur auf sauber geschriebenen Wörtern testen,
        belügen wir uns selbst (Overfitting auf Laborbedingungen). Diese Funktion simuliert
        gezielt die motorischen Fehler (Fat-Finger-Syndrom) eines Menschen, der mit einem 
        Einkaufswagen läuft und dabei auf ein Tablet tippt. 
        Der Noise überlebt den internen RegEx-TextNormalizer des Modells.
        """
        # Edge-Case Protection: Verhindert Array-Index-Out-of-Bounds bei sehr kurzen Wörtern
        if len(text) < 4:
            return text + "x" 
            
        r = random.random()
        
        # 33% Chance: Transposition (Buchstabendreher) -> z.B. Hraibo statt Haribo
        # Dies ist exakt der Fehler-Typ, den die Damerau-Erweiterung im Backend abfangen soll.
        if r < 0.33: 
            idx = random.randint(1, len(text)-3)
            return text[:idx] + text[idx+1] + text[idx] + text[idx+2:]
            
        # 33% Chance: Deletion (Auslassung) -> z.B. Harbo statt Haribo
        # Testet die TF-IDF N-Gramm Stabilität (char_wb).
        elif r < 0.66: 
            idx = random.randint(1, len(text)-2)
            return text[:idx] + text[idx+1:]
            
        # 34% Chance: Insertion (Zusätzlicher Buchstabe) -> z.B. Hariibo statt Haribo
        else:
            idx = random.randint(1, len(text)-1)
            random_char = random.choice('abcdefghijklmnopqrstuvwxyz')
            return text[:idx] + random_char + text[idx:]

    def generate_test_corpus(self, multiplier: int = 8) -> pd.DataFrame:
        """
        Vervielfacht den kleinen Basis-Katalog durch synthetische Stochastik zu 
        einem repräsentativen, großen Test-Korpus.
        """
        logger.info("Generiere Deep-Noise Test-Corpus zur Robustheitsprüfung...")
        data = []
        for category, items in BASE_CATALOG.items():
            for item in items:
                # 1x Saubere Ground Truth (Baseline)
                data.append({"text": item, "true_category": category, "is_noisy": False})
                
                # N-fache Stochastische Noise-Injection
                for _ in range(multiplier):
                    data.append({"text": self._inject_deep_noise(item), "true_category": category, "is_noisy": True})
        
        df = pd.DataFrame(data)
        # Durchmischt den Datensatz (Shuffling), um Memory-Effekte zu eliminieren
        return df.sample(frac=1, random_state=42).reset_index(drop=True)

    def run_benchmark(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Führt die Inferenz auf dem Server aus und misst die exakte CPU-Latenz.
        """
        logger.info(f"Starte Inference-Benchmark ({len(df)} Samples)...")
        
        preds = []
        latencies = []
        
        # Wir messen hier bewusst INKLUSIVE dem LRU-Cache der Backend-Architektur.
        # Ein reiner ML-Benchmark würde die reale Produktions-Latenz (die der Kunde
        # auf dem Tablet spürt) künstlich schlechtreden.
        for text in df['text']:
            # time.perf_counter() ist die exakteste C-Uhr des Betriebssystems 
            # und nicht anfällig für NTP-Server-Synchronisationen.
            t_start = time.perf_counter()
            pred = get_category_prediction(text)
            t_end = time.perf_counter()
            
            preds.append(pred)
            latencies.append((t_end - t_start) * 1000.0) # Umrechnung in Millisekunden
            
        df['pred_category'] = preds
        df['latency_ms'] = latencies
        df['is_correct'] = df['true_category'] == df['pred_category']
        
        return df

    def create_reports(self, df: pd.DataFrame):
        """
        Generiert die wissenschaftlichen Visualisierungen für die Architekturdokumentation.
        """
        logger.info("Generiere wissenschaftliche Audit-Visualisierungen...")
        
        # ---------------------------------------------------------
        # 1. LATENCY DISTRIBUTION (Inklusive P95-Metrik)
        # ---------------------------------------------------------
        # Warum P95? Ein Durchschnittswert (Mean) ist bei Server-Latenzen oft trügerisch.
        # Das P95-Quantil beweist: "95% aller Kunden erhalten ihre Antwort schneller als X ms".
        # Das ist die wichtigste Metrik für das Management (Service Level Agreement).
        plt.figure(figsize=(10, 5))
        p95 = np.percentile(df['latency_ms'], 95)
        avg = df['latency_ms'].mean()
        
        sns.histplot(df['latency_ms'], bins=60, kde=True, color='#2c3e50')
        plt.axvline(p95, color='#e74c3c', linestyle='--', linewidth=2, label=f'P95 Latency: {p95:.3f} ms')
        plt.axvline(avg, color='#27ae60', linestyle='-', linewidth=2, label=f'Avg System Latency: {avg:.3f} ms')
        
        plt.title('Production System Latency (Beweis des LRU-Cache Load Balancings)', fontweight='bold')
        plt.xlabel('Verarbeitungszeit pro Item (Millisekunden)')
        plt.ylabel('Häufigkeit')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'nlp_latency_profile.png'), dpi=300)
        plt.close()

        # ---------------------------------------------------------
        # 2. ROBUSTNESS ANALYSIS (Clean vs. Deep Noise)
        # ---------------------------------------------------------
        # Beweist die Überlegenheit der TF-IDF char_wb Tokenisierung.
        # Zeigt, wie stark die Accuracy abfällt, wenn Menschen Tippfehler machen.
        plt.figure(figsize=(8, 6))
        acc_clean = df[~df['is_noisy']]['is_correct'].mean() * 100
        acc_noisy = df[df['is_noisy']]['is_correct'].mean() * 100
        
        ax = sns.barplot(x=['Katalog-Input (Sauber)', 'User-Input (Fat-Finger Typos)'], 
                         y=[acc_clean, acc_noisy], 
                         palette=['#3498db', '#e67e22'])
        
        plt.title('Fuzzy-Matching Audit: Klassifikationsgüte bei schweren Tippfehlern', fontweight='bold')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 105)
        
        for i, val in enumerate([acc_clean, acc_noisy]):
            ax.text(i, val + 2, f'{val:.1f}%', ha='center', fontweight='bold', fontsize=12)
            
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'nlp_robustness.png'), dpi=300)
        plt.close()

        # ---------------------------------------------------------
        # 3. CONFUSION MATRIX (Dynamic Label Resolution)
        # ---------------------------------------------------------
        # FIX: Wir holen uns dynamisch ALLE Kategorien, die in Ground Truth ODER Prediction auftauchen.
        # Damit werden "Halluzinationen" der KI oder Fallbacks gnadenlos offengelegt.
        labels = sorted(list(set(df['true_category']).union(set(df['pred_category']))))
            
        cm = confusion_matrix(df['true_category'], df['pred_category'], labels=labels)
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': 'Anzahl der Zuweisungen'})
        
        plt.title('End-to-End Confusion Matrix (inkl. Heuristiken & ML Output)', fontweight='bold', fontsize=14)
        plt.ylabel('Soll-Kategorie (Ground Truth)', fontweight='bold')
        plt.xlabel('Ist-Kategorie (KI / Heuristik Output)', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'nlp_confusion_matrix.png'), dpi=300)
        plt.close()

        # ---------------------------------------------------------
        # 4. ERROR LOGGING
        # ---------------------------------------------------------
        # Speichert alle Fehlklassifikationen als CSV zur manuellen Fehleranalyse (Active Learning Trigger).
        errors = df[~df['is_correct']].copy()
        if not errors.empty:
            errors.to_csv(os.path.join(OUTPUT_DIR, 'nlp_misclassifications.csv'), index=False)

        # ---------------------------------------------------------
        # TERMINAL REPORT
        # ---------------------------------------------------------
        print("\n" + "="*65)
        print("🏆 MLOps NLP AUDIT REPORT (SÄULE C)")
        print("="*65)
        print(f" -> Getestete Samples:     {len(df)}")
        print(f" -> System Accuracy:       {df['is_correct'].mean()*100:.2f}%")
        print(f" -> Clean Accuracy:        {acc_clean:.2f}%")
        print(f" -> Deep Noise Accuracy:   {acc_noisy:.2f}%")
        print(f" -> Avg System Latency:    {avg:.4f} ms")
        print(f" -> P95 System Latency:    {p95:.4f} ms")
        print(f" -> Fallbacks ('Sonstiges'): {(df['pred_category'] == 'Sonstiges (Kasse)').sum()} Stück")
        print("="*65)
        print(f"✅ Audit abgeschlossen. Alle ungeschönten Grafiken in '{OUTPUT_DIR}' gespeichert.\n")


if __name__ == "__main__":
    try:
        # Die strukturierte Ausführung der MLOps NLP Evaluations-Kaskade
        evaluator = MLOpsEvaluator()
        test_corpus = evaluator.generate_test_corpus(multiplier=10)
        results_df = evaluator.run_benchmark(test_corpus)
        evaluator.create_reports(results_df)
    except Exception as e:
        logger.critical(f"Kritischer Skript-Abbruch: {e}")