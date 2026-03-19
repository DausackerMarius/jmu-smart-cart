"""
=========================================================================================
JMU SMART SUPERMARKET: MLOps NLP-BENCHMARK PIPELINE (SÄULE C)
=========================================================================================
Quantitative Evaluierung der 'MLOpsEngine' (Hybrid aus Anchors & Logistic Regression).
Mißt Inference Latency (inkl. LRU-Cache Hit-Rates), Robustheit (Deep-Noise-Injection) 
und semantische Klassifikationsgüte.
=========================================================================================
"""

import os
import time
import random
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Zwingend erforderlich für Headless-Ausführung
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | [%(name)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("NLP-Audit")

# =============================================================================
# 1. SYSTEM-ADAPTER (Exakt gekoppelt an model.py)
# =============================================================================
try:
    from model import ml_predictor
    
    def get_category_prediction(product_name: str) -> str:
        try:
            # model.py liefert: (Kategorie, Knoten, Konfidenz)
            prediction_tuple = ml_predictor.predict(product_name)
            return str(prediction_tuple[0])
        except Exception as e:
            logger.error(f"Fehler bei der Vorhersage von '{product_name}': {e}")
            return "Sonstiges (Kasse)"
            
except ImportError:
    logger.critical("Konnte 'model.py' nicht importieren.")
    raise

# =============================================================================
# 2. BENCHMARK KONFIGURATION
# =============================================================================
OUTPUT_DIR = "eval_plots"

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
    def __init__(self):
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("paper", font_scale=1.1)

    def _inject_deep_noise(self, text: str) -> str:
        """
        Garantierte Noise-Injection, die den internen TextNormalizer der model.py überlebt!
        Wir nutzen echte Typos, keine simplen Whitespace-Removals.
        """
        if len(text) < 4:
            return text + "x" # Bei sehr kurzen Wörtern einfach einen Buchstabe anhängen
            
        r = random.random()
        
        # 33% Chance: Transposition (Buchstabendreher) -> Hraibo statt Haribo
        if r < 0.33: 
            idx = random.randint(1, len(text)-3)
            return text[:idx] + text[idx+1] + text[idx] + text[idx+2:]
            
        # 33% Chance: Deletion (Buchstabe fehlt) -> Harbo statt Haribo
        elif r < 0.66: 
            idx = random.randint(1, len(text)-2)
            return text[:idx] + text[idx+1:]
            
        # 34% Chance: Insertion (Zusätzlicher Buchstabe) -> Hariibo statt Haribo
        else:
            idx = random.randint(1, len(text)-1)
            random_char = random.choice('abcdefghijklmnopqrstuvwxyz')
            return text[:idx] + random_char + text[idx:]

    def generate_test_corpus(self, multiplier: int = 8) -> pd.DataFrame:
        logger.info("Generiere Deep-Noise Test-Corpus...")
        data = []
        for category, items in BASE_CATALOG.items():
            for item in items:
                # 1x Clean
                data.append({"text": item, "true_category": category, "is_noisy": False})
                # N-mal echten, garantierten Noise
                for _ in range(multiplier):
                    data.append({"text": self._inject_deep_noise(item), "true_category": category, "is_noisy": True})
        
        df = pd.DataFrame(data)
        return df.sample(frac=1, random_state=42).reset_index(drop=True)

    def run_benchmark(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Starte Inference-Benchmark ({len(df)} Samples)...")
        
        preds = []
        latencies = []
        
        # Wir messen hier bewusst INKLUSIVE dem LRU-Cache deiner model.py, 
        # um die reale Produktionslatenz (System Speed) abzubilden.
        for text in df['text']:
            t_start = time.perf_counter()
            pred = get_category_prediction(text)
            t_end = time.perf_counter()
            
            preds.append(pred)
            latencies.append((t_end - t_start) * 1000.0) 
            
        df['pred_category'] = preds
        df['latency_ms'] = latencies
        df['is_correct'] = df['true_category'] == df['pred_category']
        
        return df

    def create_reports(self, df: pd.DataFrame):
        logger.info("Generiere wissenschaftliche Audit-Visualisierungen...")
        
        # ---------------------------------------------------------
        # 1. LATENCY DISTRIBUTION (Inkl. Cache Analysis)
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 5))
        p95 = np.percentile(df['latency_ms'], 95)
        avg = df['latency_ms'].mean()
        
        sns.histplot(df['latency_ms'], bins=60, kde=True, color='#2c3e50')
        plt.axvline(p95, color='#e74c3c', linestyle='--', linewidth=2, label=f'P95 Latency: {p95:.3f} ms')
        plt.axvline(avg, color='#27ae60', linestyle='-', linewidth=2, label=f'Avg System Latency: {avg:.3f} ms')
        
        plt.title('Production System Latency (inkl. LRU-Cache Hit-Rates)', fontweight='bold')
        plt.xlabel('Verarbeitungszeit pro Item (Millisekunden)')
        plt.ylabel('Häufigkeit')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'nlp_latency_profile.png'), dpi=300)
        plt.close()

        # ---------------------------------------------------------
        # 2. ROBUSTNESS ANALYSIS (Deep Noise)
        # ---------------------------------------------------------
        plt.figure(figsize=(8, 6))
        acc_clean = df[~df['is_noisy']]['is_correct'].mean() * 100
        acc_noisy = df[df['is_noisy']]['is_correct'].mean() * 100
        
        ax = sns.barplot(x=['Katalog-Input (Clean)', 'User-Input (Deep Typos)'], 
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
        # Damit werden "Halluzinationen" oder Fallbacks ("Sonstiges (Kasse)") gnadenlos offengelegt.
        labels = sorted(list(set(df['true_category']).union(set(df['pred_category']))))
            
        cm = confusion_matrix(df['true_category'], df['pred_category'], labels=labels)
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': 'Anzahl Zuweisungen'})
        
        plt.title('End-to-End Confusion Matrix (inkl. Anchors & ML Output)', fontweight='bold', fontsize=14)
        plt.ylabel('Soll-Kategorie (Ground Truth)', fontweight='bold')
        plt.xlabel('Ist-Kategorie (KI / Anchor Output)', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'nlp_confusion_matrix.png'), dpi=300)
        plt.close()

        # ---------------------------------------------------------
        # 4. ERROR LOGGING
        # ---------------------------------------------------------
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
        print(f" -> Fallbacks ('Sonstiges'): {(df['pred_category'] == 'Sonstiges (Kasse)').sum()} Stück")
        print("="*65)
        print(f"✅ Audit abgeschlossen. Alle ungeschönten Grafiken in '{OUTPUT_DIR}' gespeichert.\n")


if __name__ == "__main__":
    try:
        evaluator = MLOpsEvaluator()
        test_corpus = evaluator.generate_test_corpus(multiplier=10)
        results_df = evaluator.run_benchmark(test_corpus)
        evaluator.create_reports(results_df)
    except Exception as e:
        logger.critical(f"Skript-Abbruch: {e}")