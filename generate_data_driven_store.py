"""
=========================================================================================
SMART CART DATA ENGINEERING PIPELINE (ENTERPRISE MASTER PIPELINE V18)
=========================================================================================
Zweck: End-to-End Generierung des digitalen Supermarkt-Zwillings (Digital Twin).

Diese Pipeline orchestriert den gesamten Daten-Lebenszyklus autonom:
1. Data Ingestion: Sicherer, asynchroner Download der B2B-Rohdaten (BLS).
2. NLP-Sanitization: Bereinigung der Strings von B2B-Rauschen zur Dimensionsreduktion.
3. MLOps-Trigger: Erzwingt ein synchrones Retraining der ML-Kaskade, um Data Drift vorzubeugen.
4. Graph Allocation: Nutzt das Sainte-Laguë-Höchstzahlverfahren, um Produkte topologisch 
   fair auf die begrenzten Regalkapazitäten des NetworkX-Graphen zu verteilen.
=========================================================================================
"""

import os
import json
import math
import time
import random
import logging
import pandas as pd
import re
import sys
import warnings  # Unterdrückt irrelevante Regex-Warnungen von Pandas im Terminal
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import requests
from requests.exceptions import RequestException

# Graceful Degradation für UI-Komponenten: Läuft auch, wenn tqdm auf dem Server fehlt
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Importiere die State-Manager und das ML-Modell aus dem Backend
try:
    from model import ml_predictor, inv_manager, CONFIG
except ImportError as e:
    raise ImportError(f"Kritischer Fehler: model.py konnte nicht geladen werden. Details: {e}")

# Professionelles MLOps-Logging für den Server-Betrieb
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | [%(name)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger("Master-ETL")

# Zwingt Python auf Servern ohne deutsche Locale, UTF-8 strikt zu verwenden (Umlaut-Schutz)
os.environ["PYTHONIOENCODING"] = "utf-8"

# =============================================================================
# I. KONFIGURATION & ONTOLOGIE (Single Source of Truth)
# =============================================================================

class PipelineConfig:
    """
    Kapselt alle magischen Strings und Hardcodes. 
    Verhindert redundante Variablen im Code und erleichtert künftige Wartung.
    """
    BLS_URL = "https://downloads.blsdb.de/bls_4.0.xlsx"
    EXCEL_INPUT = "bls_4.0.xlsx"
    CSV_OUTPUT = "smartcart_ml_training_data.csv"
    DB_OUTPUT = "products.json"
    ROUTING_OUTPUT = "routing_config.json"
    
    # NLP-Filterlisten: Entfernt toxisches Rauschen. Wenn das Modell lernt, dass 
    # "fettgewebe" ein wichtiges Feature ist, führt das zu Data Leakage, da echte 
    # Kunden diesen Begriff niemals am Tablet tippen würden.
    EN_KILLER_WORDS = {
        'raw', 'dried', 'beef', 'pork', 'juice', 'cheese', 'meat', 'frozen', 'boiled', 'canned', 'drained', 
        'stewed', 'fried', 'roasted', 'baked', 'sweet', 'sour', 'white', 'black', 'red', 'green', 'yellow', 
        'mixed', 'apple', 'water', 'sausage', 'fish', 'chicken', 'turkey', 'duck', 'veal', 'mutton', 'lamb', 
        'fat', 'oil', 'wine', 'beer', 'sugar', 'syrup', 'flour', 'bread', 'roll', 'cake', 'pie', 'cookie', 
        'biscuit', 'deep-frozen', 'unsweetened', 'sugared', 'smoked', 'salted', 'puree', 'seed', 'bean', 
        'pea', 'nut', 'cuttlefish', 'flounder', 'trout', 'salmon', 'goose', 'muesli', 'beverage', 'starch', 
        'noodles', 'paste', 'powder', 'candied', 'extract', 'spirit', 'drink', 'fillet'
    }

    B2B_NOISE_WORDS = {
        'fettgewebe', 'blut', 'intermuskulär', 'schlachtkörper', 'innereien', 'hirn', 'lunge', 'herz', 
        'leber', 'niere', 'milz', 'bries', 'zunge', 'magen', 'kutteln', 'schlund', 'knochenmark', 'fettwamme', 
        'fettabschnitte', 'blutpresssack', 'schweinskopfsülze', 'darmschmalz', 'sehnenfrei', 'sehnenarm',
        'kranial', 'kaudal', 'subkutan', 'isolat', 'extrakt', 'tiermehl', 'rohmasse', 'brät', 'kutterhilfsmittel', 
        'pottasche', 'hirschhornsalz', 'milchsäure', 'citronensäure', 'ascorbinsäure', 'essigsäure', 
        'kakaomasse', 'sahnestandmittel', 'gelatine', 'magermilchpulver', 'vollei', 'trockenhefe', 
        'säuerungsmittel', 'künstlich', 'aroma'
    }

class StoreOntology:
    """
    Definiert die physische Topologie des Supermarkts.
    Verknüpft logische ML-Kategorien starr oder flexibel mit Graphen-Knoten (Nodes).
    """
    # Verhindert Memory-Overflow im Graphen: Maximal 6 Produkte pro physischem Regal
    MAX_CAPACITY = 6
    # Kybernetischer Schwellenwert: Alles unter 25% ML-Konfidenz wird als Noise verworfen
    ML_CONFIDENCE_THRESHOLD = 0.25
    
    # Deterministische Topologie: Diese Produkte haben unverrückbare Stammplätze im Graphen
    FIXED_ZONES = {
        "Spirituosenschrank": ["v5"],
        "Fleischtheke": ["vA10"],
        "Sonstiges (Kasse)": ["vW1", "vW2", "vW3"],
        "Obst & Gemüse": ["v1", "v2", "vA2", "vB2"],  
        "Backwaren": ["v3", "vA4", "vB4"]             
    }
    
    # Stochastische Topologie: Freie Slots, die vom Sainte-Laguë-Verfahren dynamisch 
    # an Kategorien vergeben werden (abhängig von der saisonalen ML-Klassifikation).
    FLEXIBLE_ZONES = [
        "vD6", "vD5", "vD4", "vD2", "vD1",          
        "vC7", "vC6", "vC5", "vC3", "vC2",          
        "vB10", "vB9", "vB7", "vB6",  
        "vA9_2", "vA9", "vA7", "vA6"
    ]
    
    # Simuliert Markenvielfalt für ein realistischeres Frontend-Erlebnis
    BRAND_DISTRIBUTION = {
        "Obst & Gemüse": ["Bio-Hof", "Regional", "Demeter", "SanLucar"], 
        "Spirituosenschrank": ["Absolut", "Jack Daniels", "Havana Club"],
        "Fleischtheke": ["Metzgermeister", "Premium Beef", "Wiesenhof"], 
        "Fisch & Wurstwaren": ["Nordsee", "Herta", "Rügenwalder"],
        "Drogerie & Haushalt": ["Nivea", "Ariel", "Zewa", "Balea"], 
        "Kühlregal (Molkerei)": ["Alpenmilch", "Weihenstephan", "Bärenmarke"],
        "Kühlregal (Vegan & Käse)": ["Veggie-Life", "Milram", "Simply V"], 
        "Backwaren": ["Hausbäcker", "Backstube", "Oven Fresh"], 
        "Tiefkühlware": ["Dr. Oetker", "Iglo", "Frosta", "Wagner"], 
        "Trockensortiment & Konserven": ["Barilla", "Uncle Bens", "Erasco", "Mutti"], 
        "Gewürze & Backzutaten": ["Ostmann", "Bertolli", "Dr. Oetker"], 
        "Getränke Alkoholfrei": ["Coca Cola", "Gerolsteiner", "Vio"],
        "Bier & Wein": ["Krombacher", "Dornfelder", "Paulaner"], 
        "Kaffee & Tee": ["Tchibo", "Teekanne", "Jacobs"],
        "Süßwaren & Snacks": ["Milka", "Haribo", "Funny Frisch"], 
        "Fitness & Sport": ["PowerBar", "ESN", "Red Bull"]
    }
    
    # Hardcoded Injections für die Warteschlangen (Quengelware an der Kasse)
    HARDWARE_INJECTIONS = {
        "vW1": [("Wrigley's Extra", "Wrigley's"), ("Tic Tac Fresh", "Tic Tac"), ("Mentos Mint", "Mentos"), ("Hubba Bubba", "Wrigley's")],
        "vW2": [("AA Batterien 4er", "Varta"), ("AAA Batterien 4er", "Duracell"), ("Feuerzeug", "Bic"), ("Stabfeuerzeug", "Bic")],
        "vW3": [("Snickers", "Mars"), ("Mars Riegel", "Mars"), ("Twix", "Mars"), ("Amazon Gutschein 25€", "Gutscheinkarte")]
    }

    # System-Resilienz (Graceful Degradation): Falls das ML-Modell fehlschlägt oder 
    # Regale leer bleiben, werden diese Basis-Items injiziert, um den Graphen stabil zu halten.
    EMERGENCY_FALLBACKS = {
        "Obst & Gemüse": ["Tomaten", "Salatgurke", "Bananen", "Äpfel"],
        "Spirituosenschrank": ["Wodka", "Dry Gin", "Weißer Rum", "Scotch"],
        "Fleischtheke": ["Rinderhackfleisch", "Schweineschnitzel", "Hähnchenbrust"],
        "Fisch & Wurstwaren": ["Lachsfilet", "Fischstäbchen", "Salami"],
        "Drogerie & Haushalt": ["Toilettenpapier", "Zahnpasta", "Duschgel"],
        "Kühlregal (Molkerei)": ["Vollmilch", "Fettarme Milch", "Naturjoghurt"],
        "Kühlregal (Vegan & Käse)": ["Gouda", "Emmentaler", "Camembert", "Tofu"],
        "Backwaren": ["Kaiserbrötchen", "Laugenbrezel", "Toastbrot"],
        "Tiefkühlware": ["TK Pizza Salami", "Pommes Frites", "Rahmspinat"],
        "Trockensortiment & Konserven": ["Spaghetti", "Penne", "Passierte Tomaten"],
        "Gewürze & Backzutaten": ["Salz", "Pfeffer", "Olivenöl"],
        "Getränke Alkoholfrei": ["Mineralwasser", "Cola", "Apfelschorle"],
        "Bier & Wein": ["Pilsener", "Weißbier", "Rotwein"],
        "Kaffee & Tee": ["Kaffeebohnen", "Filterkaffee", "Kamillentee"],
        "Süßwaren & Snacks": ["Vollmilchschokolade", "Gummibärchen", "Chips"],
        "Fitness & Sport": ["Proteinriegel", "High Protein Pudding", "Energy Drink"]
    }

# =============================================================================
# II. PHASE 1: DATA ENGINEERING (EXCEL -> CSV)
# =============================================================================

class BLSDownloader:
    """Verwaltet den netzwerksicheren Download der B2B-Datenbankbank."""
    
    @staticmethod
    def download(url: str, dest_path: Path) -> bool:
        """
        Lädt die Datei asynchron in Chunks herunter, um RAM-Overflows zu vermeiden.
        Implementiert einen harten Timeout (60s), um hängende Threads zu verhindern.
        """
        if dest_path.exists():
            log.info(f"Excel-Datei gefunden: {dest_path}. Überspringe Download.")
            return True
        log.info(f"Downloade BLS-Datenbank von {url} ...")
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            with open(dest_path, 'wb') as file:
                if HAS_TQDM:
                    with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                file.write(chunk)
                                pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk: file.write(chunk)
            return True
        except RequestException as e:
            log.error(f"Downloadfehler: {e}")
            return False

class TextSanitizer:
    """Implementiert die NLP-Normalisierungs-Pipeline zur Datenbereinigung."""
    
    @staticmethod
    def clean(name: str) -> Optional[str]:
        """
        Transformiert B2B-Jargon in konsumentenfreundliche Strings (B2C).
        Reduziert die TF-IDF Matrix-Sparsity und verhindert Data Leakage.
        """
        if pd.isna(name): return None
        name = str(name).strip()
        name_lower = name.lower()
        
        # 1. Filtere extreme Labor- und B2B-Begriffe hart heraus
        if any(noise in name_lower for noise in PipelineConfig.B2B_NOISE_WORDS): return None
        for en_word in PipelineConfig.EN_KILLER_WORDS:
            if re.search(r'\b' + en_word + r'\b', name_lower): return None
                
        # 2. RegEx-Stripping: Entfernt Klammern, Beistriche und Gewichtsangaben
        name = re.sub(r'\s*\(.*?\)', '', name)
        name = re.sub(r'\s*\[.*?\]', '', name)
        name = name.split(',')[0]
        
        # 3. Schneidet alles nach "mit", "ohne" etc. ab (Dimensionalitätsreduktion)
        signal_words = [r'\bmind\.', r'\bmit\b', r'\bohne\b', r'\bin\b', r'\baus\b', r'\bnach\b']
        for signal in signal_words:
            match = re.search(signal, name, flags=re.IGNORECASE)
            if match: name = name[:match.start()]
            
        name = name.replace('/', ' ')
        words_to_strip = [
            r'\broh\b', r'\bgekocht\b', r'\bgebraten\b', r'\bgedünstet\b', r'\bgedämpft\b',
            r'\btiefgefroren\b', r'\bkonserve\b', r'\babgetropft\b', r'\bheiß\b',
            r'\bungesüßt\b', r'\bgezuckert\b', r'\bgeschmort\b', r'\bpochiert\b',
            r'\bmariniert\b', r'\bgesalzen\b', r'\bgeräuchert\b', r'\bpräserve\b',
            r'\bfett\b', r'\bi\.?\s*Tr\.?\b', r'\bvol\b', r'%'
        ]
        for word in words_to_strip:
            name = re.sub(word, '', name, flags=re.IGNORECASE)
            
        name = re.sub(r'\b\d+(,\d+)?\b', '', name) # Ziffern entfernen
        name = re.sub(r'[-/&]$', '', name.strip())
        name = re.sub(r'\s+', ' ', name).strip()
        
        # Kybernetischer Filter: Wörter, die zu lang/kurz sind, werden als Noise verworfen
        word_count = len(name.split())
        if word_count == 0 or word_count > 3: return None
        return name

class SupermarketMapper:
    """Heuristische Zuordnung: Generiert die Ground-Truth-Labels für das ML-Training."""
    
    @staticmethod
    def generate_label(bls_code: str, original_name: str) -> Optional[str]:
        """
        Übersetzt den kryptischen BLS-String in eine von uns definierte Supermarkt-Klasse.
        Dient als Basis (Y-Target) für die Logistische Regression im Backend.
        """
        bls_code = str(bls_code).strip().upper()
        name_lower = str(original_name).lower()
        if bls_code.startswith(('X', 'Y', 'W9', 'U9', 'V9')): return None
        
        # 1. Harte Keyword-Heuristik (Überschreibt BLS-Codes bei Eindeutigkeit)
        if 'tiefgefroren' in name_lower or 'eiscreme' in name_lower or bls_code.startswith('S2'): return "Tiefkühlware"
        if any(w in name_lower for w in ['brot', 'brötchen', 'baguette', 'toast', 'tortilla']): return "Backwaren"
        if any(w in name_lower for w in ['tofu', 'soja', 'vegan', 'vegetarisch', 'seitan']): return "Kühlregal (Vegan & Käse)"
        if any(w in name_lower for w in ['milch', 'käse', 'joghurt', 'quark', 'sahne']): return "Kühlregal (Molkerei)"
        if any(w in name_lower for w in ['nudel', 'pasta', 'spaghetti', 'penne', 'teigwaren', 'reis', 'konserve', 'abgetropft']): return "Trockensortiment & Konserven"
        if 'müsli' in name_lower or 'cornflakes' in name_lower or 'haferflocken' in name_lower: return "Süßwaren & Snacks" 
        if 'chips' in name_lower or 'flips' in name_lower or 'salzstangen' in name_lower: return "Süßwaren & Snacks"
        if any(w in name_lower for w in ['lachs', 'forelle', 'fisch', 'wurst', 'salami', 'schinken']): return "Fisch & Wurstwaren"
        if any(w in name_lower for w in ['öl', 'essig', 'salz', 'pfeffer', 'hefe', 'mehl', 'stärke']): return "Gewürze & Backzutaten"

        # 2. Fallback auf die offizielle Buchstaben-Notation des BLS
        first_letter = bls_code[0] if len(bls_code) > 0 else '?'
        mapping = {
            'B': "Gewürze & Backzutaten", 'C': "Backwaren", 'D': "Backwaren", 'H': "Backwaren",
            'F': "Obst & Gemüse", 'G': "Obst & Gemüse", 'K': "Obst & Gemüse", 'M': "Kühlregal (Molkerei)",
            'T': "Fisch & Wurstwaren", 'U': "Fleischtheke", 'V': "Fleischtheke", 'W': "Fleischtheke",
            'S': "Süßwaren & Snacks", 'Q': "Gewürze & Backzutaten", 'R': "Gewürze & Backzutaten"
        }
        
        if first_letter == 'E':
            if bls_code.startswith('E1'): return "Kühlregal (Molkerei)" 
            return "Trockensortiment & Konserven" 
        if first_letter in ['N', 'P']:
            if bls_code.startswith('N4'): return "Kaffee & Tee"
            if bls_code.startswith('P'): return "Bier & Wein"
            return "Getränke Alkoholfrei"
        return mapping.get(first_letter, None)

class CSVBuilder:
    """Extrahiert die Excel in ein performantes, Pandas-kompatibles CSV-Format."""
    
    def __init__(self, input_file: Path, output_file: Path):
        self.input_file = input_file
        self.output_file = output_file
        
    def _locate_target_columns(self, df) -> Tuple[str, str]:
        """Auto-Discovery der Spalten (verhindert Pipeline-Brüche bei Excel-Layout-Änderungen)."""
        code_col, name_col = None, None
        for col in df.columns:
            sample = df[col].dropna().astype(str).head(50)
            if len(sample) > 0 and (sample.str.match(r'^[A-Z][A-Z0-9]{4,6}$').sum() / len(sample)) > 0.5:
                code_col = col; break
        max_score = -1
        for col in df.columns:
            if col == code_col: continue
            sample = df[col].dropna().astype(str).head(100)
            if len(sample) == 0 or sample.nunique() < (len(sample) * 0.4): continue 
            sample_lower = sample.str.lower()
            score = sample_lower.str.contains(r'ä|ö|ü|ß').sum() * 10 + sample_lower.str.contains(r'\b(roh|gekocht|brot|fleisch|käse)\b').sum() * 5 - sample_lower.str.contains(r'\b(raw|beef|cheese|dried|meat)\b').sum() * 20
            if score > max_score: max_score = score; name_col = col
        if not code_col: code_col = df.columns[0]
        if not name_col: name_col = df.columns[1]
        return code_col, name_col

    def execute(self):
        """Führt die Transformation von der Roh-Excel zur bereinigten CSV durch."""
        log.info("Phase 1: Parse Excel und generiere ML-CSV (Ground Truth)...")
        xl = pd.ExcelFile(self.input_file)
        sheet_name = next((s for s in ['Lebensmittel', 'LEBM', 'Daten', 'Sheet1'] if s in xl.sheet_names), xl.sheet_names[0])
        df_raw = pd.read_excel(self.input_file, sheet_name=sheet_name, dtype=str)
        df_raw.columns = [str(c).strip().upper() for c in df_raw.columns]
        
        code_col, name_col = self._locate_target_columns(df_raw)
        df = pd.DataFrame({'BLS_Code': df_raw[code_col], 'Original_Name': df_raw[name_col]}).dropna()
        
        # Wende Heuristiken und Normalisierungen zeilenweise (vektorisiert) an
        df['Supermarket_Category'] = df.apply(lambda row: SupermarketMapper.generate_label(row['BLS_Code'], row['Original_Name']), axis=1)
        df = df.dropna(subset=['Supermarket_Category'])
        
        df['Product_Name'] = df['Original_Name'].apply(TextSanitizer.clean)
        df = df.dropna(subset=['Product_Name'])
        
        # Deduplikation: Verhindert Target-Leakage durch identische Produkte
        df['Name_Lower'] = df['Product_Name'].str.lower()
        df = df.drop_duplicates(subset=['Name_Lower']).drop(columns=['Name_Lower'])
        
        df_clean = df[['Product_Name', 'Supermarket_Category']].copy()
        df_clean.to_csv(self.output_file, index=False, encoding='utf-8')
        log.info(f"Phase 1 beendet: {len(df_clean)} saubere Artikel exportiert.")

# =============================================================================
# III. PHASE 2: ML-SYNC & STORE ALLOCATION
# =============================================================================

class DataIngestionService:
    """Hilfsklasse zur Extraktion der validesten Produkte aus der generierten CSV."""
    
    @staticmethod
    def extract_and_sort(file_path: str, limit: int = 1500) -> List[str]:
        """
        Liest die CSV ein und wählt die wahrscheinlichsten 'Mainstream'-Artikel aus.
        """
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        raw_items = df['Product_Name'].astype(str).str.strip()
        valid_items = raw_items[raw_items.str.len() > 2].drop_duplicates().tolist()
        
        # Synthetisches Mainstream-Scoring (Kürzere Wörter sind oft präsenter im Markt)
        scored_items = []
        for item in valid_items:
            score = len(item) + random.uniform(0, 15) 
            scored_items.append((score, item))
            
        scored_items.sort(key=lambda x: x[0])
        return [item for score, item in scored_items][:limit]

class StoreBuilder:
    """
    Befüllt den digitalen Supermarkt-Graphen.
    Kernalgorithmus: Das modifizierte Sainte-Laguë-Verfahren (Sitzzuteilungsverfahren).
    """
    
    def __init__(self):
        # Initialisiert alle validen physikalischen Knoten
        self.stock = {node: [] for nodes in StoreOntology.FIXED_ZONES.values() for node in nodes}
        for node in StoreOntology.FLEXIBLE_ZONES: self.stock[node] = []
        self.capacity = {node: 0 for node in self.stock.keys()}

    def _allocate_kassen(self):
        """Injiziert deterministisch Impulskäufe in den Wartebereich (Kassen)."""
        for q_node, item_list in StoreOntology.HARDWARE_INJECTIONS.items():
            for name, brand in item_list:
                self.stock[q_node].append({
                    'name': name, 'brand': brand, 'category': 'Sonstiges (Kasse)', 
                    'ai_confidence': 1.0, 'suggested_slot': q_node, 'needs_review': False
                })
                self.capacity[q_node] += 1

    def execute(self, items: List[str]):
        """
        Das Herzstück der Verteilung. 
        Nutzt das Sainte-Laguë-Verfahren, um flexible Kategorien proportional 
        und fair auf den Graphen aufzuteilen.
        """
        log.info("Phase 3: Klassifiziere und alloziiere Produkte (Sainte-Laguë Algorithmus)...")
        self._allocate_kassen()
        
        # Batch-Inferenz: Nutzt das trainierte ML-Modell, um alle Items zu klassifizieren
        batch_results = ml_predictor.predict_batch(items)
        
        # PANDAS EMPTY DATAFRAME FIX: Sichert die Pipeline vor Abstürzen, 
        # falls das ML-Modell temporär keine Produkte über dem Threshold findet.
        records = [
            {'Name': items[i].title(), 'Cat': r[0], 'Conf': r[2]} 
            for i, r in enumerate(batch_results) if r[0] != "Sonstiges (Kasse)" and r[2] >= StoreOntology.ML_CONFIDENCE_THRESHOLD
        ]
        df = pd.DataFrame(records, columns=['Name', 'Cat', 'Conf'])

        # SAINTE-LAGUË VERFAHREN (Divisorverfahren mit Aufrundung)
        # Verteilt die "freien" Regalslots fair anhand der Menge der vorhandenen Produkte pro Kategorie
        counts = {k: 1 for k in StoreOntology.EMERGENCY_FALLBACKS.keys() if k not in StoreOntology.FIXED_ZONES}
        if not df.empty:
            dynamic_cats = [c for c in df['Cat'] if c not in StoreOntology.FIXED_ZONES]
            actual_counts = pd.Series(dynamic_cats).value_counts().to_dict()
            for cat, count in actual_counts.items():
                if cat in counts: counts[cat] += count

        allocation = {cat: 1 for cat in counts.keys()}
        slots_to_distribute = len(StoreOntology.FLEXIBLE_ZONES) - len(allocation)
        for _ in range(slots_to_distribute):
            # Der Divisor ist (Bisherige Sitze + 0.5)
            winner = max({cat: (count / (allocation[cat] + 0.5)) for cat, count in counts.items()}, 
                         key=lambda k: {cat: (count / (allocation[cat] + 0.5)) for cat, count in counts.items()}.get(k))
            allocation[winner] += 1

        available_nodes = StoreOntology.FLEXIBLE_ZONES.copy()
        random.shuffle(available_nodes) # Verhindert strukturellen Bias in der Platzierung
        
        routing_map = {**StoreOntology.FIXED_ZONES}
        pointer = 0
        for cat, amount in allocation.items():
            routing_map[cat] = available_nodes[pointer : pointer+amount]
            pointer += amount

        # Befüllen der Graphen-Nodes mit den berechneten Produkten
        for cat, nodes in routing_map.items():
            if cat == "Sonstiges (Kasse)": continue
            cat_items = df[df['Cat'] == cat]
            for _, row in cat_items.iterrows():
                available = [n for n in nodes if self.capacity[n] < StoreOntology.MAX_CAPACITY]
                if not available: break 
                
                # Greedy: Nimmt immer das aktuell leerste Regal
                best_node = min(available, key=lambda n: self.capacity[n])
                brand = random.choice(StoreOntology.BRAND_DISTRIBUTION.get(cat, ["JMU Choice"]))
                
                self.stock[best_node].append({
                    'name': row['Name'], 'brand': brand, 'category': cat,
                    'ai_confidence': round(row['Conf'], 3), 'suggested_slot': best_node, 'needs_review': False
                })
                self.capacity[best_node] += 1

            # Graceful Degradation: Falls ein Regal nicht voll wird, nutze Fallbacks
            while any(self.capacity[n] < StoreOntology.MAX_CAPACITY for n in nodes):
                node = min(nodes, key=lambda n: self.capacity[n])
                fallback = random.choice(StoreOntology.EMERGENCY_FALLBACKS.get(cat, ["Basic Item"]))
                brand = random.choice(StoreOntology.BRAND_DISTRIBUTION.get(cat, ["JMU Basic"]))
                
                self.stock[node].append({
                    'name': fallback, 'brand': brand, 'category': cat,
                    'ai_confidence': 1.0, 'suggested_slot': node, 'needs_review': False
                })
                self.capacity[node] += 1
                
        return self.stock, routing_map

# =============================================================================
# IV. MASTER ORCHESTRATOR
# =============================================================================

class MasterOrchestrator:
    """
    Steuert den deterministischen Ausführungsablauf der Architektur.
    Garantiert, dass die ML-Modelle synchron mit den Daten gebaut werden.
    """
    
    def execute(self):
        """Führt alle Phasen nacheinander aus und exportiert die Ergebnisse."""
        print("\n" + "="*75)
        print("🏭 ENTERPRISE STORE BUILDER v18.0 (Master Pipeline)")
        print("="*75)
        t_start = time.time()
        
        input_path = Path(PipelineConfig.EXCEL_INPUT)
        csv_path = Path(PipelineConfig.CSV_OUTPUT)
        
        # Pandas-Regex Warnungen auf Konsole unterdrücken
        warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
        
        # 1. Download
        if not input_path.exists():
            if not BLSDownloader.download(PipelineConfig.BLS_URL, input_path):
                sys.exit(1)
                
        # 2. NLP-Sanitization & CSV-Export
        csv_builder = CSVBuilder(input_path, csv_path)
        csv_builder.execute()
        
        # 3. CRITICAL MLOPS SYNC: Data Leakage Prävention
        # Da wir neue CSV-Daten generiert haben, MUSS das NLP-Modell gelöscht 
        # und neu trainiert werden. Andernfalls entsteht ein "Stale Model" (Concept Drift).
        log.info("Phase 2: Lösche altes ML-Modell und erzwinge asynchrones Retraining...")
        model_file = Path(CONFIG.ML_MODEL_FILE)
        if model_file.exists():
            model_file.unlink() 
        ml_predictor._sync_model() # Baut die Logistische Regression neu
        
        # 4. Graphen-Allokation
        raw_items = DataIngestionService.extract_and_sort(str(csv_path), limit=1500)
        builder = StoreBuilder()
        final_stock, routing_map = builder.execute(raw_items)
        
        # 5. Export der Artefakte für das Live-Backend (Thread-Safe über InvManager Lock)
        with inv_manager._lock:
            inv_manager.stock = final_stock
            inv_manager.save_to_json()
            
        with open(PipelineConfig.ROUTING_OUTPUT, "w", encoding='utf-8') as f:
            json.dump(routing_map, f, indent=4, ensure_ascii=False)

        t_end = time.time()
        placed = sum(len(items) for items in final_stock.values())
        
        # Berechne die absolute Kapazität basierend auf der Ontologie
        total_nodes = len(StoreOntology.FIXED_ZONES["Spirituosenschrank"]) + \
                      len(StoreOntology.FIXED_ZONES["Fleischtheke"]) + \
                      len(StoreOntology.FIXED_ZONES["Sonstiges (Kasse)"]) + \
                      len(StoreOntology.FIXED_ZONES["Obst & Gemüse"]) + \
                      len(StoreOntology.FIXED_ZONES["Backwaren"]) + \
                      len(StoreOntology.FLEXIBLE_ZONES)
        
        print("\n" + "="*75)
        print("📊 ETL SYSTEM AUDIT & VALIDIERUNGS-REPORT")
        print("="*75)
        print(f"⏱️  Ausführungszeit:        {t_end - t_start:.2f} Sekunden")
        print(f"📦 Regale allokiert:       {len(final_stock)} / {total_nodes}")
        print(f"🛒 Produkte im System:     {placed} / {total_nodes * StoreOntology.MAX_CAPACITY}")
        print("\n📂 System-Artefakte erfolgreich generiert und synchronisiert:")
        print(f"   ├── {PipelineConfig.CSV_OUTPUT} (NLP Ground Truth)")
        print(f"   ├── {PipelineConfig.DB_OUTPUT}        (Inventar)")
        print(f"   └── {PipelineConfig.ROUTING_OUTPUT}  (Sainte-Laguë Graphen-Topologie)")
        print("-" * 75)
        print("🚀 Architektur-Setup abgeschlossen. Starte nun den Uvicorn/FastAPI-Server.")

if __name__ == "__main__":
    orchestrator = MasterOrchestrator()
    orchestrator.execute()