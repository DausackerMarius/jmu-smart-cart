"""
Smart Cart Data Engineering Pipeline (Enterprise Master Pipeline V18)
-------------------------------------------------------------------------
Zweck: End-to-End Generierung des Supermarktes. 
1. BLS Excel Download & NLP-Transformation (CSV Generierung)
2. Retraining der Logistic Regression (Zwingender model.py Sync)
3. Mainstream-Item Extraction & ML-Inference
4. Sainte-Laguë Graph-Allocation auf exakt 30 validen schwarzen Regalknoten
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
import warnings  # FIX: Modul für Warnungs-Unterdrückung hinzugefügt
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import requests
from requests.exceptions import RequestException

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Importiere die Backends aus model.py
try:
    from model import ml_predictor, inv_manager, CONFIG
except ImportError as e:
    raise ImportError(f"Kritischer Fehler: model.py konnte nicht geladen werden. Details: {e}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | [%(name)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger("Master-ETL")
os.environ["PYTHONIOENCODING"] = "utf-8"

# =============================================================================
# I. KONFIGURATION & ONTOLOGY
# =============================================================================

class PipelineConfig:
    """Konstanten und Konfigurationen für die ETL-Pipeline und das Daten-Cleaning."""
    BLS_URL = "https://downloads.blsdb.de/bls_4.0.xlsx"
    EXCEL_INPUT = "bls_4.0.xlsx"
    CSV_OUTPUT = "smartcart_ml_training_data.csv"
    DB_OUTPUT = "products.json"
    ROUTING_OUTPUT = "routing_config.json"
    
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
    Definiert die physische Struktur des Supermarkts.
    Verknüpft Produktkategorien mit den exakten Knoten (Nodes) im NetworkX-Graphen.
    """
    MAX_CAPACITY = 6
    ML_CONFIDENCE_THRESHOLD = 0.25
    
    # Exakt 12 valide, feste Knoten. (Nur schwarze/blaue Regal-Knoten!)
    FIXED_ZONES = {
        "Spirituosenschrank": ["v5"],
        "Fleischtheke": ["vA10"],
        "Sonstiges (Kasse)": ["vW1", "vW2", "vW3"],
        "Obst & Gemüse": ["v1", "v2", "vA2", "vB2"],  
        "Backwaren": ["v3", "vA4", "vB4"]             
    }
    
    # Exakt 18 verbleibende schwarze Regal-Slots (Keine roten Gang-Knoten!)
    FLEXIBLE_ZONES = [
        "vD6", "vD5", "vD4", "vD2", "vD1",          
        "vC7", "vC6", "vC5", "vC3", "vC2",          
        "vB10", "vB9", "vB7", "vB6",  
        "vA9_2", "vA9", "vA7", "vA6"
    ]
    
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
    
    HARDWARE_INJECTIONS = {
        "vW1": [("Wrigley's Extra", "Wrigley's"), ("Tic Tac Fresh", "Tic Tac"), ("Mentos Mint", "Mentos"), ("Hubba Bubba", "Wrigley's")],
        "vW2": [("AA Batterien 4er", "Varta"), ("AAA Batterien 4er", "Duracell"), ("Feuerzeug", "Bic"), ("Stabfeuerzeug", "Bic")],
        "vW3": [("Snickers", "Mars"), ("Mars Riegel", "Mars"), ("Twix", "Mars"), ("Amazon Gutschein 25€", "Gutscheinkarte")]
    }

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
    """Verwaltet den sicheren Download der externen BLS-Datenbankbank."""
    
    @staticmethod
    def download(url: str, dest_path: Path) -> bool:
        """
        Lädt die Excel-Datei herunter, sofern sie noch nicht lokal existiert.

        Args:
            url (str): Die Quell-URL der Datei.
            dest_path (Path): Der lokale Zielpfad.

        Returns:
            bool: True bei Erfolg, False bei einem Netzwerkfehler.
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
    """Klasse für Natural Language Processing (NLP) und Datenbereinigung."""
    
    @staticmethod
    def clean(name: str) -> Optional[str]:
        """
        Bereinigt rohe Produktnamen mittels RegEx und Filterlisten.
        Entfernt B2B-Begriffe, englische Wörter und unnötige Zusatzinformationen.

        Args:
            name (str): Der rohe Produktname aus der Datenbank.

        Returns:
            Optional[str]: Der bereinigte String oder None, falls das Produkt unbrauchbar ist.
        """
        if pd.isna(name): return None
        name = str(name).strip()
        name_lower = name.lower()
        
        if any(noise in name_lower for noise in PipelineConfig.B2B_NOISE_WORDS): return None
        for en_word in PipelineConfig.EN_KILLER_WORDS:
            if re.search(r'\b' + en_word + r'\b', name_lower): return None
                
        name = re.sub(r'\s*\(.*?\)', '', name)
        name = re.sub(r'\s*\[.*?\]', '', name)
        name = name.split(',')[0]
        
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
            
        name = re.sub(r'\b\d+(,\d+)?\b', '', name)
        name = re.sub(r'[-/&]$', '', name.strip())
        name = re.sub(r'\s+', ' ', name).strip()
        
        word_count = len(name.split())
        if word_count == 0 or word_count > 3: return None
        return name

class SupermarketMapper:
    """Ordnet BLS-Codes logischen Supermarktkategorien zu."""
    
    @staticmethod
    def generate_label(bls_code: str, original_name: str) -> Optional[str]:
        """
        Weist einem Produkt basierend auf seinem BLS-Code und Namen eine Abteilung zu.
        
        Args:
            bls_code (str): Der alphanumerische Code aus der BLS-Datenbank.
            original_name (str): Der Name des Produkts.

        Returns:
            Optional[str]: Die Supermarktkategorie (z.B. "Tiefkühlware") oder None.
        """
        bls_code = str(bls_code).strip().upper()
        name_lower = str(original_name).lower()
        if bls_code.startswith(('X', 'Y', 'W9', 'U9', 'V9')): return None
        
        if 'tiefgefroren' in name_lower or 'eiscreme' in name_lower or bls_code.startswith('S2'): return "Tiefkühlware"
        if any(w in name_lower for w in ['brot', 'brötchen', 'baguette', 'toast', 'tortilla']): return "Backwaren"
        if any(w in name_lower for w in ['tofu', 'soja', 'vegan', 'vegetarisch', 'seitan']): return "Kühlregal (Vegan & Käse)"
        if any(w in name_lower for w in ['milch', 'käse', 'joghurt', 'quark', 'sahne']): return "Kühlregal (Molkerei)"
        if any(w in name_lower for w in ['nudel', 'pasta', 'spaghetti', 'penne', 'teigwaren', 'reis', 'konserve', 'abgetropft']): return "Trockensortiment & Konserven"
        if 'müsli' in name_lower or 'cornflakes' in name_lower or 'haferflocken' in name_lower: return "Süßwaren & Snacks" 
        if 'chips' in name_lower or 'flips' in name_lower or 'salzstangen' in name_lower: return "Süßwaren & Snacks"
        if any(w in name_lower for w in ['lachs', 'forelle', 'fisch', 'wurst', 'salami', 'schinken']): return "Fisch & Wurstwaren"
        if any(w in name_lower for w in ['öl', 'essig', 'salz', 'pfeffer', 'hefe', 'mehl', 'stärke']): return "Gewürze & Backzutaten"

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
    """Verantwortlich für die Extraktion und Speicherung der CSV-Datenbank."""
    
    def __init__(self, input_file: Path, output_file: Path):
        self.input_file = input_file
        self.output_file = output_file
        
    def _locate_target_columns(self, df) -> Tuple[str, str]:
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
        log.info("Phase 1: Parse Excel und generiere ML-CSV...")
        xl = pd.ExcelFile(self.input_file)
        sheet_name = next((s for s in ['Lebensmittel', 'LEBM', 'Daten', 'Sheet1'] if s in xl.sheet_names), xl.sheet_names[0])
        df_raw = pd.read_excel(self.input_file, sheet_name=sheet_name, dtype=str)
        df_raw.columns = [str(c).strip().upper() for c in df_raw.columns]
        
        code_col, name_col = self._locate_target_columns(df_raw)
        df = pd.DataFrame({'BLS_Code': df_raw[code_col], 'Original_Name': df_raw[name_col]}).dropna()
        
        df['Supermarket_Category'] = df.apply(lambda row: SupermarketMapper.generate_label(row['BLS_Code'], row['Original_Name']), axis=1)
        df = df.dropna(subset=['Supermarket_Category'])
        
        df['Product_Name'] = df['Original_Name'].apply(TextSanitizer.clean)
        df = df.dropna(subset=['Product_Name'])
        
        df['Name_Lower'] = df['Product_Name'].str.lower()
        df = df.drop_duplicates(subset=['Name_Lower']).drop(columns=['Name_Lower'])
        
        df_clean = df[['Product_Name', 'Supermarket_Category']].copy()
        df_clean.to_csv(self.output_file, index=False, encoding='utf-8')
        log.info(f"Phase 1 beendet: {len(df_clean)} Artikel in {self.output_file} geschrieben.")

# =============================================================================
# III. PHASE 2: ML-SYNC & STORE ALLOCATION
# =============================================================================

class DataIngestionService:
    """Hilfsklasse zur Extraktion der validesten Produkte aus der generierten CSV."""
    
    @staticmethod
    def extract_and_sort(file_path: str, limit: int = 1500) -> List[str]:
        """
        Liest die CSV ein und wählt die wahrscheinlichsten 'Mainstream'-Artikel aus.
        
        Args:
            file_path (str): Pfad zur CSV.
            limit (int): Maximale Anzahl an zu extrahierenden Produkten.

        Returns:
            List[str]: Liste der Produktnamen.
        """
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        raw_items = df['Product_Name'].astype(str).str.strip()
        valid_items = raw_items[raw_items.str.len() > 2].drop_duplicates().tolist()
        
        # Mainstream-Scoring
        scored_items = []
        for item in valid_items:
            score = len(item) + random.uniform(0, 15) 
            scored_items.append((score, item))
            
        scored_items.sort(key=lambda x: x[0])
        return [item for score, item in scored_items][:limit]

class PricingEngine:
    """Generiert realistische Preise auf Basis von statistischen Verteilungen."""
    _price_distributions = {
        "Fleischtheke": {"mu": 8.50, "sigma": 3.00, "min": 2.99}, "Spirituosenschrank": {"mu": 15.00, "sigma": 5.00, "min": 7.99},
        "Fisch & Wurstwaren": {"mu": 4.50, "sigma": 1.50, "min": 1.49}, "Obst & Gemüse": {"mu": 2.50, "sigma": 1.00, "min": 0.49},
        "Kühlregal (Vegan & Käse)": {"mu": 3.50, "sigma": 1.20, "min": 1.29}, "Bier & Wein": {"mu": 5.00, "sigma": 2.50, "min": 0.89},
        "Drogerie & Haushalt": {"mu": 4.00, "sigma": 2.00, "min": 0.99}, "Tiefkühlware": {"mu": 3.80, "sigma": 1.50, "min": 1.59},
        "Kaffee & Tee": {"mu": 5.50, "sigma": 2.50, "min": 1.99}, "default": {"mu": 2.50, "sigma": 1.00, "min": 0.69}
    }
    
    @classmethod
    def generate_price(cls, category: str) -> str:
        """
        Erzeugt einen Kaufpreis mittels Gauß-Verteilung (Normalverteilung).
        
        Args:
            category (str): Die Supermarktkategorie.

        Returns:
            str: Der formatierte Preis (z.B. '2.99 €').
        """
        params = cls._price_distributions.get(category, cls._price_distributions["default"])
        price = max(random.gauss(params["mu"], params["sigma"]), params["min"])
        return f"{math.floor(price) + random.choice([0.99, 0.49, 0.89, 0.29]):.2f} €"

class StoreBuilder:
    """Befüllt den digitalen Supermarkt-Graphen mittels Sainte-Laguë-Verfahren."""
    
    def __init__(self):
        self.stock = {node: [] for nodes in StoreOntology.FIXED_ZONES.values() for node in nodes}
        for node in StoreOntology.FLEXIBLE_ZONES: self.stock[node] = []
        self.capacity = {node: 0 for node in self.stock.keys()}

    def _allocate_kassen(self):
        for q_node, item_list in StoreOntology.HARDWARE_INJECTIONS.items():
            for name, brand in item_list:
                self.stock[q_node].append({
                    'name': name, 'brand': brand, 'category': 'Sonstiges (Kasse)', 'price': PricingEngine.generate_price("default"), 
                    'ai_confidence': 1.0, 'suggested_slot': q_node, 'needs_review': False
                })
                self.capacity[q_node] += 1

    def execute(self, items: List[str]):
        """
        Führt die Allokation der Produkte auf die Regale durch.
        Nutzt das Sainte-Laguë-Verfahren, um flexible Kategorien fair auf den Graphen aufzuteilen.
        
        Args:
            items (List[str]): Liste der zu platzierenden Produkte.

        Returns:
            Tuple: Das final befüllte Inventar und die Routing-Map für das Frontend.
        """
        log.info("Phase 3: Klassifiziere und alloziiere Produkte...")
        self._allocate_kassen()
        
        batch_results = ml_predictor.predict_batch(items)
        
        # PANDAS EMPTY DATAFRAME FIX
        records = [
            {'Name': items[i].title(), 'Cat': r[0], 'Conf': r[2], 'Price': PricingEngine.generate_price(r[0])} 
            for i, r in enumerate(batch_results) if r[0] != "Sonstiges (Kasse)" and r[2] >= StoreOntology.ML_CONFIDENCE_THRESHOLD
        ]
        df = pd.DataFrame(records, columns=['Name', 'Cat', 'Conf', 'Price'])

        # Sainte-Laguë für Flexible Zones
        counts = {k: 1 for k in StoreOntology.EMERGENCY_FALLBACKS.keys() if k not in StoreOntology.FIXED_ZONES}
        if not df.empty:
            dynamic_cats = [c for c in df['Cat'] if c not in StoreOntology.FIXED_ZONES]
            actual_counts = pd.Series(dynamic_cats).value_counts().to_dict()
            for cat, count in actual_counts.items():
                if cat in counts: counts[cat] += count

        allocation = {cat: 1 for cat in counts.keys()}
        slots_to_distribute = len(StoreOntology.FLEXIBLE_ZONES) - len(allocation)
        for _ in range(slots_to_distribute):
            winner = max({cat: (count / (allocation[cat] + 0.5)) for cat, count in counts.items()}, key=lambda k: {cat: (count / (allocation[cat] + 0.5)) for cat, count in counts.items()}.get(k))
            allocation[winner] += 1

        available_nodes = StoreOntology.FLEXIBLE_ZONES.copy()
        random.shuffle(available_nodes)
        
        routing_map = {**StoreOntology.FIXED_ZONES}
        pointer = 0
        for cat, amount in allocation.items():
            routing_map[cat] = available_nodes[pointer : pointer+amount]
            pointer += amount

        # Befüllen
        for cat, nodes in routing_map.items():
            if cat == "Sonstiges (Kasse)": continue
            cat_items = df[df['Cat'] == cat]
            for _, row in cat_items.iterrows():
                available = [n for n in nodes if self.capacity[n] < StoreOntology.MAX_CAPACITY]
                if not available: break 
                best_node = min(available, key=lambda n: self.capacity[n])
                brand = random.choice(StoreOntology.BRAND_DISTRIBUTION.get(cat, ["JMU Choice"]))
                self.stock[best_node].append({
                    'name': row['Name'], 'brand': brand, 'category': cat, 'price': row['Price'], 
                    'ai_confidence': round(row['Conf'], 3), 'suggested_slot': best_node, 'needs_review': False
                })
                self.capacity[best_node] += 1

            # Fallbacks
            while any(self.capacity[n] < StoreOntology.MAX_CAPACITY for n in nodes):
                node = min(nodes, key=lambda n: self.capacity[n])
                fallback = random.choice(StoreOntology.EMERGENCY_FALLBACKS.get(cat, ["Basic Item"]))
                brand = random.choice(StoreOntology.BRAND_DISTRIBUTION.get(cat, ["JMU Basic"]))
                self.stock[node].append({
                    'name': fallback, 'brand': brand, 'category': cat, 'price': PricingEngine.generate_price(cat), 
                    'ai_confidence': 1.0, 'suggested_slot': node, 'needs_review': False
                })
                self.capacity[node] += 1
                
        return self.stock, routing_map

# =============================================================================
# IV. MASTER ORCHESTRATOR
# =============================================================================

class MasterOrchestrator:
    """Steuert den gesamten Ausführungsablauf (Orchestration) der ETL-Pipeline."""
    
    def execute(self):
        """Führt alle Phasen nacheinander aus und exportiert die Ergebnisse."""
        print("\n" + "="*75)
        print("🏭 ENTERPRISE STORE BUILDER v18.0 (Master Pipeline)")
        print("="*75)
        t_start = time.time()
        
        input_path = Path(PipelineConfig.EXCEL_INPUT)
        csv_path = Path(PipelineConfig.CSV_OUTPUT)
        
        # Warnungen von Pandas bzgl. RegEx unterdrücken (Kosmetik für die Konsole)
        warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
        
        # 1. Excel Download falls nötig
        if not input_path.exists():
            if not BLSDownloader.download(PipelineConfig.BLS_URL, input_path):
                sys.exit(1)
                
        # 2. CSV Bauen (Phase 1)
        csv_builder = CSVBuilder(input_path, csv_path)
        csv_builder.execute()
        
        # 3. CRITICAL FIX: Zwingt das Backend zum Retraining! (Löscht altes Pickle)
        log.info("Phase 2: Lösche altes ML-Modell und erzwinge Retraining...")
        model_file = Path(CONFIG.ML_MODEL_FILE)
        if model_file.exists():
            model_file.unlink() 
        ml_predictor._sync_model() 
        
        # 4. Supermarkt bauen (Phase 3)
        raw_items = DataIngestionService.extract_and_sort(str(csv_path), limit=1500)
        builder = StoreBuilder()
        final_stock, routing_map = builder.execute(raw_items)
        
        # 5. Exportieren
        with inv_manager._lock:
            inv_manager.stock = final_stock
            inv_manager.save_to_json()
            
        with open(PipelineConfig.ROUTING_OUTPUT, "w", encoding='utf-8') as f:
            json.dump(routing_map, f, indent=4, ensure_ascii=False)

        t_end = time.time()
        placed = sum(len(items) for items in final_stock.values())
        total_nodes = len(StoreOntology.FIXED_ZONES["Spirituosenschrank"]) + len(StoreOntology.FIXED_ZONES["Fleischtheke"]) + len(StoreOntology.FIXED_ZONES["Sonstiges (Kasse)"]) + len(StoreOntology.FIXED_ZONES["Obst & Gemüse"]) + len(StoreOntology.FIXED_ZONES["Backwaren"]) + len(StoreOntology.FLEXIBLE_ZONES)
        
        print("\n" + "="*75)
        print("📊 ETL SYSTEM AUDIT & VALIDIERUNGS-REPORT")
        print("="*75)
        print(f"⏱️  Ausführungszeit:        {t_end - t_start:.2f} Sekunden")
        print(f"📦 Regale befüllt:         {len(final_stock)} / {total_nodes}")
        print(f"🛒 Produkte im System:     {placed} / {total_nodes * StoreOntology.MAX_CAPACITY}")
        print("\n📂 Systemdateien erfolgreich generiert und synchronisiert:")
        print(f"   ├── {PipelineConfig.CSV_OUTPUT} (NLP Ground Truth)")
        print(f"   ├── {PipelineConfig.DB_OUTPUT}        (Inventar & Pricing-Daten)")
        print(f"   └── {PipelineConfig.ROUTING_OUTPUT}  (Sainte-Laguë Graphen-Routing)")
        print("-" * 75)
        print("🚀 Setup abgeschlossen. Starte nun die app.py!")

if __name__ == "__main__":
    orchestrator = MasterOrchestrator()
    orchestrator.execute()