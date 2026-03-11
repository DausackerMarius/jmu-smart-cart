#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Smart Cart Data Engineering Pipeline (Enterprise Edition V8 - ML READY)
-------------------------------------------------------------------------
Zweck: Extraktion, NLP-Bereinigung und Transformation des 
offiziellen Bundeslebensmittelschlüssels (BLS) in einen 
Machine-Learning-bereiten Datensatz für XGBoost.

Neu in V8:
- Tokenizer-Fix: Slashes (/) werden zu Leerzeichen.
- Retail-Filter: Löscht anatomische Nischenprodukte (Wortanzahl > 3).
- Class Imbalance Reduktion: Reduziert den Fleisch-Überhang.
"""

import pandas as pd
import argparse
import logging
import sys
import os
import re
import json
from pathlib import Path
from typing import Optional, Tuple
import requests
from requests.exceptions import RequestException

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# =============================================================================
# 1. KONFIGURATION & LEXIKA 
# =============================================================================

class Config:
    BLS_URL = "https://downloads.blsdb.de/bls_4.0.xlsx"
    DEFAULT_INPUT = "bls_4.0.xlsx"
    DEFAULT_OUTPUT = "smartcart_ml_training_data.csv"
    METADATA_OUTPUT = "dataset_statistics.json"
    
    EN_KILLER_WORDS = {
        'raw', 'dried', 'beef', 'pork', 'juice', 'cheese', 'meat', 'frozen',
        'boiled', 'canned', 'drained', 'stewed', 'fried', 'roasted', 'baked',
        'sweet', 'sour', 'white', 'black', 'red', 'green', 'yellow', 'mixed',
        'apple', 'water', 'sausage', 'fish', 'chicken', 'turkey', 'duck',
        'veal', 'mutton', 'lamb', 'fat', 'oil', 'wine', 'beer', 'sugar',
        'syrup', 'flour', 'bread', 'roll', 'cake', 'pie', 'cookie', 'biscuit',
        'deep-frozen', 'unsweetened', 'sugared', 'smoked', 'salted', 'puree',
        'seed', 'bean', 'pea', 'nut', 'cuttlefish', 'flounder', 'trout', 
        'salmon', 'goose', 'muesli', 'beverage', 'starch', 'noodles', 'paste',
        'powder', 'candied', 'extract', 'spirit', 'drink', 'fillet'
    }

    B2B_NOISE_WORDS = {
        'fettgewebe', 'blut', 'intermuskulär', 'schlachtkörper', 'innereien',
        'hirn', 'lunge', 'herz', 'leber', 'niere', 'milz', 'bries', 'zunge',
        'magen', 'kutteln', 'schlund', 'knochenmark', 'fettwamme', 'fettabschnitte',
        'blutpresssack', 'schweinskopfsülze', 'darmschmalz', 'sehnenfrei', 'sehnenarm',
        'kranial', 'kaudal', 'subkutan', 'isolat', 'extrakt', 'tiermehl', 'rohmasse', 
        'brät', 'kutterhilfsmittel', 'pottasche', 'hirschhornsalz', 'milchsäure', 
        'citronensäure', 'ascorbinsäure', 'essigsäure', 'kakaomasse', 'sahnestandmittel', 
        'gelatine', 'magermilchpulver', 'vollei', 'trockenhefe', 'säuerungsmittel',
        'künstlich', 'aroma'
    }

# =============================================================================
# 2. LOGGING INFRASTRUKTUR
# =============================================================================

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("BLSPipeline")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

log = setup_logger()

# =============================================================================
# 3. DOWNLOAD MANAGER
# =============================================================================

class BLSDownloader:
    def __init__(self, url: str, dest_path: Path):
        self.url = url
        self.dest_path = dest_path

    def download(self) -> bool:
        if self.dest_path.exists():
            log.info(f"Lokale Datei gefunden: {self.dest_path}. Überspringe Download.")
            return True

        log.info(f"Initiiere Download von: {self.url}")
        try:
            response = requests.get(self.url, stream=True, timeout=60)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            with open(self.dest_path, 'wb') as file:
                if HAS_TQDM:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading BLS") as pbar:
                        for chunk in response.iter_content(chunk_size=1024 * 8):
                            if chunk:
                                file.write(chunk)
                                pbar.update(len(chunk))
                else:
                    log.info("Lade Daten herunter... (Bitte warten)")
                    for chunk in response.iter_content(chunk_size=1024 * 8):
                        if chunk: file.write(chunk)
            return True
        except RequestException as e:
            log.error(f"Netzwerk- oder Downloadfehler: {e}")
            return False

# =============================================================================
# 4. NLP TEXT PROCESSING ENGINE
# =============================================================================

class TextSanitizer:
    @staticmethod
    def clean(name: str) -> Optional[str]:
        if pd.isna(name): return None
        name = str(name).strip()
        name_lower = name.lower()
        
        # 1. B2B Filter
        if any(noise in name_lower for noise in Config.B2B_NOISE_WORDS): return None
            
        # 2. Englisch Filter
        for en_word in Config.EN_KILLER_WORDS:
            if re.search(r'\b' + en_word + r'\b', name_lower): return None
                
        # 3. Klammer-Annihilation VOR dem Split
        name = re.sub(r'\s*\(.*?\)', '', name)
        name = re.sub(r'\s*\[.*?\]', '', name)
            
        # 4. Komma-Cutoff & Signalwort-Cutoff
        name = name.split(',')[0]
        signal_words = [r'\bmind\.', r'\bmit\b', r'\bohne\b', r'\bin\b', r'\baus\b', r'\bnach\b']
        for signal in signal_words:
            match = re.search(signal, name, flags=re.IGNORECASE)
            if match: name = name[:match.start()]
            
        # 5. Tokenizer-Fix: Slashes zu Leerzeichen
        name = name.replace('/', ' ')
        
        # 6. Zombie-Wörter & Adjektive löschen (jetzt auch "Fett i.Tr." und "vol")
        words_to_strip = [
            r'\broh\b', r'\bgekocht\b', r'\bgebraten\b', r'\bgedünstet\b', r'\bgedämpft\b',
            r'\btiefgefroren\b', r'\bkonserve\b', r'\babgetropft\b', r'\bheiß\b',
            r'\bungesüßt\b', r'\bgezuckert\b', r'\bgeschmort\b', r'\bpochiert\b',
            r'\bmariniert\b', r'\bgesalzen\b', r'\bgeräuchert\b', r'\bpräserve\b',
            r'\bfett\b', r'\bi\.?\s*Tr\.?\b', r'\bvol\b', r'%'
        ]
        for word in words_to_strip:
            name = re.sub(word, '', name, flags=re.IGNORECASE)
            
        # 7. Isolierte Zahlen löschen
        name = re.sub(r'\b\d+(,\d+)?\b', '', name)
        
        # 8. RETAIL-FILTER: Maximal 3 Wörter erlaubt (verhindert Class Imbalance durch Spezial-Fleisch)
        name = re.sub(r'[-/&]$', '', name.strip())
        name = re.sub(r'\s+', ' ', name).strip()
        
        word_count = len(name.split())
        if word_count == 0 or word_count > 3: 
            return None
            
        return name

# =============================================================================
# 5. SUPERMARKT MAPPER
# =============================================================================

class SupermarketMapper:
    @staticmethod
    def generate_label(bls_code: str, original_name: str) -> Optional[str]:
        bls_code = str(bls_code).strip().upper()
        name_lower = str(original_name).lower()
        
        if bls_code.startswith(('X', 'Y', 'W9', 'U9', 'V9')): return None
            
        if 'tiefgefroren' in name_lower or 'eiscreme' in name_lower or bls_code.startswith('S2'):
            return "Tiefkühlware"
            
        if any(w in name_lower for w in ['brot', 'brötchen', 'baguette', 'toast', 'tortilla', 'stärke', 'mehl', 'hefe']):
            return "Backwaren & Backzutaten"
            
        if any(w in name_lower for w in ['tofu', 'soja', 'vegan', 'vegetarisch', 'seitan']):
            if 'trocken' not in name_lower and 'pulver' not in name_lower:
                return "Kühlregal (Molkerei & Veggie)"
                
        if any(w in name_lower for w in ['nudel', 'pasta', 'spaghetti', 'penne', 'teigwaren', 'reis']):
            if 'frischteigwaren' in name_lower: return "Kühlregal (Molkerei & Veggie)"
            return "Trockensortiment (Nudeln/Reis/Dosen)"
            
        if 'konserve' in name_lower or 'abgetropft' in name_lower or 'präserve' in name_lower:
            return "Trockensortiment (Nudeln/Reis/Dosen)"
            
        if 'müsli' in name_lower or 'cornflakes' in name_lower or 'haferflocken' in name_lower:
            return "Frühstück & Cerealien"
            
        if 'chips' in name_lower or 'flips' in name_lower or 'salzstangen' in name_lower:
            return "Süßwaren & Snacks"

        first_letter = bls_code[0] if len(bls_code) > 0 else '?'
        
        mapping = {
            'B': "Backwaren & Backzutaten",
            'C': "Backwaren & Backzutaten",
            'D': "Backwaren & Backzutaten",
            'H': "Backwaren & Backzutaten",
            'F': "Obst & Gemüse",
            'G': "Obst & Gemüse",
            'K': "Obst & Gemüse",
            'M': "Kühlregal (Molkerei & Veggie)",
            'T': "Fischtheke",
            'U': "Fleischtheke & Wurst",
            'V': "Fleischtheke & Wurst",
            'W': "Fleischtheke & Wurst",
            'S': "Süßwaren & Snacks",
            'Q': "Öle & Fette",
            'R': "Gewürze & Saucen"
        }
        
        if first_letter == 'E':
            if bls_code.startswith('E1'): return "Kühlregal (Molkerei & Veggie)" 
            return "Trockensortiment (Nudeln/Reis/Dosen)" 
        if first_letter in ['N', 'P']:
            if bls_code.startswith('N4'): return "Kaffee & Tee"
            return "Getränke & Wein"
            
        return mapping.get(first_letter, "Sonstiges")

# =============================================================================
# 6. ORCHESTRATOR
# =============================================================================

class PipelineOrchestrator:
    def __init__(self, input_file: Path, output_file: Path):
        self.input_file = input_file
        self.output_file = output_file
        self.df_raw = None
        self.df_clean = None
        
    def _locate_target_columns(self) -> Tuple[str, str]:
        code_col, name_col = None, None
        
        for col in self.df_raw.columns:
            sample = self.df_raw[col].dropna().astype(str).head(50)
            if len(sample) > 0 and (sample.str.match(r'^[A-Z][A-Z0-9]{4,6}$').sum() / len(sample)) > 0.5:
                code_col = col
                break
                
        max_score = -1
        for col in self.df_raw.columns:
            if col == code_col: continue
            
            sample = self.df_raw[col].dropna().astype(str).head(100)
            if len(sample) == 0: continue
            if sample.nunique() < (len(sample) * 0.4): continue 
                
            sample_lower = sample.str.lower()
            score = 0
            
            score += sample_lower.str.contains(r'ä|ö|ü|ß').sum() * 10
            score += sample_lower.str.contains(r'\b(roh|gekocht|brot|fleisch|käse)\b').sum() * 5
            score -= sample_lower.str.contains(r'\b(raw|beef|cheese|dried|meat)\b').sum() * 20
            
            if score > max_score:
                max_score = score
                name_col = col
                
        if not code_col: code_col = self.df_raw.columns[0]
        if not name_col: name_col = self.df_raw.columns[1]
        return code_col, name_col

    def run(self) -> bool:
        try:
            log.info("Lade Rohdaten in den Speicher...")
            xl = pd.ExcelFile(self.input_file)
            sheet_name = next((s for s in ['Lebensmittel', 'LEBM', 'Daten', 'Sheet1'] if s in xl.sheet_names), xl.sheet_names[0])
            self.df_raw = pd.read_excel(self.input_file, sheet_name=sheet_name, dtype=str)
            self.df_raw.columns = [str(c).strip().upper() for c in self.df_raw.columns]
            
            code_col, name_col = self._locate_target_columns()
            
            df = pd.DataFrame()
            df['BLS_Code'] = self.df_raw[code_col]
            df['Original_Name'] = self.df_raw[name_col]
            df = df.dropna()
            
            log.info("Wende Hybrides Labeling an...")
            df['Supermarket_Category'] = df.apply(
                lambda row: SupermarketMapper.generate_label(row['BLS_Code'], row['Original_Name']), 
                axis=1
            )
            df = df.dropna(subset=['Supermarket_Category'])
            
            log.info("Wende Tokenizer-Fix und Retail-Filter an...")
            df['Product_Name'] = df['Original_Name'].apply(TextSanitizer.clean)
            df = df.dropna(subset=['Product_Name'])
            
            df['Name_Lower'] = df['Product_Name'].str.lower()
            df = df.drop_duplicates(subset=['Name_Lower']).drop(columns=['Name_Lower'])
            
            self.df_clean = df[['Product_Name', 'Supermarket_Category']].copy()
            
            log.info(f"Transformation beendet. Finaler Datensatz: {len(self.df_clean)} Artikel.")
            return True
            
        except Exception as e:
            log.exception(f"Kritischer Pipeline-Fehler: {e}")
            return False

    def export(self) -> None:
        if self.df_clean is not None:
            self.df_clean.to_csv(self.output_file, index=False, encoding='utf-8')
            log.info(f"✅ ML-DATENSATZ EXPORTIERT: {self.output_file}")
            print("\n" + "="*55)
            print(" XGBOOST LABEL VERTEILUNG (GROUND TRUTH)")
            print("="*55)
            print(self.df_clean['Supermarket_Category'].value_counts())

# =============================================================================
# 7. ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Smart Cart ETL Pipeline V8")
    parser.add_argument("--input", "-i", type=str, default=None, help="Lokaler Pfad zur BLS Excel")
    parser.add_argument("--output", "-o", type=str, default=Config.DEFAULT_OUTPUT, help="Output CSV")
    parser.add_argument("--redownload", action="store_true", help="Download erzwingen")
    args = parser.parse_args()

    print("=" * 60)
    print(" SMART CART DATA ENGINEERING PIPELINE (V8 - ML READY)")
    print("=" * 60)

    input_path = Path(args.input) if args.input else Path(Config.DEFAULT_INPUT)
    
    if args.redownload and input_path.exists() and not args.input:
        os.remove(input_path)

    if not input_path.exists():
        if not BLSDownloader(Config.BLS_URL, input_path).download():
            sys.exit(1)

    orchestrator = PipelineOrchestrator(input_path, Path(args.output))
    if orchestrator.run():
        orchestrator.export()
    else:
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())