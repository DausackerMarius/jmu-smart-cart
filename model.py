"""
=========================================================================================
JMU SMART SUPERMARKET: ENTERPRISE ANALYTICS & DETERMINISTIC ROUTING KERNEL
=========================================================================================
Architektur: SOLID, Thread-Safe Singleton, Deterministic Strategy Pattern
Module:
1. Algorithmic String Matching (Custom Damerau-Levenshtein, Cologne Phonetics)
2. MLOps Pipeline (Word-Level Logistic Regression & Compound Anchors)
3. Evolutionary Computation (Genetic Algorithm, Ant Colony, SA, DP Held-Karp)
4. Stochastic Queuing Theory (M/M/1/K Markov Chains - Realistische Zeiten)
5. Spatial Graph Topology
=========================================================================================
"""

import os
import re
import json
import time
import math
import random
import pickle
import logging
import hashlib
import threading
import itertools
from datetime import datetime
from functools import lru_cache, wraps
from typing import List, Tuple, Dict, Optional, Any, Set, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass

import networkx as nx
import pandas as pd
import numpy as np

# --- MLOps ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib

# =============================================================================
# I. INFRASTRUCTURE & EXCEPTIONS
# =============================================================================

class StoreBackendException(Exception): pass
class ConfigurationError(StoreBackendException): pass
class GraphTopologyError(StoreBackendException): pass

@dataclass(frozen=True)
class SystemConfig:
    DB_FILE: str = "products.json"
    ROUTING_FILE: str = "routing_config.json"
    TRAINING_CSV: str = "smartcart_ml_training_data.csv"
    ML_MODEL_FILE: str = "smartcart_svm_v_final.pkl" 
    TRAFFIC_MODEL_FILE: str = "traffic_model_max_perf.pkl"
    CACHE_DIR: str = "system_cache"
    
    FUZZY_MATCH_THRESHOLD: float = 65.0
    MIN_ML_CONFIDENCE: float = 0.25
    TFIDF_NGRAM_RANGE: Tuple[int, int] = (1, 3)
    
    DP_EXACT_LIMIT: int = 11
    SA_THRESHOLD: int = 15  
    SA_START_TEMP: float = 5000.0
    SA_COOLING_RATE: float = 0.995
    SA_MIN_TEMP: float = 0.01
    SA_MAX_ITERATIONS: int = 6000
    
    GA_POPULATION_SIZE: int = 150
    GA_ELITISM_RATIO: float = 0.1
    GA_MUTATION_RATE: float = 0.15
    GA_MAX_GENERATIONS: int = 300
    
    ACO_NUM_ANTS: int = 40
    ACO_ITERATIONS: int = 100
    ACO_EVAPORATION: float = 0.15
    ACO_ALPHA: float = 1.0
    ACO_BETA: float = 2.5
    
    # UI Hex-Codes
    COLOR_RED: str = "#d62728"
    COLOR_BLACK: str = "#000000"
    COLOR_BLUE: str = "#1f77b4"
    COLOR_CYAN: str = "#17becf"
    COLOR_ORANGE: str = "#ff7f0e"
    COLOR_GREEN: str = "#2ca02c"
    COLOR_SHELF: str = "#e5e5e5"
    COLOR_SHELF_BORDER: str = "#999999"
    COLOR_HIGHLIGHT: str = "#FFD700"

CONFIG = SystemConfig()
if not os.path.exists(CONFIG.CACHE_DIR): os.makedirs(CONFIG.CACHE_DIR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | [%(name)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger("StoreEngine")

def execution_profiler(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        dur = (time.perf_counter() - t0) * 1000.0
        log.debug(f"[PROFILER] {func.__name__} executed in {dur:.4f} ms")
        return result
    return wrapper

# =============================================================================
# II. SPATIAL TOPOLOGY & ROUTING CONFIGURATION
# =============================================================================

class DynamicConfigManager:
    _lock = threading.RLock()
    @staticmethod
    def get_routing() -> Dict[str, List[str]]:
        with DynamicConfigManager._lock:
            if os.path.exists(CONFIG.ROUTING_FILE):
                try:
                    with open(CONFIG.ROUTING_FILE, 'r', encoding='utf-8') as f: return json.load(f)
                except Exception: pass
            
            return {
                'Spirituosenschrank': ['v5'], 'Fleischtheke': ['vA10'], 'Sonstiges (Kasse)': ['vW1', 'vW2', 'vW3'],
                'Backwaren': ['v2', 'vB2', 'vB7', 'vD4', 'vC7'], 'Fisch & Wurstwaren': ['v3', 'vC5', 'vB9'],
                'Obst & Gemüse': ['vC3', 'vC2', 'v1'], 'Getränke Alkoholfrei': ['vD6', 'vB10', 'vB4'],
                'Gewürze & Backzutaten': ['vD1', 'vA2', 'vD5'], 'Süßwaren & Snacks': ['vA4'],
                'Trockensortiment & Konserven': ['vB6'], 'Kühlregal (Molkerei)': ['vA9_2'],
                'Bier & Wein': ['vD2'], 'Tiefkühlware': ['vC6'], 'Kühlregal (Vegan & Käse)': ['vA6'],
                'Fitness & Sport': ['vA7'], 'Kaffee & Tee': ['vA9']
            }

CATEGORY_ROUTING = DynamicConfigManager.get_routing()
slot_to_node = {k: v[0] for k, v in CATEGORY_ROUTING.items() if v}

# FIX: Das fehlende node_to_category Dictionary
node_to_category = {}
for category, nodes in CATEGORY_ROUTING.items():
    for node in nodes:
        node_to_category[node] = category

class StoreTopology:
    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(StoreTopology, cls).__new__(cls)
                cls._instance._boot()
        return cls._instance

    def _boot(self):
        self.nodes_config: Dict[str, Dict[str, Any]] = {}
        self.edges_config: List[Tuple[str, str, float]] = []
        self.shelves: List[Dict[str, float]] = []
        self.G_base = nx.Graph()
        
        self._construct_nodes()
        self._construct_edges()
        self._construct_shelves()
        self._initialize_networkx()

    def _construct_nodes(self):
        X = [0.0, 3.0, 4.0, 9.0, 13.0, 17.0, 22.0, 26.0, 28.0, 29.0, 30.0, 32.0, 34.0, 36.0]
        Y = [0.0, 2.0, 4.0, 7.0, 8.0, 10.0, 12.0, 14.0]
        
        self.nodes_config = {
            'vD6': {'pos': (X[0], Y[6]), 'col': CONFIG.COLOR_BLACK}, 'vD5': {'pos': (X[2], Y[6]), 'col': CONFIG.COLOR_BLACK}, 
            'vD4': {'pos': (X[3], Y[6]), 'col': CONFIG.COLOR_BLACK}, 'vD3': {'pos': (X[4], Y[6]), 'col': CONFIG.COLOR_RED}, 
            'vD2': {'pos': (X[5], Y[6]), 'col': CONFIG.COLOR_BLACK}, 'vD1': {'pos': (X[6], Y[6]), 'col': CONFIG.COLOR_BLACK},
            'vC7': {'pos': (X[0], Y[4]), 'col': CONFIG.COLOR_BLACK}, 'vC6': {'pos': (X[2], Y[4]), 'col': CONFIG.COLOR_BLACK}, 
            'vC5': {'pos': (X[3], Y[4]), 'col': CONFIG.COLOR_BLACK}, 'vC4': {'pos': (X[4], Y[4]), 'col': CONFIG.COLOR_RED}, 
            'vC3': {'pos': (X[5], Y[4]), 'col': CONFIG.COLOR_BLACK}, 'vC2': {'pos': (X[6], Y[4]), 'col': CONFIG.COLOR_BLACK}, 
            'vB12': {'pos': (X[0], Y[2]), 'col': CONFIG.COLOR_RED}, 
            'vB10': {'pos': (X[2], Y[2]), 'col': CONFIG.COLOR_BLACK}, 'vB9': {'pos': (X[3], Y[2]), 'col': CONFIG.COLOR_BLACK}, 
            'vB8': {'pos': (X[4], Y[2]), 'col': CONFIG.COLOR_RED}, 'vB7': {'pos': (X[5], Y[2]), 'col': CONFIG.COLOR_BLACK}, 
            'vB6': {'pos': (X[6], Y[2]), 'col': CONFIG.COLOR_BLACK},
            'vA10': {'pos': (X[0], Y[0]), 'col': CONFIG.COLOR_BLUE}, 'vA9_2': {'pos': (X[2], Y[0]), 'col': CONFIG.COLOR_BLACK}, 
            'vA9': {'pos': (X[3], Y[0]), 'col': CONFIG.COLOR_BLACK}, 'vA8': {'pos': (X[4], Y[0]), 'col': CONFIG.COLOR_RED}, 
            'vA7': {'pos': (X[5], Y[0]), 'col': CONFIG.COLOR_BLACK}, 'vA6': {'pos': (X[6], Y[0]), 'col': CONFIG.COLOR_BLACK},
            'vW1': {'pos': (X[7], Y[6]), 'col': CONFIG.COLOR_BLACK}, 'vW2': {'pos': (X[7], Y[5]), 'col': CONFIG.COLOR_BLACK}, 
            'vW3': {'pos': (X[7], Y[4]), 'col': CONFIG.COLOR_BLACK}, 'v4': {'pos': (X[7], Y[3]), 'col': CONFIG.COLOR_RED},    
            'vB5': {'pos': (X[7], Y[2]), 'col': CONFIG.COLOR_RED}, 'v3': {'pos': (X[7], Y[1]), 'col': CONFIG.COLOR_BLACK}, 
            'vA5': {'pos': (X[7], Y[0]), 'col': CONFIG.COLOR_RED},   
            'v5': {'pos': (X[9], Y[3]), 'col': CONFIG.COLOR_BLUE},        
            'vK1': {'pos': (X[9], Y[6]), 'col': CONFIG.COLOR_ORANGE}, 'vK2': {'pos': (X[9], Y[5]), 'col': CONFIG.COLOR_ORANGE},    
            'vK3': {'pos': (X[9], Y[4]), 'col': CONFIG.COLOR_ORANGE},     
            'vB4': {'pos': (X[8], Y[2]), 'col': CONFIG.COLOR_BLACK}, 'vB3': {'pos': (X[10], Y[2]), 'col': CONFIG.COLOR_RED}, 
            'vB2': {'pos': (X[11], Y[2]), 'col': CONFIG.COLOR_BLACK}, 'vB1': {'pos': (X[12], Y[2]), 'col': CONFIG.COLOR_RED}, 
            'vA4': {'pos': (X[8], Y[0]), 'col': CONFIG.COLOR_BLACK}, 'vA3': {'pos': (X[10], Y[0]), 'col': CONFIG.COLOR_RED}, 
            'vA2': {'pos': (X[11], Y[0]), 'col': CONFIG.COLOR_BLACK}, 'vA1': {'pos': (X[12], Y[0]), 'col': CONFIG.COLOR_RED}, 
            'v2': {'pos': (X[10], Y[1]), 'col': CONFIG.COLOR_BLACK}, 'v1': {'pos': (X[12], Y[1]), 'col': CONFIG.COLOR_BLACK},
            'vAusgang': {'pos': (X[9], Y[7]), 'col': CONFIG.COLOR_GREEN}, 
            'vEingang': {'pos': (X[13], Y[7]), 'col': CONFIG.COLOR_GREEN}, 
            'vInCorner': {'pos': (X[13], Y[2]), 'col': CONFIG.COLOR_BLACK, 'hidden': True}
        }

    def _construct_edges(self):
        self.edges_config = [
            ('vD6','vD5',4.0), ('vD5','vD4',5.0), ('vD4','vD3',4.0), ('vD3','vD2',4.0), ('vD2','vD1',5.0),
            ('vC7','vC6',4.0), ('vC6','vC5',5.0), ('vC5','vC4',4.0), ('vC4','vC3',4.0), ('vC3','vC2',5.0),
            ('vB10','vB9',5.0), ('vB9','vB8',4.0), ('vB8','vB7',4.0), ('vB7','vB6',5.0), 
            ('vA10','vA9_2',4.0), ('vA9_2','vA9',5.0), ('vA9','vA8',4.0), ('vA8','vA7',4.0), ('vA7','vA6',5.0),
            ('vD6','vC7',4.0), ('vC7','vB12',4.0), ('vB12','vA10',4.0), ('vB12','vB10',4.0), 
            ('vD3','vC4',4.0), ('vC4','vB8',4.0), ('vB8','vA8',4.0),
            ('vD1','vW1',4.0), ('vC2','vW3',4.0), ('vB6','vB5',4.0), ('vA6','vA5',4.0),
            ('vW1','vW2',2.0), ('vW2','vW3',2.0), ('vW3','v4',1.0), ('v4','vB5',3.0), 
            ('vB5','v3',2.0), ('v3','vA5',2.0), ('vW1','vK1',3.0), ('vW2','vK2',3.0), 
            ('vW3','vK3',3.0), ('v4','v5',3.0), 
            ('vB5','vB4',2.0), ('vB4','vB3',2.0), ('vB3','vB2',2.0), ('vB2','vB1',2.0),
            ('vA5','vA4',2.0), ('vA4','vA3',2.0), ('vA3','vA2',2.0), ('vA2','vA1',2.0),
            ('vB3','v2',2.0), ('v2','vA3',2.0), ('vB1','v1',2.0), ('v1','vA1',2.0), 
            ('vK1','vAusgang',2.0), ('vK2','vAusgang',2.0), ('vK3','vAusgang',2.0),
            ('vEingang','vInCorner',10.0), ('vInCorner','vB1',2.0)
        ]

    def _construct_shelves(self):
        H_H = 1.2; PAD = 1.5
        SHELF_LENGTH = 9.0  # Alle Regale gleich lang (vergrößert)
        LEFT_CENTER = 6.5   # Zentrum linke Seite
        RIGHT_CENTER = 19.5  # Zentrum rechte Seite
        for y in [10.0, 6.0, 2.0]: 
            # Linkes Regal: zentriert um LEFT_CENTER
            self.shelves.append({'x0': LEFT_CENTER - SHELF_LENGTH/2, 'x1': LEFT_CENTER + SHELF_LENGTH/2, 'y0': y-H_H, 'y1': y+H_H})
            # Rechtes Regal: zentriert um RIGHT_CENTER
            self.shelves.append({'x0': RIGHT_CENTER - SHELF_LENGTH/2, 'x1': RIGHT_CENTER + SHELF_LENGTH/2, 'y0': y-H_H, 'y1': y+H_H})
        self.shelves.extend([
            {'x0': -2.5, 'x1': 24.3, 'y0': 12.8, 'y1': 13.8},
            {'x0': 24.3, 'x1': 27.0, 'y0': 12.8, 'y1': 13.8},
            {'x0': -2.5, 'x1': -1.5, 'y0': 4.8, 'y1': 13.8},
            {'x0': -1.5, 'x1': -0.8, 'y0': -1.0, 'y1': 3.2}
        ])
        # 2 Quadrate exakt zentriert in den Umrandungen - linkes weiter nach rechts
        self.shelves.extend([
            {'x0': 26.7, 'x1': 29.3, 'y0': 0.7, 'y1': 3.3},   # Linkes Quadrat: weiter nach rechts
            {'x0': 30.7, 'x1': 33.3, 'y0': 0.7, 'y1': 3.3},   # Rechtes Quadrat: perfekt
        ])
        self.shelves.extend([
            {'x0': 26.2, 'x1': 28.8, 'y0': 11.5, 'y1': 12.5},
            {'x0': 26.2, 'x1': 28.8, 'y0': 9.5, 'y1': 10.5},
            {'x0': 26.2, 'x1': 28.8, 'y0': 7.5, 'y1': 8.5}
        ])

    def _initialize_networkx(self):
        for n, data in self.nodes_config.items():
            if not data.get('hidden', False): self.G_base.add_node(n, pos=data['pos'])
        for u, v, w in self.edges_config:
            if u in self.nodes_config and v in self.nodes_config: self.G_base.add_edge(u, v, weight=float(w))
        self.G_base.add_node('vInCorner', pos=self.nodes_config['vInCorner']['pos'])
        if not nx.is_connected(self.G_base): raise GraphTopologyError("Isolierte Subgraphen.")

topology = StoreTopology()

# =============================================================================
# III. NATIVE ALGORITHMIC STRING MATCHING
# =============================================================================

class TextNormalizer:
    @staticmethod
    def clean(text: str) -> str:
        if not isinstance(text, str): return ""
        t = text.lower()
        for o, n in [('ä', 'ae'), ('ö', 'oe'), ('ü', 'ue'), ('ß', 'ss')]: t = t.replace(o, n)
        return re.sub(r'[^a-z0-9]', '', t).strip()

class WagnerFischerDistance:
    """DP Algorithmus für Damerau-Levenshtein Distanz."""
    @staticmethod
    def calculate_ratio(s1: str, s2: str) -> float:
        if s1 == s2: return 100.0
        if not s1 or not s2: return 0.0
        
        len1, len2 = len(s1), len(s2)
        d = [[0 for _ in range(len2 + 1)] for _ in range(len1 + 1)]
        for i in range(1, len1 + 1): d[i][0] = i
        for j in range(1, len2 + 1): d[0][j] = j
        
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
                if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                    d[i][j] = min(d[i][j], d[i - 2][j - 2] + cost)
                    
        max_len = max(len1, len2)
        return ((max_len - d[len1][len2]) / max_len) * 100.0

class ColognePhonetics:
    @staticmethod
    def encode(s: str) -> str:
        s = TextNormalizer.clean(s).replace(" ", "")
        if not s: return ""
        
        code = ""
        for i, c in enumerate(s):
            val = ""
            next_c = s[i+1] if i < len(s)-1 else ""
            prev_c = s[i-1] if i > 0 else ""
            
            if c in 'aeiouy': val = "0"
            elif c in 'bp': val = "1" if c == 'b' or next_c != 'h' else "3"
            elif c in 'dt': val = "2" if next_c not in 'csz' else "8"
            elif c in 'fvw': val = "3"
            elif c in 'gkqs': val = "4"
            elif c == 'c': val = "4" if (i == 0 and next_c in 'ahkloqrux') or (i > 0 and next_c in 'ahkoqux' and prev_c not in 'sz') else "8"
            elif c == 'x': val = "48"
            elif c == 'l': val = "5"
            elif c in 'mn': val = "6"
            elif c == 'r': val = "7"
            elif c in 'sz': val = "8"
            
            if not code or val != code[-1]: code += val
        
        if len(code) > 0:
            first = code[0]
            code = code.replace("0", "")
            if first == "0": code = "0" + code
        return code

class SearchKernel:
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SearchKernel, cls).__new__(cls)
                cls._instance._init_index()
        return cls._instance

    def _init_index(self):
        self.stock = {}
        self.index_exact = [] 
        self.load_from_json()

    def _build_search_index(self):
        self.index_exact = []
        for node, items in self.stock.items():
            if node in ['vB12', 'vB11']: continue 
            for item in items:
                n_norm = TextNormalizer.clean(item['name'])
                self.index_exact.append({
                    'node': node, 'item': item, 
                    'name_norm': n_norm,
                    'brand_norm': TextNormalizer.clean(item['brand']),
                    'phonetic': ColognePhonetics.encode(n_norm)
                })

    @execution_profiler
    def find_product(self, search_text: str) -> Optional[Tuple[str, str, str]]:
        if not search_text or len(search_text.strip()) < 2: return None
        q = TextNormalizer.clean(search_text)
        q_phonetic = ColognePhonetics.encode(q)
        
        best_score = 0.0
        best_match = None
        
        for entry in self.index_exact:
            name_db = entry['name_norm']
            brand_db = entry['brand_norm']
            
            if q in name_db or q in brand_db:
                return entry['node'], entry['item']['name'], entry['item']['brand']
                
            if q_phonetic and q_phonetic == entry['phonetic']:
                return entry['node'], entry['item']['name'], entry['item']['brand']
                
            score = WagnerFischerDistance.calculate_ratio(q, name_db)
            if score > best_score:
                best_score = score
                best_match = entry

        if best_score >= CONFIG.FUZZY_MATCH_THRESHOLD and best_match:
            return best_match['node'], best_match['item']['name'], best_match['item']['brand']
        return None

    def load_from_json(self):
        try:
            if not os.path.exists(CONFIG.DB_FILE): raise FileNotFoundError
            with open(CONFIG.DB_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.stock = {}
                for node, items in data.items():
                    self.stock[node] = [{'name': i['name'], 'brand': i['brand'], 'category': i.get('category', 'Sonstiges')} for i in items]
            self._build_search_index()
        except Exception:
            self.stock = {}
            self._build_search_index()

    def save_to_json(self):
        try:
            with open(CONFIG.DB_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.stock, f, indent=2, ensure_ascii=False)
        except Exception as e: log.error(f"IO Error DB: {e}")

    def get_items_text(self, node: str) -> str:
        if node in ['vK1', 'vK2', 'vK3']: return f"<b style='color:red'>Kasse {node}</b>"
        if node not in self.stock or not self.stock[node]: return f"<i>Knoten {node} (Leer)</i>"
        
        text = f"<b style='font-size:14px; text-decoration:underline'>Knoten {node}</b><br><br>"
        for i in self.stock[node]: 
            text += f"• <b>{i['name']}</b> <span style='color:#666'>({i['brand']})</span><br>"
        return text

inv_manager = SearchKernel()

# =============================================================================
# IV. MLOps: COMPOUND PARSER & PIPELINE
# =============================================================================

class MLOpsEngine:
    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MLOpsEngine, cls).__new__(cls)
                cls._instance._init_system()
        return cls._instance

    def _init_system(self):
        self.is_loaded = False
        self.pipeline = None
        self.label_encoder = LabelEncoder()
        
        self.anchors = {
            "rind": "Fleischtheke", "hack": "Fleischtheke", "hackfleisch": "Fleischtheke",
            "schwein": "Fleischtheke", "schnitzel": "Fleischtheke", "hähnchen": "Fleischtheke", "steak": "Fleischtheke", "fleisch": "Fleischtheke",
            "wurst": "Fisch & Wurstwaren", "lachs": "Fisch & Wurstwaren", "fisch": "Fisch & Wurstwaren", "salami": "Fisch & Wurstwaren", "schinken": "Fisch & Wurstwaren",
            "batterie": "Sonstiges (Kasse)", "batterien": "Sonstiges (Kasse)", "akku": "Sonstiges (Kasse)",
            "kaugummi": "Sonstiges (Kasse)", "feuerzeug": "Sonstiges (Kasse)", "gutschein": "Sonstiges (Kasse)", "mentos": "Sonstiges (Kasse)",
            "milch": "Kühlregal (Molkerei)", "joghurt": "Kühlregal (Molkerei)", "quark": "Kühlregal (Molkerei)", "butter": "Kühlregal (Molkerei)", "sahne": "Kühlregal (Molkerei)",
            "käse": "Kühlregal (Vegan & Käse)", "gouda": "Kühlregal (Vegan & Käse)", "vegan": "Kühlregal (Vegan & Käse)", "tofu": "Kühlregal (Vegan & Käse)", 
            "brot": "Backwaren", "brötchen": "Backwaren", "kuchen": "Backwaren", "toast": "Backwaren", "plunder": "Backwaren", "baguette": "Backwaren", "croissant": "Backwaren", "brezel": "Backwaren",
            "apfel": "Obst & Gemüse", "banane": "Obst & Gemüse", "tomate": "Obst & Gemüse", "salat": "Obst & Gemüse", "gemüse": "Obst & Gemüse", "obst": "Obst & Gemüse", "kartoffel": "Obst & Gemüse",
            "wasser": "Getränke Alkoholfrei", "cola": "Getränke Alkoholfrei", "saft": "Getränke Alkoholfrei", "schorle": "Getränke Alkoholfrei", "limo": "Getränke Alkoholfrei",
            "bier": "Bier & Wein", "wein": "Bier & Wein", "pils": "Bier & Wein", "sekt": "Bier & Wein", "weizen": "Bier & Wein",
            "printe": "Backwaren", "printen": "Backwaren", "gebäck": "Backwaren", "brownie": "Backwaren", "brownies": "Backwaren", "torte": "Backwaren",
            "mehl": "Gewürze & Backzutaten", "salz": "Gewürze & Backzutaten", "pfeffer": "Gewürze & Backzutaten", "öl": "Gewürze & Backzutaten", "essig": "Gewürze & Backzutaten", "gewürz": "Gewürze & Backzutaten",
            "schokolade": "Süßwaren & Snacks", "chips": "Süßwaren & Snacks", "haribo": "Süßwaren & Snacks", "keks": "Süßwaren & Snacks", "kekse": "Süßwaren & Snacks", "nuss": "Süßwaren & Snacks", "snack": "Süßwaren & Snacks",
            "pizza": "Tiefkühlware", "eis": "Tiefkühlware", "spinat": "Tiefkühlware", "pommes": "Tiefkühlware", "tiefkühl": "Tiefkühlware",
            "kaffee": "Kaffee & Tee", "tee": "Kaffee & Tee", "espresso": "Kaffee & Tee",
            "nudel": "Trockensortiment & Konserven", "nudeln": "Trockensortiment & Konserven", "pasta": "Trockensortiment & Konserven", "reis": "Trockensortiment & Konserven", "suppe": "Trockensortiment & Konserven", "dose": "Trockensortiment & Konserven",
            "papier": "Drogerie & Haushalt", "zahnpasta": "Drogerie & Haushalt", "duschgel": "Drogerie & Haushalt", "seife": "Drogerie & Haushalt", "waschmittel": "Drogerie & Haushalt", "shampoo": "Drogerie & Haushalt",
            "protein": "Fitness & Sport", "energy": "Fitness & Sport", "riegel": "Fitness & Sport"
        }
        self._sync_model()

    def _map_raw_category(self, raw: str, name: str) -> str:
        r, n = str(raw).lower(), str(name).lower()
        if "getränk" in r or "wein" in r:
            if any(w in n for w in ["vodka", "wodka", "gin", "rum", "whiskey"]): return "Spirituosenschrank"
            if any(w in n for w in ["bier", "wein", "sekt", "pils"]): return "Bier & Wein"
            return "Getränke Alkoholfrei"
        if "fleisch" in r or "wurst" in r or "fisch" in r:
            if any(w in n for w in ["salami", "schinken", "wurst", "fisch", "lachs", "forelle"]): return "Fisch & Wurstwaren"
            return "Fleischtheke"
        if "kühlregal" in r:
            if any(w in n for w in ["käse", "gouda", "tofu", "vegan"]): return "Kühlregal (Vegan & Käse)"
            return "Kühlregal (Molkerei)"
        if "backwaren" in r or "frühstück" in r or "gebäck" in r:
            if any(w in n for w in ["kaffee", "tee"]): return "Kaffee & Tee"
            return "Backwaren"
        if "süßwaren" in r or "snacks" in r: return "Süßwaren & Snacks"
        if "obst" in r or "gemüse" in r: return "Obst & Gemüse"
        if "tiefkühl" in r: return "Tiefkühlware"
        if "gewürze" in r or "öle" in r or "backzutaten" in r: return "Gewürze & Backzutaten"
        if "drogerie" in r or "haushalt" in r: return "Drogerie & Haushalt"
        return "Trockensortiment & Konserven"

    def _sync_model(self):
        if os.path.exists(CONFIG.ML_MODEL_FILE):
            try:
                dump = joblib.load(CONFIG.ML_MODEL_FILE)
                self.pipeline = dump['pipeline']
                self.label_encoder = dump['encoder']
                self.is_loaded = True
                return
            except Exception: pass

        if not os.path.exists(CONFIG.TRAINING_CSV): return

        try:
            # FIX: Macht Spaltennamen zu Kleinbuchstaben, entfernt den KeyError
            df = pd.read_csv(CONFIG.TRAINING_CSV, encoding='utf-8', on_bad_lines='skip')
            df.columns = df.columns.str.strip().str.lower()
            
            name_col = 'product_name' if 'product_name' in df.columns else 'name'
            
            # Checkt auf beide Varianten, die bei dir im Log aufgetaucht sind
            if 'supermarket_category' in df.columns:
                cat_col = 'supermarket_category'
            elif 'category' in df.columns:
                cat_col = 'category'
            else:
                log.error(f"Konnte Trainings-Spalten nicht finden. Vorhanden: {list(df.columns)}")
                return

            df = df.dropna(subset=[name_col, cat_col])
            X_raw = df[name_col].astype(str).apply(TextNormalizer.clean).tolist()
            y_raw = df[cat_col].astype(str).tolist()
            
            mapped_y = [self._map_raw_category(c, n) for c, n in zip(y_raw, X_raw)]
            y_enc = self.label_encoder.fit_transform(mapped_y)
            
            self.pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(analyzer='word', ngram_range=CONFIG.TFIDF_NGRAM_RANGE, min_df=1)),
                ('clf', LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42))
            ])
            
            self.pipeline.fit(X_raw, y_enc)
            joblib.dump({'pipeline': self.pipeline, 'encoder': self.label_encoder}, CONFIG.ML_MODEL_FILE)
            self.is_loaded = True
        except Exception as e: 
            log.error(f"MLOps Training Crash: {e}")

    def _get_balanced_node(self, category: str, item_name: str) -> str:
        nodes = CATEGORY_ROUTING.get(category, ["vW1"])
        if not nodes: return "vW1"
        h = int(hashlib.md5(item_name.encode('utf-8')).hexdigest(), 16)
        return nodes[h % len(nodes)]

    def _get_compound_anchor(self, text: str) -> Optional[str]:
        t = text.lower()
        tokens = re.split(r'[^a-zäöüß]+', t)
        sorted_anchors = sorted(self.anchors.keys(), key=len, reverse=True)
        for token in tokens:
            if not token: continue
            for kw in sorted_anchors:
                if token == kw or token.endswith(kw): return self.anchors[kw]
        return None

    @lru_cache(maxsize=4096)
    def predict(self, text: str) -> Tuple[str, str, float]:
        if not text: return "Sonstiges (Kasse)", "vW1", 0.0
        clean = TextNormalizer.clean(text)
        
        anchor_cat = self._get_compound_anchor(text)
        if anchor_cat:
            return anchor_cat, self._get_balanced_node(anchor_cat, clean), 0.99
                
        if self.is_loaded and self.pipeline:
            try:
                probs = self.pipeline.predict_proba([clean])[0]
                max_idx = np.argmax(probs)
                conf = float(probs[max_idx])
                cat = self.label_encoder.inverse_transform([max_idx])[0]
                
                if conf < CONFIG.MIN_ML_CONFIDENCE: 
                    return "Sonstiges (Kasse)", self._get_balanced_node("Sonstiges (Kasse)", clean), conf
                return cat, self._get_balanced_node(cat, clean), conf
            except Exception: pass
            
        return "Sonstiges (Kasse)", "vW1", 0.0

    @execution_profiler
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, str, float]]:
        if not texts: return []
        if not self.is_loaded or not self.pipeline:
            return [("Sonstiges (Kasse)", "vW1", 0.0) for _ in texts]
            
        clean_texts = [TextNormalizer.clean(t) for t in texts]
        results = []
        
        try:
            probs_matrix = self.pipeline.predict_proba(clean_texts)
            max_indices = np.argmax(probs_matrix, axis=1)
            confs = np.max(probs_matrix, axis=1)
            cats = self.label_encoder.inverse_transform(max_indices)
            
            for i, raw_text in enumerate(texts):
                clean = clean_texts[i]
                anchor_cat = self._get_compound_anchor(raw_text)
                
                if anchor_cat:
                    results.append((anchor_cat, self._get_balanced_node(anchor_cat, clean), 0.99))
                else:
                    if confs[i] < CONFIG.MIN_ML_CONFIDENCE:
                        results.append(("Sonstiges (Kasse)", self._get_balanced_node("Sonstiges (Kasse)", clean), confs[i]))
                    else:
                        results.append((cats[i], self._get_balanced_node(cats[i], clean), confs[i]))
        except Exception as e:
            return [("Sonstiges (Kasse)", "vW1", 0.0) for _ in texts]
            
        return results

ml_predictor = MLOpsEngine()

# =============================================================================
# V. STOCHASTIC QUEUING THEORY: M/M/1/K REALISTIC MARKOV MODEL
# =============================================================================

class EnterpriseQueuingModel:
    """Implementiert die Markow-Ketten Simulation M/M/1/K für individuelle Kassen."""
    @staticmethod
    def calculate_wait_metrics(base_lambda: float, current_hour: int, checkout_id: str) -> Dict[str, float]:
        c = 1 
        mu = 1.5  
        K = 10    
        
        variation = 0.0
        if checkout_id == 'vK1': variation = 0.8  
        elif checkout_id == 'vK2': variation = 0.2
        elif checkout_id == 'vK3': variation = -0.4 
        
        time_factor = math.sin((current_hour - 8) / 12 * math.pi) * 1.5
        if time_factor < 0: time_factor = 0
        
        lam = max(0.2, base_lambda + variation + time_factor)
        rho = lam / mu
        
        if rho == 1.0:
            p0 = 1.0 / (K + 1)
            pk = 1.0 / (K + 1)
            lq = (K * (K - 1)) / (2 * (K + 1))
        else:
            p0 = (1 - rho) / (1 - rho**(K + 1))
            pk = (rho**K) * p0
            lq = (rho / (1 - rho)) - ((K + 1) * rho**(K + 1) / (1 - rho**(K + 1)))
            
        lambda_eff = lam * (1 - pk)
        wait_minutes = lq / lambda_eff if lambda_eff > 0 else 0.0
        wait_sec = (wait_minutes * 60.0) + random.uniform(1.0, 5.0)
        
        return {
            "wait_sec": wait_sec, 
            "p_wait": 1.0 - p0,
            "lq": lq,
            "p_loss": pk
        }

class TrafficPredictor:
    """Fallback Klassse (wird durch app.py überschrieben, bleibt hier als Backup)"""
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TrafficPredictor, cls).__new__(cls)
                cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        if os.path.exists(CONFIG.TRAFFIC_MODEL_FILE):
            try:
                data = pickle.load(open(CONFIG.TRAFFIC_MODEL_FILE, 'rb'))
                self.model = data['model']
                self.encoder = data['encoder']
                self.features = data.get('features', [])
                self.edge_map = {l: i for i, l in enumerate(self.encoder.classes_)}
            except Exception: self.model = None
        else: self.model = None

    def predict_load(self, u: str, v: str, dt: datetime) -> float:
        if self.model is None: return 1.5 if 16 <= dt.hour <= 19 else 0.5
        edge_key = f"{min(u, v)}-{max(u, v)}"
        if edge_key not in self.edge_map: return 0.0
        
        hf = dt.hour + dt.minute / 60.0
        bl = 10 + 80*np.exp(-0.2*(hf-10.5)**2) + 150*np.exp(-0.15*(hf-17.5)**2)
        
        df = pd.DataFrame([{
            'edge_id_enc': self.edge_map[edge_key], 'is_holiday': 0, 'is_weekend': 1 if dt.weekday() >= 4 else 0, 
            'is_rush_hour': 1 if 16 <= dt.hour <= 19 else 0, 'total_agents': max(10, bl),
            'hours_to_close': 20 - dt.hour, 'weekend_rush': 0, 
            'hour_sin': 0, 'hour_cos': 0, 'min_sin': 0, 'min_cos': 0, 'day_sin': 0, 'day_cos': 0
        }])
        
        for c in self.features:
            if c not in df.columns: df[c] = 0
            
        return max(0.0, np.expm1(self.model.predict(df[self.features])[0]))

    def get_congested_graph(self, G_base: nx.Graph, dt: datetime) -> nx.Graph:
        G_new = G_base.copy()
        for u, v, data in G_new.edges(data=True):
            bw = data.get('weight', 1.0)
            pl = self.predict_load(u, v, dt)
            penalty = 1.0 + (pl * 0.5) if pl > 0.8 else 1.0
            data['predicted_load'] = pl
            data['weight'] = bw * (10.0 if pl > 6.0 else penalty)
        return G_new

predictor = TrafficPredictor()

# =============================================================================
# VI. OPERATIONS RESEARCH: METAHEURISTICS STRATEGY PATTERN
# =============================================================================

class RoutingStrategy(ABC):
    @abstractmethod
    def solve(self, dist_matrix: dict, start: str, targets: List[str], end: Optional[str]) -> Tuple[List[str], str]: 
        pass

class NearestNeighborSolver(RoutingStrategy):
    def solve(self, dist_matrix: dict, start: str, targets: List[str], end: Optional[str]) -> Tuple[List[str], str]:
        path = [start]
        unv = set(targets)
        curr = start
        while unv:
            nxt = min(unv, key=lambda n: (dist_matrix.get((curr, n), float('inf')), n))
            path.append(nxt)
            unv.remove(nxt)
            curr = nxt
        if end: path.append(end)
        return path, "Nearest Neighbor"

class HeldKarpDPSolver(RoutingStrategy):
    def solve(self, dist_matrix: dict, start: str, targets: List[str], end: Optional[str]) -> Tuple[List[str], str]:
        n = len(targets)
        if n == 0: return [start] + ([end] if end else []), "Trivial"
        if n > CONFIG.DP_EXACT_LIMIT: return NearestNeighborSolver().solve(dist_matrix, start, targets, end)
        
        memo = {}
        def dp(curr: str, unvisited: frozenset) -> Tuple[float, List[str]]:
            if not unvisited:
                if end: return dist_matrix.get((curr, end), float('inf')), [end]
                return 0.0, []
            
            state = (curr, unvisited)
            if state in memo: return memo[state]
            
            min_cost = float('inf')
            best_path = []
            
            for nxt in sorted(list(unvisited)):
                cost = dist_matrix.get((curr, nxt), float('inf'))
                rem_cost, rem_path = dp(nxt, unvisited - frozenset([nxt]))
                if cost + rem_cost < min_cost:
                    min_cost = cost + rem_cost
                    best_path = [nxt] + rem_path
                    
            memo[state] = (min_cost, best_path)
            return memo[state]
            
        _, path = dp(start, frozenset(targets))
        return [start] + path, "Held-Karp Exakt-Optimum"

class SimulatedAnnealingSolver(RoutingStrategy):
    def solve(self, dist_matrix: dict, start: str, targets: List[str], end: Optional[str]) -> Tuple[List[str], str]:
        if not targets: return [start] + ([end] if end else []), "Trivial"
        
        cp = NearestNeighborSolver().solve(dist_matrix, start, targets, None)[0][1:] 
        
        def cost(p): 
            c = dist_matrix.get((start, p[0]), float('inf'))
            for i in range(len(p)-1): c += dist_matrix.get((p[i], p[i+1]), float('inf'))
            if end: c += dist_matrix.get((p[-1], end), float('inf'))
            return c
            
        cc = cost(cp)
        bp, bc = cp.copy(), cc
        temp = CONFIG.SA_START_TEMP
        
        for _ in range(CONFIG.SA_MAX_ITERATIONS):
            if temp < CONFIG.SA_MIN_TEMP: break
            
            i1, i2 = sorted(random.sample(range(len(cp)), 2))
            np_path = cp.copy()
            np_path[i1:i2] = reversed(np_path[i1:i2]) 
            nc = cost(np_path)
            
            if nc < cc or random.random() < math.exp((cc - nc) / temp):
                cp, cc = np_path, nc
                if cc < bc: 
                    bc, bp = cc, cp.copy()
            temp *= CONFIG.SA_COOLING_RATE
            
        return [start] + bp + ([end] if end else []), "Simulated Annealing"

class Chromosome:
    __slots__ = ['route', 'fitness', 'distance']
    def __init__(self, route: List[str]):
        self.route = route
        self.fitness = 0.0
        self.distance = 0.0
        
    def evaluate(self, dist_matrix: dict, start: str, end: Optional[str]):
        d = dist_matrix.get((start, self.route[0]), 1e6)
        for i in range(len(self.route)-1):
            d += dist_matrix.get((self.route[i], self.route[i+1]), 1e6)
        if end:
            d += dist_matrix.get((self.route[-1], end), 1e6)
        self.distance = d
        self.fitness = 1.0 / (d + 1e-6)

class GeneticAlgorithmSolver(RoutingStrategy):
    def solve(self, dist_matrix: dict, start: str, targets: List[str], end: Optional[str]) -> Tuple[List[str], str]:
        if not targets: return [start] + ([end] if end else []), "Trivial"
        
        population: List[Chromosome] = []
        for _ in range(CONFIG.GA_POPULATION_SIZE):
            route = targets.copy()
            random.shuffle(route)
            population.append(Chromosome(route))
            
        for chrom in population: chrom.evaluate(dist_matrix, start, end)
            
        elite_size = int(CONFIG.GA_POPULATION_SIZE * CONFIG.GA_ELITISM_RATIO)
        
        for generation in range(CONFIG.GA_MAX_GENERATIONS):
            population.sort(key=lambda x: x.fitness, reverse=True)
            new_generation = population[:elite_size]
            
            while len(new_generation) < CONFIG.GA_POPULATION_SIZE:
                p1 = self._tournament_selection(population)
                p2 = self._tournament_selection(population)
                
                child_route = self._partially_mapped_crossover(p1.route, p2.route)
                self._inversion_mutation(child_route)
                
                child = Chromosome(child_route)
                child.evaluate(dist_matrix, start, end)
                new_generation.append(child)
                
            population = new_generation
            
        best = max(population, key=lambda x: x.fitness)
        return [start] + best.route + ([end] if end else []), f"Genetic Algorithm (Gen={CONFIG.GA_MAX_GENERATIONS})"

    def _tournament_selection(self, population: List[Chromosome], k: int = 5) -> Chromosome:
        return max(random.sample(population, k), key=lambda x: x.fitness)
        
    def _partially_mapped_crossover(self, parent1: List[str], parent2: List[str]) -> List[str]:
        size = len(parent1)
        child = [None] * size
        c1, c2 = sorted(random.sample(range(size), 2))
        child[c1:c2] = parent1[c1:c2]
        
        for i in range(c1, c2):
            if parent2[i] not in child:
                pos = i
                while c1 <= pos < c2:
                    pos = parent2.index(parent1[pos])
                child[pos] = parent2[i]
                
        for i in range(size):
            if child[i] is None:
                child[i] = parent2[i]
        return child

    def _inversion_mutation(self, route: List[str]):
        if random.random() < CONFIG.GA_MUTATION_RATE:
            m1, m2 = sorted(random.sample(range(len(route)), 2))
            route[m1:m2] = reversed(route[m1:m2])

class AntColonySolver(RoutingStrategy):
    def solve(self, dist_matrix: dict, start: str, targets: List[str], end: Optional[str]) -> Tuple[List[str], str]:
        all_nodes = [start] + targets + ([end] if end else [])
        pheromones = {(u, v): 1.0 for u in all_nodes for v in all_nodes if u != v}
        best_path, best_cost = None, float('inf')
        
        for iteration in range(CONFIG.ACO_ITERATIONS):
            ant_paths = []
            for ant in range(CONFIG.ACO_NUM_ANTS):
                path, cost, curr = [start], 0.0, start
                unvisited = set(targets)
                while unvisited:
                    probs = []
                    for v in unvisited:
                        dist = dist_matrix.get((curr, v), 1e-4)
                        prob = pheromones.get((curr, v), 1.0) ** CONFIG.ACO_ALPHA * ((1.0 / dist)**CONFIG.ACO_BETA)
                        probs.append((v, prob))
                        
                    tot_p = sum(pr for _, pr in probs)
                    if tot_p > 0:
                        nxt = random.choices([v for v, _ in probs], weights=[pr/tot_p for _, pr in probs])[0] 
                    else:
                        nxt = random.choice(list(unvisited))
                        
                    cost += dist_matrix.get((curr, nxt), float('inf'))
                    path.append(nxt); unvisited.remove(nxt); curr = nxt
                    
                if end:
                    cost += dist_matrix.get((curr, end), float('inf'))
                    path.append(end)
                    
                ant_paths.append((path, cost))
                if cost < best_cost:
                    best_cost = cost
                    best_path = path
                    
            for k in pheromones: 
                pheromones[k] *= (1.0 - CONFIG.ACO_EVAPORATION)
            
            ant_paths.sort(key=lambda x: x[1])
            for p, d in ant_paths[:10]:
                deposit = 100.0 / (d + 1e-6)
                for i in range(len(p) - 1):
                    pheromones[(p[i], p[i+1])] += deposit
                    
        return best_path, f"Ant Colony System (Ameisen={CONFIG.ACO_NUM_ANTS})"

# =============================================================================
# VII. THE MASTER ORCHESTRATOR
# =============================================================================

@execution_profiler
def calculate_hybrid_route(graph: nx.Graph, start_node: str, shopping_nodes: List[str], end_node: str, current_time: datetime = None):
    if current_time is None: current_time = datetime.now()
    if start_node not in graph or end_node not in graph: return [], [], 0, [], ["Graph Exception: Boundaries invalid."]

    targets = sorted(list(set([t for t in shopping_nodes if t in graph.nodes and t != start_node and t != end_node])))
    
    # Deterministischer Seed für Metaheuristiken basierend auf den Knoten
    seed_str = "".join(targets)
    random.seed(int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32))

    def filter_end(n): return n != end_node
    G_shop = nx.subgraph_view(graph, filter_node=filter_end)
    valid_targets = [t for t in targets if t in G_shop.nodes]
    notes = []

    px = [topology.nodes_config[start_node]['pos'][0]] if start_node in topology.nodes_config else []
    py = [topology.nodes_config[start_node]['pos'][1]] if start_node in topology.nodes_config else []
    tot_dist = 0

    if not valid_targets:
        try:
            path = nx.dijkstra_path(graph, start_node, end_node, weight='weight')
            dist = nx.dijkstra_path_length(graph, start_node, end_node, weight='weight')
            return [topology.nodes_config[n]['pos'][0] for n in path], [topology.nodes_config[n]['pos'][1] for n in path], dist, path, notes
        except: return [], [], 0, [], notes

    queue_nodes = {'vW1', 'vW2', 'vW3'}
    store_t = [t for t in valid_targets if t not in queue_nodes]
    queue_t = [t for t in valid_targets if t in queue_nodes]
    
    d_mat, p_mat = {}, {}
    rel_nodes = [start_node] + valid_targets
    for u in rel_nodes:
        try:
            l, p = nx.single_source_dijkstra(G_shop, u, weight='weight')
            for v in rel_nodes:
                if u != v and v in l: 
                    d_mat[(u, v)], p_mat[(u, v)] = l[v], p[v]
                elif u != v: 
                    d_mat[(u, v)] = float('inf')
        except: pass

    if store_t:
        n_targets = len(store_t)
        if n_targets <= CONFIG.DP_EXACT_LIMIT:
            solver = HeldKarpDPSolver()
        elif n_targets > 25:
            solver = GeneticAlgorithmSolver()
        elif n_targets > CONFIG.SA_THRESHOLD: 
            solver = AntColonySolver()
        else:
            solver = SimulatedAnnealingSolver()
            
        store_seq, msg = solver.solve(d_mat, start_node, store_t, None)
        notes.append(msg)
    else:
        store_seq = [start_node]
        
    last_store = store_seq[-1]

    ck_map = {'vW1': 'vK1', 'vW2': 'vK2', 'vW3': 'vK3'}
    best_ex_path, min_ex_cost, ch_ck = [], float('inf'), None
    smart = any('predicted_load' in data for _, _, data in graph.edges(data=True))

    if not queue_t:
        for w, ck in ck_map.items():
            if w not in G_shop.nodes or ck not in graph.nodes: continue
            try:
                wt = nx.dijkstra_path_length(G_shop, last_store, w, weight='weight')
                
                base_lam = graph[w][ck].get('predicted_load', 0.5) if smart and graph.has_edge(w, ck) else 0.5
                qp = EnterpriseQueuingModel.calculate_wait_metrics(base_lam, current_time.hour, ck)["wait_sec"]
                
                et = graph[ck][end_node].get('weight', 1.0) if graph.has_edge(ck, end_node) else 2.0
                
                tt = wt + qp + et
                if tt < min_ex_cost:
                    min_ex_cost, ch_ck = tt, ck
                    best_ex_path = nx.dijkstra_path(G_shop, last_store, w, weight='weight') + [ck, end_node]
            except: continue
    else:
        for perm in itertools.permutations(queue_t):
            try:
                tt = d_mat.get((last_store, perm[0]), float('inf'))
                for i in range(len(perm)-1): tt += d_mat.get((perm[i], perm[i+1]), float('inf'))
                fw = perm[-1]
                ck = ck_map[fw]
                
                base_lam = graph[fw][ck].get('predicted_load', 0.5) if smart and graph.has_edge(fw, ck) else 0.5
                qp = EnterpriseQueuingModel.calculate_wait_metrics(base_lam, current_time.hour, ck)["wait_sec"]
                
                et = graph[ck][end_node].get('weight', 1.0) if graph.has_edge(ck, end_node) else 2.0
                tt += qp + et
                
                if tt < min_ex_cost:
                    min_ex_cost, ch_ck = tt, ck
                    ep = nx.dijkstra_path(G_shop, last_store, perm[0], weight='weight')
                    for i in range(len(perm)-1):
                        seg = p_mat.get((perm[i], perm[i+1]), [])
                        if len(seg) > 1: ep.extend(seg[1:])
                    best_ex_path = ep + [ck, end_node]
            except: continue

    if not best_ex_path: best_ex_path = [end_node]

    f_seq = store_seq.copy()
    if queue_t: f_seq.extend(queue_t)
    if ch_ck: f_seq.append(ch_ck)
    f_seq.append(end_node)

    for i in range(len(store_seq) - 1):
        seg = p_mat.get((store_seq[i], store_seq[i+1]), [])
        for k in range(len(seg)-1):
            if graph.has_edge(seg[k], seg[k+1]):
                w = graph[seg[k]][seg[k+1]]['weight']
                if w < 9000: tot_dist += w
        for n in seg[1:]:
            if n in topology.nodes_config:
                px.append(topology.nodes_config[n]['pos'][0])
                py.append(topology.nodes_config[n]['pos'][1])

    if len(best_ex_path) > 1:
        for k in range(len(best_ex_path)-1):
            if graph.has_edge(best_ex_path[k], best_ex_path[k+1]):
                w = graph[best_ex_path[k]][best_ex_path[k+1]]['weight']
                if w < 9000: tot_dist += w
            n = best_ex_path[k+1]
            if n in topology.nodes_config:
                px.append(topology.nodes_config[n]['pos'][0])
                py.append(topology.nodes_config[n]['pos'][1])

    random.seed()

    return px, py, tot_dist, f_seq, notes

# =============================================================================
# VIII. EXPORT LAYER
# =============================================================================

G_base = topology.G_base
nodes_config = topology.nodes_config
edges_config = topology.edges_config
shelves = topology.shelves

COLOR_RED = CONFIG.COLOR_RED
COLOR_BLACK = CONFIG.COLOR_BLACK
COLOR_BLUE = CONFIG.COLOR_BLUE
COLOR_CYAN = CONFIG.COLOR_CYAN
COLOR_ORANGE = CONFIG.COLOR_ORANGE
COLOR_GREEN = CONFIG.COLOR_GREEN
COLOR_SHELF = CONFIG.COLOR_SHELF
COLOR_SHELF_BORDER = CONFIG.COLOR_SHELF_BORDER
COLOR_HIGHLIGHT = CONFIG.COLOR_HIGHLIGHT

class QueuingModelFacade:
    @staticmethod
    def calculate_wait_time(lam: float, h: int = 12, checkout_id: str = "vK1") -> float:
        return EnterpriseQueuingModel.calculate_wait_metrics(lam, h, checkout_id)["wait_sec"]

QueuingModel = QueuingModelFacade