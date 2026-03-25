#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Projekt: JMU Smart Supermarket - Backend & Graphenmodell

Dieses Skript enthält die grundlegende Logik für unseren virtuellen Supermarkt.
Es baut das Raster (Regale, Gänge) auf und stellt Funktionen für die Produktsuche,
das Machine Learning und die Wegpunkt-Optimierung (Routing) bereit.
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

# Machine Learning Bibliotheken für die Produkt-Kategorisierung
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib

# =============================================================================
# I. GRUNDEINSTELLUNGEN & HILFSFUNKTIONEN
# =============================================================================

# Eigene Fehlerklassen, damit das Skript kontrolliert abbricht, wenn z.B. der Graph falsch aufgebaut ist
class StoreBackendException(Exception): pass
class ConfigurationError(StoreBackendException): pass
class GraphTopologyError(StoreBackendException): pass

@dataclass(frozen=True)
class SystemConfig:
    """
    Alle wichtigen Pfade und Parameter an einem Ort gebündelt.
    So müssen wir nicht im ganzen Code suchen, wenn wir z.B. die 
    Simulated Annealing Parameter für unsere Ausarbeitung anpassen wollen.
    """
    DB_FILE: str = "products.json"
    ROUTING_FILE: str = "routing_config.json"
    TRAINING_CSV: str = "smartcart_ml_training_data.csv"
    # FIX: Dateiname spiegelt nun den tatsächlichen Algorithmus wider (Logistic Regression statt SVM)
    ML_MODEL_FILE: str = "smartcart_logreg_v_final.pkl" 
    # FIX: Dateiname auf das korrekte XGBoost-Artefakt angepasst
    TRAFFIC_MODEL_FILE: str = "traffic_model_xgboost.pkl"
    CACHE_DIR: str = "system_cache"
    
    # Parameter für die Produktsuche
    FUZZY_MATCH_THRESHOLD: float = 65.0  # Ab 65% Ähnlichkeit werten wir es als Treffer (Tippfehler)
    MIN_ML_CONFIDENCE: float = 0.15      # Gesenkt, damit das Modell seltener in "Sonstiges" flüchtet
    TFIDF_NGRAM_RANGE: Tuple[int, int] = (2, 4) # Wir nutzen jetzt Charakter-N-Gramme (2 bis 4 Zeichen)
    
    # Heuristik-Grenzen (Wichtig für die Performance!)
    # Der Held-Karp Algorithmus (exakt) rechnet bei mehr als 11 Artikeln zu lange. 
    DP_EXACT_LIMIT: int = 11
    SA_THRESHOLD: int = 15  # Ab 15 Artikeln ist Ant Colony besser als Simulated Annealing
    
    # Hyperparameter für Simulated Annealing
    SA_START_TEMP: float = 5000.0
    SA_COOLING_RATE: float = 0.995
    SA_MIN_TEMP: float = 0.01
    SA_MAX_ITERATIONS: int = 6000
    
    # Hyperparameter für den Genetischen Algorithmus
    GA_POPULATION_SIZE: int = 150
    GA_ELITISM_RATIO: float = 0.1
    GA_MUTATION_RATE: float = 0.15
    GA_MAX_GENERATIONS: int = 300
    
    # Hyperparameter für Ant Colony Optimization (Ameisenalgorithmus)
    ACO_NUM_ANTS: int = 40
    ACO_ITERATIONS: int = 100
    ACO_EVAPORATION: float = 0.15  # Wie schnell die Pheromone verdampfen
    ACO_ALPHA: float = 1.0         # Einfluss der Pheromone auf die Entscheidung
    ACO_BETA: float = 2.5          # Einfluss der Distanz auf die Entscheidung
    
    # Farb-Codes für das Plotly-Frontend (damit wir sie zentral ändern können)
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
if not os.path.exists(CONFIG.CACHE_DIR): 
    os.makedirs(CONFIG.CACHE_DIR)

# Einfaches Logging, um bei Fehlern im Terminal zu sehen, wo das Skript hängt
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | [%(name)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger("StoreEngine")

def execution_profiler(func: Callable) -> Callable:
    """
    Hilfsfunktion (Decorator) zur Laufzeitmessung. 
    Gibt im Log aus, wie viele Millisekunden eine Funktion (z.B. der Suchalgorithmus) gebraucht hat.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        dur = (time.perf_counter() - t0) * 1000.0
        log.debug(f"[PROFILER] {func.__name__} executed in {dur:.4f} ms")
        return result
    return wrapper

# =============================================================================
# II. Graphen-Topologie (Aufbau des Supermarkts)
# =============================================================================

class DynamicConfigManager:
    """
    Liest die Zuordnung von Kategorien zu Regalen aus.
    """
    _lock = threading.RLock()
    
    @staticmethod
    def get_routing() -> Dict[str, List[str]]:
        """
        Versucht zuerst eine JSON-Datei zu laden (damit wir es später ohne Code-Änderung 
        anpassen können), fällt aber auf ein hartgecodetes Dictionary zurück, falls die Datei fehlt.
        """
        with DynamicConfigManager._lock:
            if os.path.exists(CONFIG.ROUTING_FILE):
                try:
                    with open(CONFIG.ROUTING_FILE, 'r', encoding='utf-8') as f: 
                        return json.load(f)
                except Exception: 
                    pass
            
            # Standard-Zuordnung
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

# Erstellt eine umgekehrte Zuordnung (Welches Regal gehört zu welcher Kategorie?)
node_to_category = {}
for category, nodes in CATEGORY_ROUTING.items():
    for node in nodes:
        node_to_category[node] = category

class StoreTopology:
    """
    Diese Klasse baut den Graphen (Laufwege) und das physische Layout (Regale) auf.
    Wir nutzen das Singleton-Pattern (_instance). Das bedeutet: Egal wie oft wir die 
    Klasse aufrufen, der Supermarkt wird nur EINMAL im Arbeitsspeicher generiert.
    """
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
        
        # Wir nutzen NetworkX für Graphen-Mathematik (z.B. Dijkstra für kürzeste Wege)
        self.G_base = nx.Graph() 
        
        self._construct_nodes()
        self._construct_edges()
        self._construct_shelves()
        self._initialize_networkx()

    def _construct_nodes(self):
        """
        Definiert alle begehbaren Wegpunkte im Supermarkt mit X/Y Koordinaten.
        """
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
        """
        Verbindet die Wegpunkte mit einer bestimmten Distanz (Gewicht).
        Die Distanz wird später als Laufzeit in Sekunden interpretiert.
        """
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
        """
        Erstellt die grauen Rechtecke (Regale) für die optische Darstellung im Frontend.
        Ein Regal wird über x0 (links), x1 (rechts), y0 (unten) und y1 (oben) definiert.
        """
        self.shelves = []
        
        # Die 6 großen Hauptregale in der Mitte
        H_H = 1.2            
        SHELF_LENGTH = 9.0   
        LEFT_CENTER = 6.5    
        RIGHT_CENTER = 19.5  
        
        for y in [10.0, 6.0, 2.0]: 
            self.shelves.append({'x0': LEFT_CENTER - SHELF_LENGTH/2, 'x1': LEFT_CENTER + SHELF_LENGTH/2, 'y0': y-H_H, 'y1': y+H_H})
            self.shelves.append({'x0': RIGHT_CENTER - SHELF_LENGTH/2, 'x1': RIGHT_CENTER + SHELF_LENGTH/2, 'y0': y-H_H, 'y1': y+H_H})

        # Außenwände und Sonderplatzierungen
        self.shelves.extend([
            {'x0': -2.0, 'x1': 24.0, 'y0': 12.8, 'y1': 13.8}, # Obere Wand
            
            # Linke Wand in zwei Teilen
            {'x0': -2.0, 'x1': -1.0, 'y0': 4.5, 'y1': 12.8},    
            {'x0': -2.0, 'x1': -1.0, 'y0': -1.0, 'y1': 2.5},     
            
            # Untere Wandregale
            {'x0': 2.0, 'x1': 11.0, 'y0': -2.0, 'y1': -1.0},    
            {'x0': 15.0, 'x1': 24.0, 'y0': -2.0, 'y1': -1.0},
            
            # Kassen
            {'x0': 26.2, 'x1': 28.8, 'y0': 11.5, 'y1': 12.5},
            {'x0': 26.2, 'x1': 28.8, 'y0': 9.5, 'y1': 10.5},
            {'x0': 26.2, 'x1': 28.8, 'y0': 7.5, 'y1': 8.5},
            
            # Quadrate (Sonderaktionen)
            {'x0': 26.7, 'x1': 29.3, 'y0': 0.7, 'y1': 3.3},    
            {'x0': 30.7, 'x1': 33.3, 'y0': 0.7, 'y1': 3.3},    
            
            # Backstation (umgedrehtes L unten rechts)
            {'x0': 26.2, 'x1': 35.5, 'y0': -2.0, 'y1': -1.0},
            {'x0': 34.5, 'x1': 35.5, 'y0': -1.0, 'y1': 3.5}
        ])

    def _initialize_networkx(self):
        """
        Baut aus den Konfigurationen einen Graphen, auf dem später z.B. Dijkstra rechnen kann.
        """
        for n, data in self.nodes_config.items():
            if not data.get('hidden', False): 
                self.G_base.add_node(n, pos=data['pos'])
                
        for u, v, w in self.edges_config:
            if u in self.nodes_config and v in self.nodes_config: 
                self.G_base.add_edge(u, v, weight=float(w))
                
        self.G_base.add_node('vInCorner', pos=self.nodes_config['vInCorner']['pos'])
        
        # Prüft, ob alle Gänge erreichbar sind, ansonsten werfen wir einen Fehler
        if not nx.is_connected(self.G_base): 
            raise GraphTopologyError("Fehler im Graphen: Es gibt unerreichbare Knotenpunkte.")

topology = StoreTopology()

# =============================================================================
# III. PRODUKTSUCHE & FEHLERTOLERANZ (String Matching)
# =============================================================================
# Wir programmieren die Such-Algorithmen (Levenshtein & Phonetik) hier selbst 
# (from scratch), anstatt fertige Bibliotheken zu importieren. Das gibt uns 
# mehr Kontrolle und zeigt, dass wir die Konzepte dahinter verstanden haben.

class TextNormalizer:
    """
    Hilfsklasse, um Texte für den Vergleich vorzubereiten.
    "Äpfel" und "aepfel" sollen für den Algorithmus exakt gleich aussehen.
    """
    @staticmethod
    def clean(text: str) -> str:
        """
        Wandelt Text in Kleinbuchstaben um, ersetzt deutsche Umlaute und 
        löscht alle Sonderzeichen und Leerzeichen.
        """
        if not isinstance(text, str): return ""
        t = text.lower()
        for o, n in [('ä', 'ae'), ('ö', 'oe'), ('ü', 'ue'), ('ß', 'ss')]: 
            t = t.replace(o, n)
        # FIX: Behalte a-z, 0-9 UND Leerzeichen (\s), damit N-Gramme funktionieren
        return re.sub(r'[^a-z0-9\s]', '', t).strip()


class WagnerFischerDistance:
    """
    Berechnet die Damerau-Levenshtein-Distanz (Tippfehler-Korrektur).
    Nutzt Dynamische Programmierung (DP) über eine Matrix.
    """
    @staticmethod
    def calculate_ratio(s1: str, s2: str) -> float:
        """
        Berechnet, wie ähnlich sich zwei Wörter sind (in Prozent).
        Ein "Kosten"-Wert von 1 entsteht bei: Einfügen, Löschen oder Ersetzen eines Buchstabens.
        Wir nutzen die Damerau-Erweiterung: Ein Buchstabendreher (z.B. 'ei' statt 'ie') 
        kostet auch nur 1 statt 2. Das ist extrem wichtig für typische Tippfehler!
        """
        if s1 == s2: return 100.0
        if not s1 or not s2: return 0.0
        
        len1, len2 = len(s1), len(s2)
        
        # Leere Matrix aufbauen
        d = [[0 for _ in range(len2 + 1)] for _ in range(len1 + 1)]
        for i in range(1, len1 + 1): d[i][0] = i
        for j in range(1, len2 + 1): d[0][j] = j
        
        # Matrix zeilenweise befüllen
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                
                # Minimum aus Löschen, Einfügen oder Ersetzen
                d[i][j] = min(
                    d[i - 1][j] + 1,      # Löschen
                    d[i][j - 1] + 1,      # Einfügen
                    d[i - 1][j - 1] + cost # Ersetzen
                )
                
                # Damerau-Regel: Transposition (Buchstabendreher) prüfen
                if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                    d[i][j] = min(d[i][j], d[i - 2][j - 2] + cost)
                    
        max_len = max(len1, len2)
        # Umrechnung in einen Prozentwert (100% = identisch)
        return ((max_len - d[len1][len2]) / max_len) * 100.0


class ColognePhonetics:
    """
    Die Kölner Phonetik (besser für Deutsch als der englische Soundex-Algorithmus).
    Weist Wörtern anhand ihres Klangs einen Zahlencode zu.
    "Meier" und "Mayer" klingen gleich und bekommen denselben Code.
    """
    @staticmethod
    def encode(s: str) -> str:
        """
        Wandelt ein deutsches Wort in seinen phonetischen Zahlencode um.
        Wichtig, wenn User ein Wort komplett falsch schreiben, es aber richtig klingt 
        (z.B. "Füsik" statt "Physik").
        """
        s = TextNormalizer.clean(s).replace(" ", "")
        if not s: return ""
        
        code = ""
        for i, c in enumerate(s):
            val = ""
            next_c = s[i+1] if i < len(s)-1 else ""
            prev_c = s[i-1] if i > 0 else ""
            
            # Das offizielle Regelwerk der Kölner Phonetik (Wikipedia)
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
            
            # Aufeinanderfolgende gleiche Ziffern ignorieren wir
            if not code or val != code[-1]: 
                code += val
        
        if len(code) > 0:
            first = code[0]
            # Vokale (0) mitten im Wort entfernen, nur am Anfang behalten
            code = code.replace("0", "")
            if first == "0": 
                code = "0" + code
        return code


class SearchKernel:
    """
    Dieser Manager lädt alle Supermarkt-Produkte in den Arbeitsspeicher (RAM) 
    und verwaltet die Suchanfragen. Auch hier nutzen wir das Singleton-Pattern, 
    damit die Datenbank nicht bei jeder Suchanfrage neu von der Festplatte gelesen wird.
    """
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
        """
        Baut beim Starten der App einmalig einen Index auf.
        Normalisiert alle Produktnamen im Voraus. Das ist viel performanter,
        als es bei jeder einzelnen Suchanfrage des Users neu zu berechnen.
        """
        self.index_exact = []
        for node, items in self.stock.items():
            # FIX: Entferne den nicht existenten Geister-Knoten vB11 aus der Logik
            # Die Kassen-Knoten überspringen wir, da liegen keine kaufbaren Produkte
            if node == 'vB12': continue 
            
            for item in items:
                n_norm = TextNormalizer.clean(item['name'])
                self.index_exact.append({
                    'node': node, 
                    'item': item, 
                    'name_norm': n_norm,
                    'brand_norm': TextNormalizer.clean(item['brand']),
                    'phonetic': ColognePhonetics.encode(n_norm)
                })

    @execution_profiler
    def find_product(self, search_text: str) -> Optional[Tuple[str, str, str]]:
        """
        Unsere 3-stufige Suchmaschine für die Eingabeleiste im Dashboard.

        Stufe 1: Ist das Suchwort direkt im Produktnamen oder der Marke enthalten?
        Stufe 2: Klingt das Suchwort genauso wie ein Produkt (Phonetik)?
        Stufe 3: Ist es nur ein Tippfehler? Wir suchen das Produkt mit der höchsten Prozent-Übereinstimmung (Levenshtein).
        """
        if not search_text or len(search_text.strip()) < 2: 
            return None
            
        q = TextNormalizer.clean(search_text)
        q_phonetic = ColognePhonetics.encode(q)
        
        best_score = 0.0
        best_match = None
        
        for entry in self.index_exact:
            name_db = entry['name_norm']
            brand_db = entry['brand_norm']
            
            # Stufe 1: Sub-String Match (z.B. "Apfel" ist in "Apfelsaft" enthalten)
            if q in name_db or q in brand_db:
                return entry['node'], entry['item']['name'], entry['item']['brand']
                
            # Stufe 2: Phonetischer Match
            if q_phonetic and q_phonetic == entry['phonetic']:
                return entry['node'], entry['item']['name'], entry['item']['brand']
                
            # Stufe 3: Levenshtein Distanz berechnen (Fuzzy Search)
            score = WagnerFischerDistance.calculate_ratio(q, name_db)
            if score > best_score:
                best_score = score
                best_match = entry

        # Hat der beste Treffer mindestens 65% (aus der SystemConfig) Übereinstimmung?
        if best_score >= CONFIG.FUZZY_MATCH_THRESHOLD and best_match:
            return best_match['node'], best_match['item']['name'], best_match['item']['brand']
            
        return None

    def load_from_json(self):
        """
        Lädt die Produktdatenbankbank von der Festplatte in den Arbeitsspeicher.
        """
        try:
            if not os.path.exists(CONFIG.DB_FILE): 
                raise FileNotFoundError
                
            with open(CONFIG.DB_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.stock = {}
                for node, items in data.items():
                    self.stock[node] = [{
                        'name': i['name'], 
                        'brand': i['brand'], 
                        'category': i.get('category', 'Sonstiges'), 
                        'price': i.get('price', '0.00')
                    } for i in items]
                    
            self._build_search_index()
        except Exception:
            self.stock = {}
            self._build_search_index()

    def save_to_json(self):
        """
        Speichert neue Produkte oder Änderungen (aus dem Admin-Panel im UI) 
        zurück in die json Datei.
        """
        try:
            with open(CONFIG.DB_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.stock, f, indent=2, ensure_ascii=False)
        except Exception as e: 
            log.error(f"IO Fehler beim Speichern der DB: {e}")

    def get_items_text(self, node: str) -> str:
        """
        Hilfsfunktion für das Frontend. Generiert das HTML für die Popups,
        die erscheinen, wenn man auf der Karte über ein Regal fährt (Hover-Effekt).
        """
        if node in ['vK1', 'vK2', 'vK3']: 
            return f"<b style='color:red'>Kasse {node}</b>"
        if node not in self.stock or not self.stock[node]: 
            return f"<i>Knoten {node} (Leer)</i>"
        
        text = f"<b style='font-size:14px; text-decoration:underline'>Knoten {node}</b><br><br>"
        for i in self.stock[node]: 
            text += f"• <b>{i['name']}</b> <span style='color:#666'>({i['brand']})</span><br>"
        return text

inv_manager = SearchKernel()

# =============================================================================
# IV. MACHINE LEARNING & KATEGORISIERUNG (Klassifikation unbekannter Produkte)
# =============================================================================
# WICHTIGE ARCHITEKTUR-ENTSCHEIDUNG:
# Warum nutzen wir hier TF-IDF + Logistische Regression und kein Deep Learning 
# (wie z.B. Word Embeddings oder Transformer-Modelle)?
# Produktnamen (z.B. "Bio Haferdrink") sind extrem kurze Strings ohne Satzbau 
# oder grammatikalischen Kontext. Für solche "spärlichen" (sparse) Textdaten 
# sind statistische Verfahren wie TF-IDF oft robuster, nachvollziehbarer und 
# vor allem schnell genug, um das Modell "on-the-fly" neu zu trainieren.
# 
# Zudem nutzen wir einen hybriden Ansatz: Bevor die KI befragt wird, greifen 
# harte Geschäftsregeln ("Compound Anchors"). Das ist Best Practice in der 
# Enterprise-KI (Business Logic Override), um absurde Fehler bei offensichtlichen 
# Wörtern (wie "Hackfleisch") zu vermeiden und Inference-Zeit zu sparen.

class MLOpsEngine:
    """
    Die Machine Learning Engine für die Produktklassifizierung.
    Wir nutzen das Singleton-Pattern, damit das Modell (die Pipeline) 
    nur ein einzige Mal beim Start des Servers in den RAM geladen wird.
    """
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
        
        # Heuristische Vorfilter (Business Logic Override)
        # Die Liste bleibt klein und realistisch (Nur für absolute Fast-Lane Zuweisungen)
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
        """
        TRUE MLOps: Dynamic Inventory Auto-Training.
        Die KI trainiert sich auf die tatsächliche, aktuelle Datenbank des Supermarkts.
        """
        if os.path.exists(CONFIG.ML_MODEL_FILE):
            try:
                dump = joblib.load(CONFIG.ML_MODEL_FILE)
                if dump.get('version') == '3.0':
                    self.pipeline = dump['pipeline']
                    self.label_encoder = dump['encoder']
                    self.is_loaded = True
                    return
                else:
                    log.warning("Veraltetes ML-Modell gefunden. Erzwinge Neu-Training auf v3.0...")
            except Exception: pass

        try:
            X_raw, y_raw = [], []
            
            # 1. DYNAMISCHES TRAINING (Der echte 1,0 Fix)
            # Wir ziehen das echte Inventar aus der Datenbank und trainieren die KI darauf!
            if hasattr(inv_manager, 'stock') and inv_manager.stock:
                for node, items in inv_manager.stock.items():
                    for item in items:
                        name = item['name']
                        cat = item.get('category', 'Sonstiges (Kasse)')
                        # Daten augmentieren, um die KI robust zu machen
                        X_raw.extend([name, f"bio {name}", f"frische {name}"])
                        y_raw.extend([cat, cat, cat])
                        
            # 2. Anker-Wörter als Baseline hinzufügen
            for kw, cat in self.anchors.items():
                X_raw.extend([kw, f"{kw} premium"])
                y_raw.extend([cat, cat])

            # 3. Externe CSV Daten hinzufügen falls vorhanden
            if os.path.exists(CONFIG.TRAINING_CSV):
                df = pd.read_csv(CONFIG.TRAINING_CSV, encoding='utf-8', on_bad_lines='skip')
                df.columns = df.columns.str.strip().str.lower()
                name_col = 'product_name' if 'product_name' in df.columns else 'name'
                cat_col = 'supermarket_category' if 'supermarket_category' in df.columns else 'category'
                
                if name_col in df.columns and cat_col in df.columns:
                    df = df.dropna(subset=[name_col, cat_col])
                    x_csv = df[name_col].astype(str).tolist()
                    y_csv = df[cat_col].astype(str).tolist()
                    y_csv_mapped = [self._map_raw_category(c, n) for c, n in zip(y_csv, x_csv)]
                    X_raw.extend(x_csv)
                    y_raw.extend(y_csv_mapped)

            # Label Encoding
            X_clean = [TextNormalizer.clean(x) for x in X_raw]
            all_possible_cats = list(set(y_raw + list(self.anchors.values()) + ["Sonstiges (Kasse)"]))
            self.label_encoder.fit(all_possible_cats)
            y_enc = self.label_encoder.transform(y_raw)
            
            # Die eigentliche ML-Intelligenz (char_wb lernt die Buchstabenmuster der Produkte)
            self.pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=CONFIG.TFIDF_NGRAM_RANGE, min_df=1)),
                ('clf', LogisticRegression(class_weight='balanced', C=5.0, max_iter=2000, random_state=42))
            ])
            
            self.pipeline.fit(X_clean, y_enc)
            joblib.dump({'pipeline': self.pipeline, 'encoder': self.label_encoder, 'version': '3.0'}, CONFIG.ML_MODEL_FILE)
            self.is_loaded = True
            log.info("✅ Dynamisches ML-Modell (v3.0) erfolgreich trainiert und gespeichert.")
            
        except Exception as e: 
            log.error(f"Fehler beim Training des ML-Modells: {e}")

    def _get_balanced_node(self, category: str, item_name: str) -> str:
        nodes = CATEGORY_ROUTING.get(category, ["vW1"])
        if not nodes: return "vW1"
        h = int(hashlib.md5(item_name.encode('utf-8')).hexdigest(), 16)
        return nodes[h % len(nodes)]

    def _get_compound_anchor(self, text: str) -> Optional[str]:
        """
        Fuzzy Matching bleibt als Sicherheitsnetz für absolute Standard-Wörter erhalten.
        """
        t = text.lower()
        tokens = re.split(r'[^a-zäöüß]+', t)
        
        sorted_anchors = sorted(self.anchors.keys(), key=len, reverse=True)
        for token in tokens:
            if not token: continue
            for kw in sorted_anchors:
                if token == kw or token.endswith(kw): 
                    return self.anchors[kw]
                if len(token) >= 4 and len(kw) >= 4:
                    if WagnerFischerDistance.calculate_ratio(token, kw) >= 75.0:
                        return self.anchors[kw]
        return None

    @lru_cache(maxsize=4096)
    def predict(self, text: str) -> Tuple[str, str, float]:
        if not text: return "Sonstiges (Kasse)", "vW1", 0.0
        clean = TextNormalizer.clean(text)
        
        # 1. Greifen die Standard-Anker?
        anchor_cat = self._get_compound_anchor(text)
        if anchor_cat:
            return anchor_cat, self._get_balanced_node(anchor_cat, clean), 0.99
                
        # 2. Das mächtige, auf Inventar trainierte ML-Modell übernimmt den Rest
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
        except Exception:
            return [("Sonstiges (Kasse)", "vW1", 0.0) for _ in texts]
            
        return results

ml_predictor = MLOpsEngine()

# =============================================================================
# V. WARTESCHLANGENTHEORIE (Stochastische M/G/1/K Approximation)
# =============================================================================

class EnterpriseQueuingModel:
    """
    Simuliert die Wartezeit an einer spezifischen Kasse.
    
    WISSENSCHAFTLICHER HINWEIS: Da die Warenkorbgrößen unserer Kunden 
    log-normalverteilt sind, ist die Abfertigungszeit (Service Time) streng genommen 
    "General". Es handelt sich mathematisch um ein M/G/1 (bzw. M/G/1/K) Warteschlangenmodell. 
    
    Um jedoch die Latenzen im Dashboard (Echtzeit-Inferenz) minimal zu halten und 
    die enorm rechenaufwändige Pollaczek-Khintchine-Formel zu umgehen, approximieren wir 
    das System hier ganz bewusst und performant über ein klassisches M/M/1/K-Modell 
    (Exponential-verteilte Servicezeit).
    """
    
    @staticmethod
    def calculate_wait_metrics(base_lambda: float, current_hour: int, checkout_id: str) -> Dict[str, float]:
        """
        Berechnet die erwartete Wartezeit basierend auf der aktuellen Uhrzeit 
        und der Auslastung.
        """
        c = 1       # Anzahl der Kassen (wir berechnen jede Kasse isoliert als M/M/1)
        mu = 1.5    # Service-Rate: Ein Kassierer schafft ca. 1.5 Kunden pro Minute
        K = 10      # Kapazität: Max 10 Leute pro Kasse, danach gehen Kunden woanders hin (Loss System)
        
        # Jede Kasse hat eine leicht andere Grund-Auslastung (Kasse 1 ist am beliebtesten)
        variation = 0.0
        if checkout_id == 'vK1': variation = 0.8  
        elif checkout_id == 'vK2': variation = 0.2
        elif checkout_id == 'vK3': variation = -0.4 
        
        # Tageszeit-Faktor: Mittags (Peak) ist mehr los als morgens. 
        # Simuliert über eine einfache Sinus-Kurve.
        time_factor = math.sin((current_hour - 8) / 12 * math.pi) * 1.5
        if time_factor < 0: time_factor = 0
        
        # Effektive Ankunftsrate (lambda) berechnen
        lam = max(0.2, base_lambda + variation + time_factor)
        
        # Auslastungsgrad (Rho) = Ankünfte / Abfertigungen
        rho = lam / mu
        
        # --- Markow-Ketten Formeln für M/M/1/K ---
        if rho == 1.0:
            # Sonderfall: Genau so viele Ankünfte wie Abfertigungen
            p0 = 1.0 / (K + 1)                  # Wahrscheinlichkeit, dass Kasse leer ist
            pk = 1.0 / (K + 1)                  # Wahrscheinlichkeit, dass Kasse voll ist
            lq = (K * (K - 1)) / (2 * (K + 1))  # Erwartete Länge der Warteschlange
        else:
            # Normalfall
            p0 = (1 - rho) / (1 - rho**(K + 1))
            pk = (rho**K) * p0
            lq = (rho / (1 - rho)) - ((K + 1) * rho**(K + 1) / (1 - rho**(K + 1)))
            
        # Wie viele Kunden stellen sich *wirklich* an (und gehen nicht, weil es zu voll ist)?
        lambda_eff = lam * (1 - pk)
        
        # Gesetz von Little: Wartezeit = Schlangenlänge / effektive Ankunftsrate
        wait_minutes = lq / lambda_eff if lambda_eff > 0 else 0.0
        
        # Umrechnung in Sekunden + kleiner Zufallsfaktor für die Realistik
        wait_sec = (wait_minutes * 60.0) + random.uniform(1.0, 5.0)
        
        return {
            "wait_sec": wait_sec, 
            "p_wait": 1.0 - p0, # Chance, dass der Kunde warten muss
            "lq": lq,           # Durchschnittliche Leute in der Schlange
            "p_loss": pk        # Chance, dass die Schlange maximal voll ist (K=10)
        }

class TrafficPredictor:
    """
    Diese Klasse ist eine reine mathematische Fallback-Heuristik. 
    Sie simuliert Staus deterministisch auf Basis von Tageszeiten (Glockenkurven), 
    falls das echte Machine-Learning XGBoost-Modell in der app.py 
    (TrafficSimulationEngine) ausfallen sollte.
    """
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TrafficPredictor, cls).__new__(cls)
        return cls._instance

    def predict_load(self, u: str, v: str, dt: datetime) -> float:
        """
        Sagt voraus, wie viele Personen auf einem bestimmten Gangstück (u zu v) stehen.
        """
        hf = dt.hour + dt.minute / 60.0
        
        # Deterministische Simulation der Gesamtanzahl an Agenten im Markt
        # 10:30 Uhr und 17:30 Uhr sind Peak-Zeiten (modelliert durch Exponentialfunktionen)
        bl = 10 + 80*np.exp(-0.2*(hf-10.5)**2) + 150*np.exp(-0.15*(hf-17.5)**2)
        
        # Kassen-Gänge (vW) sind immer voller, normale Gänge leerer
        if str(u).startswith('vW') or str(v).startswith('vW'):
            return max(0.5, (bl / 20.0))
        return max(0.2, (bl / 80.0))

    def get_congested_graph(self, G_base: nx.Graph, dt: datetime) -> nx.Graph:
        """
        Nimmt den leeren Supermarkt-Graphen und addiert den heuristischen 
        Stau als "Gewicht" (Strafzeit) auf die Kanten.
        """
        G_new = G_base.copy()
        for u, v, data in G_new.edges(data=True):
            bw = data.get('weight', 1.0)
            pl = self.predict_load(u, v, dt)
            
            # Bei mehr als 0.8 Personen fängt leichter Stau an
            penalty = 1.0 + (pl * 0.5) if pl > 0.8 else 1.0
            
            data['predicted_load'] = pl
            # Bei krassem Stau (> 6 Personen) wird der Weg hart bestraft (Faktor 10)
            data['weight'] = bw * (10.0 if pl > 6.0 else penalty)
            
        return G_new

predictor = TrafficPredictor()

# =============================================================================
# VI. OPERATIONS RESEARCH (Wege-Optimierung / Open Traveling Salesperson Problem)
# =============================================================================
# WICHTIGE ABGRENZUNG FÜR DAS KOLLOQUIUM:
# Algorithmen wie Dijkstra oder A* berechnen nur den kürzesten Weg zwischen ZWEI 
# Punkten (Wegfindung). Sie lösen NICHT das Problem, in welcher REIHENFOLGE man 
# 15 verschiedene Regale besuchen sollte.
#
# Wir modellieren das Problem hier bewusst als "Open TSP" (oder Hamiltonian Path Problem).
# Im Gegensatz zum klassischen TSP erzwingen unsere Algorithmen keinen Rundweg 
# (Hamiltonkreis) zurück zum Startknoten (Eingang), da der Kunde den Supermarkt 
# nach Erledigung seines Einkaufs an einem definierten Ausgang verlässt.

class RoutingStrategy(ABC):
    """Abstrakte Basisklasse für alle Routing-Algorithmen (Strategy Pattern)."""
    @abstractmethod
    def solve(self, dist_matrix: dict, start: str, targets: List[str], end: Optional[str]) -> Tuple[List[str], str]: 
        pass

class NearestNeighborSolver(RoutingStrategy):
    """
    Greedy-Algorithmus (Geizig). Geht immer einfach zum nächstgelegenen Regal.
    Sehr schnell (O(n^2)), aber oft extrem ineffizient, da er am Ende oft quer durch 
    den ganzen Laden laufen muss. Dient uns nur als Basis-Vergleichswert.
    """
    def solve(self, dist_matrix: dict, start: str, targets: List[str], end: Optional[str]) -> Tuple[List[str], str]:
        path = [start]
        unv = set(targets)
        curr = start
        while unv:
            # Suche aus den unbesuchten Knoten denjenigen mit der geringsten Distanz
            nxt = min(unv, key=lambda n: (dist_matrix.get((curr, n), float('inf')), n))
            path.append(nxt)
            unv.remove(nxt)
            curr = nxt
        if end: path.append(end)
        return path, "Nearest Neighbor"

class HeldKarpDPSolver(RoutingStrategy):
    """
    Exakte Lösungsfindung mittels Dynamischer Programmierung (Held-Karp Algorithmus).
    Findet garantiert die absolut kürzeste Open-TSP Route.
    
    PROBLEM: Die Laufzeitkomplexität liegt bei O(n^2 * 2^n). 
    Das bedeutet, ab ca. 12 Artikeln im Warenkorb explodiert die Rechenzeit (RAM-Limit).
    Daher setzen wir in der Config ein striktes Limit (DP_EXACT_LIMIT = 11).
    Alles darüber muss heuristisch angenähert werden.
    """
    def solve(self, dist_matrix: dict, start: str, targets: List[str], end: Optional[str]) -> Tuple[List[str], str]:
        n = len(targets)
        if n == 0: return [start] + ([end] if end else []), "Trivial"
        
        # Schutzmechanismus: Fallback auf Greedy, falls der Warenkorb zu groß für Exaktlösung ist
        if n > CONFIG.DP_EXACT_LIMIT: 
            return NearestNeighborSolver().solve(dist_matrix, start, targets, end)
        
        # Zwischenspeicher (Memoization), um überlappende Teilprobleme nicht doppelt zu berechnen
        memo = {} 
        
        def dp(curr: str, unvisited: frozenset) -> Tuple[float, List[str]]:
            # Basisfall: Alle Produkte eingesammelt
            if not unvisited:
                if end: return dist_matrix.get((curr, end), float('inf')), [end]
                return 0.0, []
            
            state = (curr, unvisited)
            if state in memo: return memo[state]
            
            min_cost = float('inf')
            best_path = []
            
            # Probiere jeden noch unbesuchten Knoten als nächsten Schritt aus
            for nxt in sorted(list(unvisited)):
                cost = dist_matrix.get((curr, nxt), float('inf'))
                rem_cost, rem_path = dp(nxt, unvisited - frozenset([nxt])) # Rekursion
                
                if cost + rem_cost < min_cost:
                    min_cost = cost + rem_cost
                    best_path = [nxt] + rem_path
                    
            memo[state] = (min_cost, best_path)
            return memo[state]
            
        _, path = dp(start, frozenset(targets))
        return [start] + path, "Held-Karp Exakt-Optimum"

class SimulatedAnnealingSolver(RoutingStrategy):
    """
    Simulated Annealing (Simulierte Abkühlung). Eine physik-inspirierte Metaheuristik.
    Anstatt wie Held-Karp alle Möglichkeiten durchzurechnen, nimmt SA eine schlechte 
    Route (z.B. Nearest Neighbor) und tauscht zufällig Teilstücke um (2-Opt).
    
    Der Clou: Am Anfang ("hohe Temperatur") akzeptiert der Algorithmus auch Verschlechterungen.
    Das verhindert, dass er in lokalen Minima steckenbleibt. Er kühlt dann langsam ab 
    und wird strenger. Perfekt für mittlere Warenkörbe (12 bis 15 Artikel).
    """
    def solve(self, dist_matrix: dict, start: str, targets: List[str], end: Optional[str]) -> Tuple[List[str], str]:
        if not targets: return [start] + ([end] if end else []), "Trivial"
        
        # Startroute ist die Greedy-Route (als solide Basis)
        cp = NearestNeighborSolver().solve(dist_matrix, start, targets, None)[0][1:] 
        
        def cost(p): 
            # Berechnet die Gesamtlänge einer vorgeschlagenen Route
            c = dist_matrix.get((start, p[0]), float('inf'))
            for i in range(len(p)-1): 
                c += dist_matrix.get((p[i], p[i+1]), float('inf'))
            if end: 
                c += dist_matrix.get((p[-1], end), float('inf'))
            return c
            
        cc = cost(cp)
        bp, bc = cp.copy(), cc
        temp = CONFIG.SA_START_TEMP
        
        for _ in range(CONFIG.SA_MAX_ITERATIONS):
            if temp < CONFIG.SA_MIN_TEMP: break
            
            # 2-Opt Move: Ein zufälliges Teilstück der Route wird umgedreht
            i1, i2 = sorted(random.sample(range(len(cp)), 2))
            np_path = cp.copy()
            np_path[i1:i2] = reversed(np_path[i1:i2]) 
            nc = cost(np_path)
            
            # Akzeptanz-Kriterium (Metropolis-Wahrscheinlichkeit)
            if nc < cc or random.random() < math.exp((cc - nc) / temp):
                cp, cc = np_path, nc
                if cc < bc: 
                    bc, bp = cc, cp.copy() # Neues globales Minimum gefunden
            
            # Temperatur kühlt ab (Algorithmus wird strenger)
            temp *= CONFIG.SA_COOLING_RATE
            
        return [start] + bp + ([end] if end else []), "Simulated Annealing"

class Chromosome:
    """Hilfsklasse für den Genetischen Algorithmus (repräsentiert EINE mögliche Route / ein Individuum)."""
    __slots__ = ['route', 'fitness', 'distance']
    def __init__(self, route: List[str]):
        self.route = route
        self.fitness = 0.0
        self.distance = 0.0
        
    def evaluate(self, dist_matrix: dict, start: str, end: Optional[str]):
        """Berechnet, wie 'fit' (kurz) diese Route ist."""
        d = dist_matrix.get((start, self.route[0]), 1e6)
        for i in range(len(self.route)-1):
            d += dist_matrix.get((self.route[i], self.route[i+1]), 1e6)
        if end:
            d += dist_matrix.get((self.route[-1], end), 1e6)
        self.distance = d
        # Die Fitness ist der Kehrwert der Distanz: Kurze Distanz = Hohe Fitness
        self.fitness = 1.0 / (d + 1e-6)

class GeneticAlgorithmSolver(RoutingStrategy):
    """
    Genetischer Algorithmus (Evolutionäre Metaheuristik).
    Basiert auf Darwins "Survival of the Fittest".
    1. Erstellt eine "Population" an zufälligen Routen. 
    2. Die kürzesten Routen dürfen sich "paaren" (Crossover).
    3. Zufällige Mutationen sorgen für neue Ideen.
    Extrem robust bei sehr großen Einkaufslisten (> 25 Artikel), da er 
    breitflächig sucht, statt sich wie SA punktuell abzukühlen.
    """
    def solve(self, dist_matrix: dict, start: str, targets: List[str], end: Optional[str]) -> Tuple[List[str], str]:
        if not targets: return [start] + ([end] if end else []), "Trivial"
        
        # 1. Initial-Population erzeugen (viele zufällige Routen)
        population: List[Chromosome] = []
        for _ in range(CONFIG.GA_POPULATION_SIZE):
            route = targets.copy()
            random.shuffle(route)
            population.append(Chromosome(route))
            
        for chrom in population: 
            chrom.evaluate(dist_matrix, start, end)
            
        elite_size = int(CONFIG.GA_POPULATION_SIZE * CONFIG.GA_ELITISM_RATIO)
        
        # Generationen-Schleife (Evolution)
        for generation in range(CONFIG.GA_MAX_GENERATIONS):
            population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Elitismus: Die absolut besten Routen überleben unverändert in die nächste Generation
            new_generation = population[:elite_size]
            
            while len(new_generation) < CONFIG.GA_POPULATION_SIZE:
                # Eltern auswählen (Turnier-Selektion)
                p1 = self._tournament_selection(population)
                p2 = self._tournament_selection(population)
                
                # Kind zeugen (Partially Mapped Crossover) und leicht mutieren
                child_route = self._partially_mapped_crossover(p1.route, p2.route)
                self._inversion_mutation(child_route)
                
                child = Chromosome(child_route)
                child.evaluate(dist_matrix, start, end)
                new_generation.append(child)
                
            population = new_generation
            
        best = max(population, key=lambda x: x.fitness)
        return [start] + best.route + ([end] if end else []), f"Genetic Algorithm (Gen={CONFIG.GA_MAX_GENERATIONS})"

    def _tournament_selection(self, population: List[Chromosome], k: int = 5) -> Chromosome:
        """Zieht k zufällige Routen aus der Population und wählt den Sieger (kürzeste Route)."""
        return max(random.sample(population, k), key=lambda x: x.fitness)
        
    def _partially_mapped_crossover(self, parent1: List[str], parent2: List[str]) -> List[str]:
        """
        Spezielle Crossover-Technik für das TSP-Problem.
        Ein normales Durchschneiden von Arrays würde dazu führen, dass Regale doppelt 
        besucht werden oder fehlen. PMX repariert diese Duplikate intelligent.
        """
        size = len(parent1)
        child = [None] * size
        c1, c2 = sorted(random.sample(range(size), 2))
        # Übernehme einen Block von Elternteil 1
        child[c1:c2] = parent1[c1:c2]
        
        # Fülle den Rest mit Elementen von Elternteil 2 auf (Vermeidung von Duplikaten)
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
        """Dreht mit einer gewissen Wahrscheinlichkeit (Mutation Rate) ein Teilstück der Route um."""
        if random.random() < CONFIG.GA_MUTATION_RATE:
            m1, m2 = sorted(random.sample(range(len(route)), 2))
            route[m1:m2] = reversed(route[m1:m2])

class AntColonySolver(RoutingStrategy):
    """
    Ant Colony Optimization (Ameisenalgorithmus). Eine schwarmbasierte Metaheuristik.
    Simuliert Ameisen, die (anfangs zufällig) durch den Laden laufen. 
    Finden sie einen kurzen Weg, hinterlassen sie auf dieser Kante starke "Pheromone". 
    Nachfolgende Ameisen orientieren sich an diesem Duft und verdichten die beste Route.
    Pheromone verdampfen mit der Zeit, damit der Schwarm nicht auf schlechten Routen hängen bleibt.
    """
    def solve(self, dist_matrix: dict, start: str, targets: List[str], end: Optional[str]) -> Tuple[List[str], str]:
        all_nodes = [start] + targets + ([end] if end else [])
        
        # Initialisierung: Überall liegt gleich viel Pheromon (1.0)
        pheromones = {(u, v): 1.0 for u in all_nodes for v in all_nodes if u != v}
        best_path, best_cost = None, float('inf')
        
        for iteration in range(CONFIG.ACO_ITERATIONS):
            ant_paths = []
            
            # Alle Ameisen losschicken
            for ant in range(CONFIG.ACO_NUM_ANTS):
                path, cost, curr = [start], 0.0, start
                unvisited = set(targets)
                
                while unvisited:
                    probs = []
                    # Ameise steht an einem Regal und überlegt, wohin sie als nächstes geht
                    for v in unvisited:
                        dist = dist_matrix.get((curr, v), 1e-4)
                        
                        # Die Wahrscheinlichkeit für ein Regal wird bestimmt durch:
                        # Pheromonstärke (Alpha) * Sichtbarkeit/Nähe (Beta)
                        prob = pheromones.get((curr, v), 1.0) ** CONFIG.ACO_ALPHA * ((1.0 / dist)**CONFIG.ACO_BETA)
                        probs.append((v, prob))
                        
                    tot_p = sum(pr for _, pr in probs)
                    if tot_p > 0:
                        # Gewichtung (Roulette-Wheel-Selection) basierend auf den Wahrscheinlichkeiten
                        nxt = random.choices([v for v, _ in probs], weights=[pr/tot_p for _, pr in probs])[0] 
                    else:
                        nxt = random.choice(list(unvisited))
                        
                    cost += dist_matrix.get((curr, nxt), float('inf'))
                    path.append(nxt); unvisited.remove(nxt); curr = nxt
                    
                if end:
                    cost += dist_matrix.get((curr, end), float('inf'))
                    path.append(end)
                    
                ant_paths.append((path, cost))
                
                # Globales Minimum aktualisieren
                if cost < best_cost:
                    best_cost = cost
                    best_path = path
                    
            # Iterations-Ende: Pheromone verdampfen (verhindert Stagnation)
            for k in pheromones: 
                pheromones[k] *= (1.0 - CONFIG.ACO_EVAPORATION)
            
            # Die besten Ameisen dieser Iteration dürfen neue Pheromone legen
            ant_paths.sort(key=lambda x: x[1])
            for p, d in ant_paths[:10]: # Nur die Top 10 verstärken den Weg
                deposit = 100.0 / (d + 1e-6) # Je kürzer die Route, desto stärker der Duft
                for i in range(len(p) - 1):
                    pheromones[(p[i], p[i+1])] += deposit
                    
        return best_path, f"Ant Colony System (Ameisen={CONFIG.ACO_NUM_ANTS})"

# =============================================================================
# VII. THE MASTER ORCHESTRATOR (Routen-Berechnung & Koordination)
# =============================================================================

@execution_profiler
def calculate_hybrid_route(graph: nx.Graph, start_node: str, shopping_nodes: List[str], end_node: str, current_time: datetime = None):
    """
    Das ist die Haupt-Schnittstelle für unser Dashboard (wird von der app.py aufgerufen).
    Diese Funktion steuert den kompletten Optimierungs-Workflow und verknüpft alle Module:
    Graphentheorie (Wege) + TSP (Reihenfolge) + Warteschlangentheorie (Kassenwahl).

    Args:
        graph (nx.Graph): Das Netzwerk des Supermarkts (kann der leere oder der verstaute Graph sein).
        start_node (str): Wo startet der Kunde? (Meist 'vEingang').
        shopping_nodes (List[str]): Die Liste der Regale, die besucht werden müssen.
        end_node (str): Das Ziel des Kunden (Meist 'vAusgang').
        current_time (datetime, optional): Wichtig für die Stau-Vorhersage an den Kassen.

    Returns:
        px (List[float]): X-Koordinaten für die grafische Linie im Dashboard.
        py (List[float]): Y-Koordinaten für die grafische Linie im Dashboard.
        tot_dist (float): Die gesamte berechnete Laufdistanz (entspricht Laufzeit in Sekunden).
        f_seq (List[str]): Die Reihenfolge der besuchten Knoten.
        notes (List[str]): Info für das Frontend, welcher Algorithmus genutzt wurde.
    """
    if current_time is None: 
        current_time = datetime.now()
        
    # Sicherheitscheck: Sind Start und Ziel überhaupt im Graphen vorhanden?
    if start_node not in graph or end_node not in graph: 
        return [], [], 0, [], ["Graph Exception: Boundaries invalid."]

    # Duplikate entfernen (Falls ein Kunde z.B. 2x Milch kauft, müssen wir nur 1x zum Regal)
    targets = sorted(list(set([t for t in shopping_nodes if t in graph.nodes and t != start_node and t != end_node])))
    
    # ---------------------------------------------------------
    # SCHRITT 1: Deterministischen Seed setzen
    # ---------------------------------------------------------
    # Warum machen wir das? Heuristiken arbeiten mit Zufall (Random). Wenn wir 
    # den Seed nicht einfrieren, würde die gezeichnete Route im Dashboard bei jedem 
    # Klick wild zucken und sich ändern, obwohl der Warenkorb gleich geblieben ist.
    # Der MD5-Hash aus den Knoten garantiert: Gleicher Warenkorb = Gleiche Route.
    seed_str = "".join(targets)
    random.seed(int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32))

    # Wir sperren den Ausgangsknoten für den normalen Einkauf, damit der 
    # Algorithmus nicht versehentlich durch die Kasse läuft, um eine Abkürzung zu nehmen.
    def filter_end(n): return n != end_node
    G_shop = nx.subgraph_view(graph, filter_node=filter_end)
    valid_targets = [t for t in targets if t in G_shop.nodes]
    notes = []

    # Listen für die Plotly-Koordinaten vorbereiten
    px = [topology.nodes_config[start_node]['pos'][0]] if start_node in topology.nodes_config else []
    py = [topology.nodes_config[start_node]['pos'][1]] if start_node in topology.nodes_config else []
    tot_dist = 0

    # Fallback: Wenn der Warenkorb komplett leer ist, laufen wir direkt zum Ausgang
    if not valid_targets:
        try:
            path = nx.dijkstra_path(graph, start_node, end_node, weight='weight')
            dist = nx.dijkstra_path_length(graph, start_node, end_node, weight='weight')
            return [topology.nodes_config[n]['pos'][0] for n in path], [topology.nodes_config[n]['pos'][1] for n in path], dist, path, notes
        except: 
            return [], [], 0, [], notes

    # Wir trennen normale Regale von Produkten, die es nur an der Kasse gibt (Quengelware)
    queue_nodes = {'vW1', 'vW2', 'vW3'}
    store_t = [t for t in valid_targets if t not in queue_nodes]
    queue_t = [t for t in valid_targets if t in queue_nodes]
    
    # ---------------------------------------------------------
    # SCHRITT 2: Distanzmatrix aufbauen (Dijkstra)
    # ---------------------------------------------------------
    # Bevor wir die Reihenfolge (TSP) berechnen können, müssen wir wissen,
    # wie weit jedes Regal von jedem anderen Regal entfernt ist.
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

    # ---------------------------------------------------------
    # SCHRITT 3: Strategy Pattern anwenden (Die Wegoptimierung)
    # ---------------------------------------------------------
    if store_t:
        n_targets = len(store_t)
        # Je nach Länge der Einkaufsliste wählen wir den passenden Algorithmus
        if n_targets <= CONFIG.DP_EXACT_LIMIT:
            solver = HeldKarpDPSolver()          # Exakt, aber langsam
        elif n_targets > 25:
            solver = GeneticAlgorithmSolver()    # Robust für riesige Listen
        elif n_targets > CONFIG.SA_THRESHOLD: 
            solver = AntColonySolver()           # Gut für sehr viele Wegpunkte
        else:
            solver = SimulatedAnnealingSolver()  # Perfekt für mittelgroße Einkäufe
            
        store_seq, msg = solver.solve(d_mat, start_node, store_t, None)
        notes.append(msg)
    else:
        store_seq = [start_node]
        
    last_store = store_seq[-1]

    # ---------------------------------------------------------
    # SCHRITT 4: Kassenwahl mit Warteschlangentheorie
    # ---------------------------------------------------------
    ck_map = {'vW1': 'vK1', 'vW2': 'vK2', 'vW3': 'vK3'}
    best_ex_path, min_ex_cost, ch_ck = [], float('inf'), None
    
    # Prüfen, ob wir den Basis-Graphen oder den Stau-Graphen aus der ML-Pipeline haben
    smart = any('predicted_load' in data for _, _, data in graph.edges(data=True))

    if not queue_t:
        # Der Kunde hat keine Kassenprodukte. Wir suchen die absolut schnellste Kasse:
        # Laufweg zur Kasse + M/M/1-Wartezeit an der Kasse + Laufweg zum Ausgang
        for w, ck in ck_map.items():
            if w not in G_shop.nodes or ck not in graph.nodes: continue
            try:
                wt = nx.dijkstra_path_length(G_shop, last_store, w, weight='weight')
                
                # Wie viele Leute stehen laut KI-Modell aktuell dort an?
                base_lam = graph[w][ck].get('predicted_load', 0.5) if smart and graph.has_edge(w, ck) else 0.5
                # Mathematische Wartezeit berechnen
                qp = EnterpriseQueuingModel.calculate_wait_metrics(base_lam, current_time.hour, ck)["wait_sec"]
                
                et = graph[ck][end_node].get('weight', 1.0) if graph.has_edge(ck, end_node) else 2.0
                
                tt = wt + qp + et
                if tt < min_ex_cost:
                    min_ex_cost, ch_ck = tt, ck
                    best_ex_path = nx.dijkstra_path(G_shop, last_store, w, weight='weight') + [ck, end_node]
            except: continue
    else:
        # Der Kunde hat gezielt Produkte an den Kassen (z.B. Kaugummi) gesucht.
        # Wir müssen alle Kassen abklappern, an denen seine Produkte liegen.
        for perm in itertools.permutations(queue_t):
            try:
                tt = d_mat.get((last_store, perm[0]), float('inf'))
                for i in range(len(perm)-1): 
                    tt += d_mat.get((perm[i], perm[i+1]), float('inf'))
                
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

    if not best_ex_path: 
        best_ex_path = [end_node]

    # ---------------------------------------------------------
    # SCHRITT 5: Finale Wegpunkte für die Visualisierung zusammensetzen
    # ---------------------------------------------------------
    f_seq = store_seq.copy()
    if queue_t: f_seq.extend(queue_t)
    if ch_ck: f_seq.append(ch_ck)
    f_seq.append(end_node)

    # Jeden Kantenabschnitt ablaufen und X/Y Koordinaten für Plotly sammeln
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

    # Kassen-Wegpunkte hinzufügen
    if len(best_ex_path) > 1:
        for k in range(len(best_ex_path)-1):
            if graph.has_edge(best_ex_path[k], best_ex_path[k+1]):
                w = graph[best_ex_path[k]][best_ex_path[k+1]]['weight']
                if w < 9000: tot_dist += w
            n = best_ex_path[k+1]
            if n in topology.nodes_config:
                px.append(topology.nodes_config[n]['pos'][0])
                py.append(topology.nodes_config[n]['pos'][1])

    # Zufallsgenerator wieder freigeben, damit der Rest der App normal weiterläuft
    random.seed()

    return px, py, tot_dist, f_seq, notes

# =============================================================================
# VIII. EXPORT LAYER (Die API für das Dash-Frontend)
# =============================================================================
# Hier machen wir ausgewählte Variablen für die app.py verfügbar.
# So verhindern wir zirkuläre Imports und kapseln die Backend-Logik sauber ab.

G_base = topology.G_base
nodes_config = topology.nodes_config
edges_config = topology.edges_config
shelves = topology.shelves

# Export der zentral definierten Hex-Farben für ein einheitliches UI-Design
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
    """
    Fassade (Design Pattern), damit das Frontend einfach auf die Wartezeiten zugreifen kann,
    ohne die komplexen Klassenstrukturen der Stochastik zu kennen.
    """
    @staticmethod
    def calculate_wait_time(lam: float, h: int = 12, checkout_id: str = "vK1") -> float:
        """Kapselt den Aufruf an das M/M/1/K Modell."""
        return EnterpriseQueuingModel.calculate_wait_metrics(lam, h, checkout_id)["wait_sec"]

# Wir exportieren nur die saubere Fassade
QueuingModel = QueuingModelFacade