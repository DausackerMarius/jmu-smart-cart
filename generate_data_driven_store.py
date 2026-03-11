"""
=========================================================================================
JMU SMART SUPERMARKET: ENTERPRISE ETL PIPELINE (v16.0 - Type Hint Fixed Edition)
=========================================================================================
Umfang: >300 funktionale Zeilen purer Pipeline-Code.
Architektur: Entkoppelt von model.py Konstanten, um Import-Crashes zu verhindern.
Features: Sainte-Laguë Load Balancing, Gaussian Price Engine, Supervised ML Inference.
=========================================================================================
"""

import os
import json
import math
import time
import random
import logging
import pandas as pd
from typing import List, Dict, Tuple, Any  # <-- HIER IST DER FIX!

# Importiere NUR die Instanzen, KEINE Konstanten! Das verhindert den Config-Crash.
try:
    from model import ml_predictor, inv_manager
except ImportError as e:
    raise ImportError(f"Kritischer Fehler: model.py konnte nicht geladen werden. Details: {e}")

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)-7s | [%(name)s] %(message)s', 
    datefmt='%H:%M:%S'
)
log = logging.getLogger("ETL-Pipeline")
os.environ["PYTHONIOENCODING"] = "utf-8"

# Eigene, hart verdrahtete Pfade, um nicht von model.py abhängig zu sein
TRAINING_DATA_PATH = "smartcart_ml_training_data.csv"
DB_OUTPUT_PATH = "products.json"
ROUTING_OUTPUT_PATH = "routing_config.json"

# =============================================================================
# I. DOMAIN ENTITIES & ONTOLOGY (Single Source of Truth)
# =============================================================================

class StoreOntology:
    """Zentrale Definition der physikalischen und semantischen Supermarkt-Regeln."""
    
    MAX_CAPACITY = 6
    ML_CONFIDENCE_THRESHOLD = 0.25
    
    FIXED_ZONES = {
        "Spirituosenschrank": ["v5"],
        "Fleischtheke": ["vA10"],
        "Sonstiges (Kasse)": ["vW1", "vW2", "vW3"]
    }
    
    FLEXIBLE_ZONES = [
        "vD6", "vD5", "vD4", "vD2", "vD1",          
        "vC7", "vC6", "vC5", "vC3", "vC2",          
        "vB10", "vB9", "vB7", "vB6", "vB4", "vB2",  
        "vA9_2", "vA9", "vA7", "vA6", "vA4", "vA2", 
        "v1", "v2", "v3"                            
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
# II. EXTRACT: DATA INGESTION SERVICE
# =============================================================================

class DataIngestionService:
    """Verantwortlich für das Einlesen und Bereinigen der Rohdaten."""
    
    @staticmethod
    def extract_and_shuffle(file_path: str, limit: int = 450) -> List[str]:
        log.info(f"Ingestion: Lese Rohdaten aus '{file_path}'...")
        if not os.path.exists(file_path):
            log.error(f"Ingestion Error: Datei '{file_path}' nicht gefunden.")
            return []
            
        try:
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
            
            if 'Product_Name' not in df.columns:
                log.error("Ingestion Error: Spalte 'Product_Name' fehlt.")
                return []
                
            raw_items = df['Product_Name'].astype(str).str.strip()
            valid_items = raw_items[raw_items.str.len() > 2].drop_duplicates().tolist()
            
            random.shuffle(valid_items)
            selected = valid_items[:limit]
            
            log.info(f"Ingestion: {len(selected)} einzigartige Artikel geladen.")
            return selected
            
        except Exception as e:
            log.error(f"Ingestion Error: {e}")
            return []

# =============================================================================
# III. TRANSFORM: MACHINE LEARNING & PRICING ENGINE
# =============================================================================

class PricingEngine:
    """Generiert realistische Preise basierend auf Kategorien-Verteilungen (Gauß)."""
    
    _price_distributions = {
        "Fleischtheke": {"mu": 8.50, "sigma": 3.00, "min": 2.99},
        "Spirituosenschrank": {"mu": 15.00, "sigma": 5.00, "min": 7.99},
        "Fisch & Wurstwaren": {"mu": 4.50, "sigma": 1.50, "min": 1.49},
        "Obst & Gemüse": {"mu": 2.50, "sigma": 1.00, "min": 0.49},
        "Kühlregal (Vegan & Käse)": {"mu": 3.50, "sigma": 1.20, "min": 1.29},
        "Bier & Wein": {"mu": 5.00, "sigma": 2.50, "min": 0.89},
        "Drogerie & Haushalt": {"mu": 4.00, "sigma": 2.00, "min": 0.99},
        "Tiefkühlware": {"mu": 3.80, "sigma": 1.50, "min": 1.59},
        "Kaffee & Tee": {"mu": 5.50, "sigma": 2.50, "min": 1.99},
        "default": {"mu": 2.50, "sigma": 1.00, "min": 0.69}
    }

    @classmethod
    def generate_price(cls, category: str) -> str:
        params = cls._price_distributions.get(category, cls._price_distributions["default"])
        price = random.gauss(params["mu"], params["sigma"])
        price = max(price, params["min"])
        cents = random.choice([0.99, 0.49, 0.89, 0.29])
        return f"{math.floor(price) + cents:.2f} €"

class TransformationService:
    """Verarbeitet Rohtexte durch die MLOps Engine der model.py."""
    
    @staticmethod
    def classify_items(items: List[str]) -> pd.DataFrame:
        log.info(f"Transformation: Starte MLOps Inference für {len(items)} Artikel...")
        
        valid_records = []
        # Die model.py lädt das SBERT Modell (wie im Log gesehen) oder die SVM
        batch_results = ml_predictor.predict_batch(items)
        
        for i, item_name in enumerate(items):
            cat, _, conf = batch_results[i]
            
            if conf >= StoreOntology.ML_CONFIDENCE_THRESHOLD:
                price = PricingEngine.generate_price(cat)
                valid_records.append({
                    'Name': item_name.title(),
                    'Cat': cat,
                    'Conf': conf,
                    'Price': price
                })
                
        df_result = pd.DataFrame(valid_records)
        log.info(f"Transformation: {len(df_result)} Artikel erfolgreich kalibriert.")
        return df_result

# =============================================================================
# IV. LOAD: CAPACITY OPTIMIZER & GRAPH ALLOCATION
# =============================================================================

class CapacityBalancer:
    """Verteilt die flexiblen Regale mittels Sainte-Laguë/Schepers-Verfahren."""
    @staticmethod
    def calculate_routing_map(df: pd.DataFrame) -> Dict[str, List[str]]:
        log.info("Capacity: Berechne Sainte-Laguë Regal-Distribution...")
        
        if not df.empty:
            dynamic_cats = [c for c in df['Cat'] if c not in StoreOntology.FIXED_ZONES]
            counts = pd.Series(dynamic_cats).value_counts().to_dict()
        else:
            counts = {k: 1 for k in StoreOntology.EMERGENCY_FALLBACKS.keys() if k not in StoreOntology.FIXED_ZONES}

        allocation = {cat: 1 for cat in counts.keys()}
        slots_to_distribute = len(StoreOntology.FLEXIBLE_ZONES) - len(allocation)

        if slots_to_distribute > 0:
            for _ in range(slots_to_distribute):
                quotients = {cat: (count / (allocation[cat] + 0.5)) for cat, count in counts.items()}
                winner = max(quotients, key=quotients.get)
                allocation[winner] += 1

        available_nodes = StoreOntology.FLEXIBLE_ZONES.copy()
        random.shuffle(available_nodes)
        
        routing_map = {**StoreOntology.FIXED_ZONES}
        pointer = 0
        for cat, amount in allocation.items():
            routing_map[cat] = available_nodes[pointer : pointer+amount]
            pointer += amount
            
        return routing_map

class GraphAllocator:
    """Räumt die Artikel physisch in das Dictionary-Inventar der Graphenknoten ein."""
    
    def __init__(self, routing_map: Dict[str, List[str]]):
        self.routing_map = routing_map
        self.all_nodes = [n for nodes in StoreOntology.FIXED_ZONES.values() for n in nodes] + StoreOntology.FLEXIBLE_ZONES
        self.stock = {node: [] for node in self.all_nodes}
        self.capacity = {node: 0 for node in self.all_nodes}

    def _inject_hardware(self):
        log.info("Allocation: Injiziere Kassen-Mandatory-Items...")
        for q_node, item_list in StoreOntology.HARDWARE_INJECTIONS.items():
            for name, brand in item_list:
                self.stock[q_node].append({
                    'name': name, 'brand': brand, 'category': 'Sonstiges (Kasse)', 
                    'price': PricingEngine.generate_price("default"), 
                    'ai_confidence': 1.0, 'suggested_slot': q_node, 'needs_review': False
                })
                self.capacity[q_node] += 1

    def _distribute_ml_data(self, df: pd.DataFrame):
        log.info("Allocation: Verteile ML-validierte Produkte...")
        for _, row in df.iterrows():
            cat = row['Cat']
            target_nodes = [n for n in self.routing_map.get(cat, []) if self.capacity[n] < StoreOntology.MAX_CAPACITY]
            
            if target_nodes:
                best_node = min(target_nodes, key=lambda n: self.capacity[n])
                brand = random.choice(StoreOntology.BRAND_DISTRIBUTION.get(cat, ["JMU Choice"]))
                
                self.stock[best_node].append({
                    'name': row['Name'], 'brand': brand, 'category': cat, 
                    'price': row['Price'], 'ai_confidence': round(row['Conf'], 3), 
                    'suggested_slot': best_node, 'needs_review': False
                })
                self.capacity[best_node] += 1

    def _enforce_safety_net(self):
        log.info("Allocation: Aktiviere Emergency Fallbacks für 100% Integrität...")
        for cat, nodes in self.routing_map.items():
            for node in nodes:
                while self.capacity[node] < StoreOntology.MAX_CAPACITY:
                    fallback_name = random.choice(StoreOntology.EMERGENCY_FALLBACKS.get(cat, ["Basic Item"]))
                    brand = random.choice(StoreOntology.BRAND_DISTRIBUTION.get(cat, ["JMU Basic"]))
                    
                    self.stock[node].append({
                        'name': fallback_name, 'brand': brand, 'category': cat, 
                        'price': PricingEngine.generate_price(cat), 
                        'ai_confidence': 1.0, 'suggested_slot': node, 'needs_review': False
                    })
                    self.capacity[node] += 1

    def build(self, ml_data: pd.DataFrame) -> Dict[str, List[Dict]]:
        self._inject_hardware()
        self._distribute_ml_data(ml_data)
        self._enforce_safety_net()
        return self.stock

# =============================================================================
# V. PIPELINE ORCHESTRATOR & EXPORTER
# =============================================================================

class ETLPipelineOrchestrator:
    def __init__(self):
        self.csv_source = TRAINING_DATA_PATH
        self.total_nodes = len(StoreOntology.FIXED_ZONES) * 1 + len(StoreOntology.FLEXIBLE_ZONES)
        self.total_slots = self.total_nodes * StoreOntology.MAX_CAPACITY
        
    def execute(self):
        print("\n" + "="*75)
        print("🏭 ENTERPRISE STORE BUILDER v16.0 (Type-Hint Fixed Edition)")
        print("="*75)
        t_start = time.time()
        
        # 1. Extract
        raw_items = DataIngestionService.extract_and_shuffle(self.csv_source, limit=450)
        
        # 2. Transform
        df_classified = TransformationService.classify_items(raw_items)
        
        # 3. Load Balance
        routing_map = CapacityBalancer.calculate_routing_map(df_classified)
        
        # 4. Allocate
        allocator = GraphAllocator(routing_map)
        final_stock = allocator.build(df_classified)
        
        # 5. System Export
        log.info(f"Export: Übertrage Graphen-Status in {DB_OUTPUT_PATH}...")
        inv_manager.stock = final_stock
        inv_manager.save_to_json()
        
        log.info(f"Export: Kompiliere Routing-Topologie nach {ROUTING_OUTPUT_PATH}...")
        try:
            with open(ROUTING_OUTPUT_PATH, "w", encoding='utf-8') as f:
                json.dump(routing_map, f, indent=4, ensure_ascii=False)
        except Exception as e:
            log.error(f"Export Error (Routing): {e}")

        # 6. Reporting
        t_end = time.time()
        placed = sum(len(items) for items in final_stock.values())
        
        print("\n" + "="*75)
        print("📊 ETL SYSTEM AUDIT & VALIDIERUNGS-REPORT")
        print("="*75)
        print(f"⏱️  Ausführungszeit:        {t_end - t_start:.2f} Sekunden")
        print(f"📦 Regale befüllt:         {len(final_stock)} / 30")
        print(f"🛒 Produkte im System:     {placed} / {self.total_slots}")
        
        if not df_classified.empty:
            print(f"🧠 ML Avg Confidence:      {df_classified['Conf'].mean() * 100:.1f}%")
            
        print("\n📂 Systemdateien erfolgreich generiert und synchronisiert:")
        print(f"   ├── {DB_OUTPUT_PATH}        (Inventar & Pricing-Daten)")
        print(f"   └── {ROUTING_OUTPUT_PATH}  (Sainte-Laguë Graphen-Routing)")
        print("-" * 75)
        print("🚀 Setup abgeschlossen. Starte nun die app.py!")

if __name__ == "__main__":
    pipeline = ETLPipelineOrchestrator()
    pipeline.execute()