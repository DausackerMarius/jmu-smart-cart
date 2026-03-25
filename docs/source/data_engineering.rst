Data Engineering & Supermarkt-Ontologie (ETL-Pipeline)
======================================================

Das vorherige Kapitel definierte die räumliche Infrastruktur des JMU Smart Carts: Ein Graphen-Modell des Supermarkts dient als Digitaler Zwilling im Arbeitsspeicher. Für das dynamische Routing fungieren die Knoten (physische Regale) an diesem Punkt jedoch lediglich als leere Container. 

Ein prädiktives Routing-System erfordert eine durchgehend konsistente Datenbasis. Ist die Produktdatenbank fehlerhaft oder existieren referenzierte Produkte topologisch nicht, führt dies deterministisch zu fehlerhaften Pfadberechnungen oder Laufzeitfehlern im Backend. 

Um eine skalierbare, produktionsnahe Architektur zu gewährleisten, implementiert das System eine autonome Data-Engineering-Pipeline nach dem ETL-Muster (Extract, Transform, Load). Diese Pipeline dient als automatisierte Schnittstelle, die den nativen "Bundeslebensmittelschlüssel" (BLS) importiert, semantisch bereinigt, stochastisch anreichert und die resultierenden Produkte algorithmisch auf die virtuellen Supermarkt-Regale abbildet.

Teil I: Extract – Speichereffizienz durch Stream-Processing
-----------------------------------------------------------
Eine zentrale Herausforderung im Data Engineering ist das Ressourcenmanagement bei großen Datensätzen. Der Bundeslebensmittelschlüssel liegt als umfangreiche CSV-Datei vor. 

*Die architektonische Abwägung:* In der explorativen Datenanalyse ist es Branchenstandard, Daten über die Bibliothek ``pandas`` via ``pandas.read_csv()`` in den RAM zu laden. In einer produktiven Microservice-Architektur stellt dieses "Eager Loading" (sofortiges Laden) jedoch ein Performance-Risiko dar. Bei großen Dateien oder parallelen Instanzen führt dies zu Speicherengpässen (Memory Spikes) und potenziellen Out-of-Memory-Abstürzen (OOM). 

Um die Skalierbarkeit des Backends zu garantieren, nutzt das System stattdessen hardwarenahes Stream-Processing.

1.1 OS-Level Chunking: Blockweises Herunterladen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Das System implementiert einen speicherschonenden Downloader, der die Datei in definierten Blöcken (Chunks) über das Netzwerk abruft und direkt auf die Festplatte schreibt.

.. code-block:: python

   import requests
   import logging

   class BLSDownloader:
       @staticmethod
       def download_in_chunks(url: str, filepath: str, chunk_size: int = 8192) -> None:
           """
           Lädt eine Datei blockweise herunter. Der RAM-Verbrauch 
           bleibt unabhängig von der Dateigröße konstant auf O(1).
           """
           # stream=True verhindert das sofortige Laden in den RAM
           with requests.get(url, stream=True) as response:
               response.raise_for_status() 
               
               with open(filepath, 'wb') as file:
                   for chunk in response.iter_content(chunk_size=chunk_size):
                       if chunk: 
                           file.write(chunk)
           
           logging.info(f"Stream-Download abgeschlossen: {filepath}")

*Die Wahl der Blockgröße:* Betriebssysteme und Festplatten-Controller verarbeiten I/O-Operationen physisch in Paging-Blöcken von meist 4 KB oder 8 KB. Eine Chunk-Größe von exakt 8192 Bytes (8 KB) synchronisiert den Download optimal mit dem Betriebssystem, minimiert den I/O-Overhead und sichert die $\mathcal{O}(1)$ Speicherkomplexität.

1.2 Lazy Evaluation: Das Generator-Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Auch beim Einlesen der lokalen Datei in Python wird auf klassische Listen-Rückgaben verzichtet, da diese den Arbeitsspeicher iterativ füllen würden. Stattdessen kommt das Konzept der verzögerten Auswertung (Lazy Evaluation) zum Einsatz. 

Der Code nutzt das Python-Schlüsselwort ``yield``, um einen Generator zu erschaffen, der die Zeilen einzeln auswertet und den Speicher direkt wieder freigibt.

.. code-block:: python

   import csv
   from typing import Generator, Dict

   def stream_bls_data(filepath: str) -> Generator[Dict[str, str], None, None]:
       """ 
       Liest die CSV-Datei als Stream. Verarbeitet jeweils eine Zeile
       und hält die Speicherkomplexität des Lesezugriffs bei O(1).
       """
       with open(filepath, mode='r', encoding='utf-8') as f:
           reader = csv.DictReader(f, delimiter=';')
           for row in reader:
               # Data Cleaning Level 1: Inkonsistente Zeilen überspringen
               if not row.get('S_Bezeichnung') or not row.get('BLS_Code'):
                   continue 
                   
               # 'yield' gibt die Zeile iterativ an die Pipeline weiter
               yield row

Teil II: Transform – Semantische Reduktion und Stochastik
---------------------------------------------------------
Behördliche B2B-Datensätze sind durch komplexe Nomenklaturen geprägt (z.B. *"Kuhmilch, pasteurisiert, homogenisiert, mindestens 3.5% Fett"*), die für eine B2C-Suchfunktion ungeeignet sind.

2.1 Reduktion der Textkomplexität
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Für die in Kapitel 2 definierte fehlertolerante Suchmaschine (Fuzzy Search via Damerau-Levenshtein) verursachen überlange Strings unnötigen Rechenaufwand. Die Laufzeit der dynamischen Programmierung steigt mit $\mathcal{O}(N \cdot M)$, was bei langen Suchbegriffen zu Latenzen und einer erhöhten Rate an False-Positives führt.

Die Klasse ``TextSanitizer`` bereinigt den Text und isoliert den semantischen Kern:

.. code-block:: python

   import re
   from typing import Optional

   class TextSanitizer:
       def __init__(self):
           # B2B-Begriffe, die für den Endkunden irrelevant sind
           self.stopwords = {"schlachtkörper", "kutterhilfsmittel", "zubereitung", "roh"}
       
       def clean_product_name(self, raw_name: str) -> Optional[str]:
           # 1. String-Kappung: Extraktion des primären Nomens vor dem ersten Komma.
           name = raw_name.split(',')[0].strip() 
           
           # 2. Reguläre Ausdrücke: Entfernung von Klammerzusätzen
           name = re.sub(r'\s*\(.*?\)', '', name).strip()
           words = name.split()
           
           # 3. Filterung: Verwerfen von überkomplexen Bezeichnungen (> 3 Wörter)
           if len(words) == 0 or len(words) > 3:
               return None 
               
           # 4. Filterung: Ausschluss industrieller Stoppwörter
           if any(word.lower() in self.stopwords for word in words):
               return None 
               
           return name.title() 

2.2 Stochastische Anreicherung (Idempotenz)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Um das spätere Agentenverhalten realistisch zu simulieren, erfordert jedes Produkt Parameter für Preis und Popularität. Da der BLS diese ökonomischen Daten nicht liefert, werden sie synthetisch generiert.

*Anforderung an Determinismus:* Die Nutzung von reinem Zufall (``random.random()``) würde dazu führen, dass sich die Metriken bei jedem Systemstart ändern. Für ein konsistentes Machine-Learning-Training muss die Pipeline jedoch idempotent arbeiten (gleicher Input generiert stets gleichen Output). 

Das System löst dies durch einen deterministischen Seed, der an den Hash-Wert des Produktnamens gekoppelt ist. Die Preise werden zudem aus einer Normalverteilung (Gauß-Kurve) abgeleitet, um eine realistische ökonomische Streuung zu gewährleisten.

.. code-block:: python

   import random

   def transform_and_enrich(row: dict, sanitizer: TextSanitizer) -> Optional[dict]:
       clean_name = sanitizer.clean_product_name(row['S_Bezeichnung'])
       if not clean_name: return None
       
       bls_prefix = row['BLS_Code'].upper()[0]
       category = "Kühlregal" if bls_prefix in ['M', 'K'] else "Sonstiges"
       
       # Deterministischer Zufall garantiert die Idempotenz der Pipeline
       random.seed(hash(clean_name)) 
       
       # Ökonomische Modellierung via Gauß-Kurve
       mu, sigma = (1.80, 0.80) if category == "Kühlregal" else (3.50, 1.50)
       price = round(max(random.gauss(mu, sigma), 0.19), 2)
       popularity = round(random.uniform(0.1, 0.9), 2)
       
       return {
           "id": row['BLS_Code'],
           "name": clean_name,
           "category": category,
           "price": price,
           "popularity": popularity
       }

Teil III: Load – Algorithmische Graphen-Befüllung
-------------------------------------------------
Die transformierten Produkte müssen abschließend proportional auf die verfügbaren Raumknoten des Graphen verteilt werden, was zwei topologische Anforderungen an das System stellt.

3.1 Proportionale Allokation (Sainte-Laguë & Grundmandate)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Bei einer starken Diskrepanz in den Kategoriemengen (z.B. 300 Molkerei-Produkte vs. 1 Gewürz-Produkt) führt eine reine Prozentrechnung zu fehlerhaften Null-Allokationen für kleine Kategorien. 

Zur Sicherstellung der topologischen Erreichbarkeit nutzt der Algorithmus **Grundmandate**: Bevor die proportionale Verteilung beginnt, wird jeder identifizierten Kategorie exakt ein physisches Regal zugewiesen. Die verbleibenden Regale werden anschließend über das aus der Wahlmathematik bekannte **Sainte-Laguë-Verfahren** (Höchstzahlverfahren) fair auf die größeren Kategorien verteilt.

.. code-block:: python

   from typing import Dict

   def allocate_shelves(total_shelves: int, category_counts: Dict[str, int]) -> Dict[str, int]:
       """ Verteilt physische Graphen-Regale fair auf Basis der Produktmengen. """
       # Grundmandate sichern die topologische Existenz jeder Kategorie
       shelves_allocated = {cat: 1 for cat in category_counts.keys()}
       remaining_shelves = total_shelves - len(category_counts)
       
       if remaining_shelves < 0:
           raise ValueError("Ressourcenkonflikt: Weniger Knoten im Graph als Kategorien.")
           
       # Proportionale Verteilung des Rests via Sainte-Laguë
       for _ in range(remaining_shelves):
           best_category = None
           max_quotient = -1.0
           
           for cat, amount in category_counts.items():
               # Das Höchstzahlverfahren verhindert die Benachteiligung kleiner Gruppen
               quotient = amount / (shelves_allocated[cat] + 0.5) 
               
               if quotient > max_quotient:
                   max_quotient = quotient
                   best_category = cat
                   
           if best_category:
               shelves_allocated[best_category] += 1
               
       return shelves_allocated

3.2 Präventives Load-Balancing (Round-Robin)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Besitzt eine Kategorie mehrere Regale (z.B. Knoten A, B und C), würde ein lineares Auffüllen (zuerst A, dann B, dann C) das Kundeninteresse im Markt ungleichmäßig bündeln und lokale Engpässe (Bottlenecks) provozieren. 

Zur physischen Stauprävention implementiert die Architektur ein **Round-Robin-Routing**. Über einen iterativen Zyklus (``itertools.cycle``) werden die Produkte gleichmäßig über alle dedizierten Knoten der jeweiligen Kategorie gestreut.

.. code-block:: python

   import itertools
   from typing import List

   def assign_nodes_to_products(products: List[dict], shelf_nodes: List[str]):
       """ Streut Produkte iterativ über die Knoten zur Stau-Prävention. """
       if not shelf_nodes:
           raise ValueError("Topologie-Fehler: Kategorie ohne Raum-Knoten.")
           
       # Erzeugt einen endlosen Ring-Iterator für die gleichmäßige Zuweisung
       node_cycle = itertools.cycle(shelf_nodes)
       
       for product in products:
           product['node_id'] = next(node_cycle)

Teil IV: Materialisierung und Data Contract (Two-Pass-Architektur)
------------------------------------------------------------------
Da die Zuweisung nach Sainte-Laguë das absolute Wissen über alle Produktmengen voraussetzt, erfordert die Beibehaltung der $\mathcal{O}(1)$ Speichereffizienz eine **Two-Pass-Architektur**. Dies stellt einen bewussten Trade-off dar: Es wird ein zusätzlicher Festplatten-Lesezugriff in Kauf genommen, um den Arbeitsspeicher zu entlasten.

*   **Pass 1:** Der Stream iteriert die CSV-Datei zur Aggregation der absoluten Zählerstände pro Kategorie.
*   **Pass 2:** Nach der Allokationsberechnung wird die Datei ein zweites Mal gestreamt. Die Produkte werden transformiert, einem Knoten zugewiesen und gegen das Pydantic-Schema validiert.

Fehlerhafte Datensätze werden im Sinne der Graceful Degradation aussortiert. Die validen Daten werden als $\mathcal{O}(1)$ Dictionary im RAM vorgehalten und abschließend sowohl als JSON (für die REST-API) als auch als CSV-Matrix (für das Machine-Learning-Training) materialisiert.

.. code-block:: python

   import json
   import csv
   from pydantic import BaseModel, Field, ValidationError
   from typing import List, Dict

   class ProductModel(BaseModel):
       """ Der Data Contract: Strikte Typisierung zur Laufzeit. """
       id: str
       name: str = Field(..., min_length=2)
       category: str
       node_id: str
       price: float = Field(..., gt=0.0) 
       popularity: float = Field(..., ge=0.0, le=1.0) 

   def run_etl_pipeline(bls_url: str, raw_filepath: str, graph_nodes: List[str]) -> Dict[str, ProductModel]:
       # 1. EXTRACT
       BLSDownloader.download_in_chunks(bls_url, raw_filepath)
       sanitizer = TextSanitizer()
       
       # 2. PASS 1: Aggregation der Metadaten (RAM bleibt O(1))
       category_counts = {}
       for raw_row in stream_bls_data(raw_filepath):
           enriched = transform_and_enrich(raw_row, sanitizer)
           if enriched:
               category_counts[enriched['category']] = category_counts.get(enriched['category'], 0) + 1
               
       # 3. KNOTEN-ALLOKATION
       shelf_allocations = allocate_shelves(len(graph_nodes), category_counts)
       node_iterator = iter(graph_nodes)
       category_to_nodes = {
           cat: [next(node_iterator) for _ in range(amount)] 
           for cat, amount in shelf_allocations.items()
       }
       
       round_robin_cycles = {
           cat: itertools.cycle(nodes) for cat, nodes in category_to_nodes.items() if nodes
       }
       
       # 4. PASS 2: Zuweisen, Validieren & Speichern (Trade-off: Erneuter Disk I/O)
       final_inventory_dict = {}
       for raw_row in stream_bls_data(raw_filepath):
           enriched = transform_and_enrich(raw_row, sanitizer)
           if not enriched: continue
           
           cat = enriched['category']
           enriched['node_id'] = next(round_robin_cycles[cat])
           
           try:
               valid_product = ProductModel(**enriched)
               final_inventory_dict[valid_product.id] = valid_product
           except ValidationError:
               pass # Inkonsistente Daten werden aussortiert
               
       # 5. MATERIALISIERUNG (Export für API und ML-Training)
       with open('products_live.json', 'w', encoding='utf-8') as f:
           json_data = [p.dict() for p in final_inventory_dict.values()]
           json.dump(json_data, f, ensure_ascii=False, indent=2)
           
       with open('ml_training_features.csv', 'w', encoding='utf-8', newline='') as f:
           writer = csv.writer(f)
           writer.writerow(['id', 'name', 'category', 'node_id', 'price', 'popularity'])
           for p in final_inventory_dict.values():
               writer.writerow([p.id, p.name, p.category, p.node_id, p.price, p.popularity])
               
       return final_inventory_dict