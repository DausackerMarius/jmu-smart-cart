Data Engineering & Supermarkt-Ontologie (ETL-Pipeline)
======================================================

Das vorherige Kapitel hat die räumliche Infrastruktur des JMU Smart Carts definiert: Ein Graphen-Modell des Supermarkts liegt als Digitaler Zwilling im Arbeitsspeicher vor. Doch an diesem Punkt der Systemarchitektur sind die Knoten (die physischen Regale) lediglich leere Container. 

Ein prädiktives Routing-System funktioniert nur mit einer absolut validen Datenbasis. Das Grundgesetz der Informatik *"Garbage In, Garbage Out"* (Müll rein, Müll raus) gilt hier rigoros: Wenn die Produktdatenbank fehlerhaft ist oder Produkte topologisch nicht existieren, navigiert der Dijkstra-Algorithmus den Kunden im besten Fall vor eine leere Wand und im schlimmsten Fall stürzt der Webserver ab. 

Anstatt rudimentäre Dummy-Daten von Hand zu schreiben (was für eine produktionsnahe Architektur nicht skalierbar wäre), implementiert das System eine autonome **Data-Engineering-Pipeline nach dem ETL-Muster (Extract, Transform, Load)**. Diese Pipeline fungiert als automatisierte Brücke zwischen der unstrukturierten Außenwelt und unserem streng typisierten Graphen.

Sie lädt den echten, zehntausende Zeilen umfassenden "Bundeslebensmittelschlüssel" (BLS), reinigt ihn von bürokratischem Rauschen, reichert ihn stochastisch an und verteilt die Produkte anschließend algorithmisch und fair auf die virtuellen Supermarkt-Regale.

Teil I: Extract – Das Pandas-Paradoxon und Speichereffizienz
------------------------------------------------------------
Die erste große ingenieurtechnische Hürde ist der Umgang mit großen Datenmengen (Big Data). Der Bundeslebensmittelschlüssel ist eine gewaltige CSV-Datei. 

*Die architektonische Baseline:* In der reinen Data Science ist es Branchenstandard, Daten über die Bibliothek ``pandas`` via ``pandas.read_csv()`` einzulesen. In einer produktiven, serverseitigen Microservice-Architektur ist dies jedoch ein schwerwiegender Designfehler (Anti-Pattern). Pandas ist "eager" (gierig) und lädt die gesamte Datei auf einmal in den Arbeitsspeicher. Bei sehr großen Dateien führt dies unweigerlich zu Memory-Spikes und einem **Out-of-Memory (OOM) Absturz**.

Um die Skalierbarkeit des Backends zu garantieren, verwirft das System den Einsatz von Pandas für den Extract-Prozess und nutzt stattdessen hardwarenahes Stream-Processing.

1.1 OS-Level Chunking: Die Datei in Blöcken lesen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Das System implementiert einen speicherschonenden Downloader, der die Datei in kleinen Häppchen (Chunks) über das Netzwerk zieht. 

.. code-block:: python

   import requests
   import logging

   class BLSDownloader:
       @staticmethod
       def download_in_chunks(url: str, filepath: str, chunk_size: int = 8192) -> None:
           """
           Lädt eine Big-Data-Datei häppchenweise (in Chunks) herunter.
           Der RAM-Verbrauch bleibt konstant O(1), selbst wenn die Datei Terabytes groß ist.
           """
           # stream=True verhindert das sofortige Laden in den RAM
           with requests.get(url, stream=True) as response:
               response.raise_for_status() 
               
               with open(filepath, 'wb') as file:
                   # Wir lesen exakt 8192 Bytes (8 KB) auf einmal
                   for chunk in response.iter_content(chunk_size=chunk_size):
                       if chunk: 
                           file.write(chunk)
           
           logging.info(f"Speicherschonender Download abgeschlossen: {filepath}")

*Die Wahl der Blockgröße:* Warum exakt 8192 Bytes? Betriebssysteme (OS) und Festplatten-Controller verarbeiten I/O-Operationen (Input/Output) physisch in Paging-Blöcken von meist 4 KB oder 8 KB. Die Wahl von 8192 Bytes sorgt für eine perfekte Synchronisation mit dem Betriebssystem, minimiert I/O-Overhead und hält den RAM-Verbrauch konstant auf **O(1)**.

1.2 Lazy Evaluation: Das Generator-Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Nachdem die Datei sicher auf der Festplatte liegt, muss sie zeilenweise in Python eingelesen werden. Auch hier verzichtet das System auf klassische Listen-Rückgaben (``return list``), da dies den RAM sofort wieder füllen würde. Stattdessen kommt das Konzept der **Lazy Evaluation** (Verzögerte Auswertung) zum Einsatz.

Der Code nutzt das Python-Schlüsselwort ``yield``, um einen Generator zu erschaffen.

.. code-block:: python

   import csv
   from typing import Generator, Dict

   def stream_bls_data(filepath: str) -> Generator[Dict[str, str], None, None]:
       """ 
       Liest die CSV-Datei als Stream. Gibt immer nur exakt EINE Zeile in den RAM,
       verarbeitet diese und gibt den Speicher danach sofort wieder frei.
       """
       with open(filepath, mode='r', encoding='utf-8') as f:
           reader = csv.DictReader(f, delimiter=';')
           for row in reader:
               
               # Data Cleaning Level 1: Korrupte Zeilen ohne ID sofort verwerfen
               if not row.get('S_Bezeichnung') or not row.get('BLS_Code'):
                   continue 
                   
               # 'yield' friert die Funktion ein und gibt exakt diese Zeile an die Pipeline.
               # Die Speicherkomplexität bleibt bei O(1).
               yield row

Teil II: Transform – Semantische Reduktion und Stochastik
---------------------------------------------------------
Behördliche B2B-Datensätze (wie der BLS) sind für Endkonsumenten in einer B2C-Applikation unlesbar. Ein Eintrag heißt dort nicht "Milch", sondern oft bürokratisch: *"Kuhmilch, pasteurisiert, homogenisiert, mindestens 3.5% Fett, in Verkaufsverpackung"*.

2.1 Semantische Reduktion gegen den "Fluch der Dimensionalität"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Für unsere in Kapitel 2 definierte, fehlertolerante Suchmaschine (Fuzzy Search via Damerau-Levenshtein) ist ein derart langer String toxisch. In der Mathematik der Textverarbeitung (NLP) nennt man dies den **Fluch der Dimensionalität**. Wenn der Algorithmus Suchbegriffe in einem 15-Wörter-Satz abgleichen muss, explodiert die Laufzeit der O(n*m) Matrix und generiert unzählige False-Positives.

Die Klasse ``TextSanitizer`` reinigt den Text radikal und reduziert ihn auf den semantischen Kern:

.. code-block:: python

   import re
   from typing import Optional

   class TextSanitizer:
       def __init__(self):
           # Stoppwörter: B2B-Begriffe, die für den Endkunden im Markt irrelevant sind
           self.stopwords = {"schlachtkörper", "kutterhilfsmittel", "zubereitung", "roh"}
       
       def clean_product_name(self, raw_name: str) -> Optional[str]:
           # 1. String-Kappung: Wir schneiden alles ab dem ersten Komma ab.
           # Aus "Milch, frisch, 3.5%" wird die semantische Essenz "Milch".
           name = raw_name.split(',')[0].strip() 
           
           # 2. Reguläre Ausdrücke (Regex): Entfernt alle Klammerzusätze
           name = re.sub(r'\s*\(.*?\)', '', name).strip()
           
           words = name.split()
           
           # 3. Dimensionalitäts-Filter: Wenn der Name nach der Reinigung noch immer 
           # länger als 3 Wörter ist, ist es kein kundenfähiges Produkt. -> Verwerfen.
           if len(words) == 0 or len(words) > 3:
               return None 
               
           # 4. B2B-Filter: Enthält das Wort industrielle Stoppwörter? -> Verwerfen.
           if any(word.lower() in self.stopwords for word in words):
               return None 
               
           # Schön formatiert (Title Case) zurückgeben
           return name.title() 

2.2 Stochastische Anreicherung (Idempotenz)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Um die Agenten in der späteren Simulation realistisch agieren zu lassen, benötigt jedes Produkt zwingend einen Preis und einen Popularitäts-Wert. Da der behördliche BLS diese ökonomischen Daten nicht liefert, muss das Backend diese "würfeln".

*Das Problem des reinen Zufalls:* Würde man einfach ``random.random()`` nutzen, würde ein Apfel bei jedem Neustart des Servers einen anderen Preis erhalten. Machine-Learning-Modelle lassen sich mit ständig mutierenden Ground-Truth-Daten nicht trainieren. Die Pipeline muss **idempotent** sein (Gleicher Input führt immer zwingend zum gleichen Output).

*Die Lösung:* Das System nutzt einen deterministischen Seed basierend auf dem Hash-Wert des Produktnamens. Der Zufall ist somit an den Namen gekettet. Zudem werden Preise nicht wild, sondern aus einer Normalverteilung (Gauß-Kurve) gezogen, um eine ökonomisch realistische Preisverteilung im Supermarkt zu simulieren.

.. code-block:: python

   import random

   def transform_and_enrich(row: dict, sanitizer: TextSanitizer) -> Optional[dict]:
       clean_name = sanitizer.clean_product_name(row['S_Bezeichnung'])
       if not clean_name: return None
       
       bls_prefix = row['BLS_Code'].upper()[0]
       category = "Kühlregal" if bls_prefix in ['M', 'K'] else "Sonstiges"
       
       # Deterministischer Zufall: Ein "Apfel" kostet bei jedem Server-Boot immer gleich viel.
       # Dies garantiert die Idempotenz der Pipeline für ML-Training.
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
An diesem Punkt existieren tausende saubere, angereicherte Produkte im RAM. Diese müssen nun auf die leeren Container-Knoten (Flexible Zones) des Graphen aus Kapitel 3 gemappt werden. Hier müssen zwei massive topologische Probleme gelöst werden.

3.1 Die Verhinderung von Null-Allokationen (Sainte-Laguë)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Das erste Problem: Wir haben 300 Molkerei-Produkte und nur 10 Gewürz-Produkte, aber nur 15 leere Regale im Graphen. 

Würde das System die Regale per normaler Prozentrechnung (Dreisatz) verteilen und mit ``math.floor()`` abrunden, bekäme die Gewürz-Abteilung mathematisch $0.48 \rightarrow 0$ Regale zugewiesen. Ein Versuch, Gewürze in "null" Regale zu sortieren, wirft eine Exception und tötet den Server.

Das System nutzt daher einen Algorithmus aus dem parlamentarischen Wahlrecht: Das **Sainte-Laguë-Verfahren** (Höchstzahlverfahren). Es nutzt den Divisor ``(allocations + 0.5)``. Dieser Divisor garantiert mathematisch, dass selbst die kleinste Kategorie (Minderheit) zwingend mindestens ein physisches Regal zugewiesen bekommt, bevor große Kategorien ihr fünftes oder sechstes Regal erhalten.

.. code-block:: python

   from typing import Dict

   def allocate_shelves(total_shelves: int, category_counts: Dict[str, int]) -> Dict[str, int]:
       """ Verteilt physische Graphen-Regale fair auf Basis der Produktmengen. """
       shelves_allocated = {cat: 0 for cat in category_counts.keys()}
       
       for _ in range(total_shelves):
           best_category = None
           max_quotient = -1.0
           
           for cat, amount in category_counts.items():
               # Das Sainte-Laguë Höchstzahlverfahren. 
               # Der +0.5 Divisor ist der mathematische Schutz vor Null-Zuweisungen.
               quotient = amount / (shelves_allocated[cat] + 0.5) 
               
               if quotient > max_quotient:
                   max_quotient = quotient
                   best_category = cat
                   
           if best_category:
               shelves_allocated[best_category] += 1
       
       return shelves_allocated

3.2 Präventives Load-Balancing (Round-Robin)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Das zweite Problem: Die "Molkerei" hat nun fair 3 Regale (Knoten A, B und C) gewonnen. Wir haben 300 Milchprodukte. Ein naiver Ansatz würde die ersten 100 Produkte in Regal A pressen, bis dieses voll ist. 

*Die topologische Katastrophe:* Wenn jeder Kunde nun nach beliebiger Milch sucht, routet das System alle Kunden exakt zu Knoten A. Knoten B und C bleiben leer. Es entsteht ein gigantischer physischer Stau (Bottleneck) vor Regal A. 

Die Architektur löst dieses Problem durch systematisches **Round-Robin-Routing**. Die Produkte werden wie beim Kartengeben reihum verteilt: Ein Produkt in Regal A, eines in B, eins in C, das nächste wieder in A. Dadurch wird das Kundeninteresse automatisch auf mehrere Knoten im Supermarkt gestreut. Es ist ein präventives, physikalisches Load-Balancing.

.. code-block:: python

   import itertools
   from typing import List

   def assign_nodes_to_products(products: List[dict], shelf_nodes: List[str]):
       """ Verteilt Produkte reihum auf die Knoten zur Stau-Prävention. """
       if not shelf_nodes:
           raise ValueError("Kritischer Topologie-Fehler: Kategorie ohne Raum-Knoten!")
           
       # itertools.cycle erschafft einen endlosen Ring-Iterator: A -> B -> C -> A -> B ...
       node_cycle = itertools.cycle(shelf_nodes)
       
       for product in products:
           product['node_id'] = next(node_cycle)

Teil IV: Materialisierung und Data Contract
-------------------------------------------
Die Pipeline muss am Ende die verarbeiteten Daten persistieren. Bevor ein Produkt jedoch als Ground-Truth auf der Festplatte gespeichert wird, muss es einen letzten, unbestechlichen Check durchlaufen. 

Das Framework **Pydantic** definiert das Schema als strengen "Data Contract" (Vertrag). Jedes Produkt, das einen negativen Preis hat oder dessen Name zu kurz ist, wird vom Türsteher gnadenlos verworfen (``ValidationError``). 

Am Ende materialisiert die Pipeline zwei Artefakte:
1. Eine hochperformante **JSON-Datei**. Diese dient als zustandslose, extrem schnell einlesbare Datenbank für die Live-REST-API der Tablets.
2. Eine flache **CSV-Matrix**. Diese dient ausschließlich der Data-Science-Abteilung, da das XGBoost-Machine-Learning-Modell CSV-Feature-Matrizen für das Training bevorzugt.

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
       price: float = Field(..., gt=0.0) # Preis MUSS zwingend größer 0 sein
       popularity: float = Field(..., ge=0.0, le=1.0) 

   def run_etl_pipeline(bls_url: str, raw_filepath: str, graph_nodes: List[str]) -> Dict[str, ProductModel]:
       # 1. EXTRACT
       BLSDownloader.download_in_chunks(bls_url, raw_filepath)
       
       sanitizer = TextSanitizer()
       transformed_products = []
       category_counts = {}
       
       # 2. TRANSFORM (Stream-Processing via Lazy Evaluation)
       for raw_row in stream_bls_data(raw_filepath):
           enriched = transform_and_enrich(raw_row, sanitizer)
           if enriched:
               transformed_products.append(enriched)
               # Zählt Kategorien für das spätere Sainte-Laguë Verfahren
               category_counts[enriched['category']] = category_counts.get(enriched['category'], 0) + 1
               
       # 3. LOAD (Allokation & Round-Robin Load Balancing)
       shelf_allocations = allocate_shelves(len(graph_nodes), category_counts)
       
       node_iterator = iter(graph_nodes)
       category_to_nodes = {
           cat: [next(node_iterator) for _ in range(amount)] 
           for cat, amount in shelf_allocations.items()
       }
       
       for category, nodes in category_to_nodes.items():
           products_in_cat = [p for p in transformed_products if p['category'] == category]
           if nodes:
               assign_nodes_to_products(products_in_cat, nodes)
       
       # 4. Validierung durch den Data Contract & Aufbau des O(1) Dictionaries
       final_inventory_dict = {}
       for prod_data in transformed_products:
           try:
               valid_product = ProductModel(**prod_data)
               # Ein Dictionary erlaubt der Live-Engine später Suchzugriffe in O(1)
               final_inventory_dict[valid_product.id] = valid_product
           except ValidationError:
               pass # Fehlerhafte Daten fliegen lautlos aus der Pipeline
               
       # 5. MATERIALISIERUNG (Dualer Export für API und Machine Learning)
       with open('products_live.json', 'w', encoding='utf-8') as f:
           json_data = [p.dict() for p in final_inventory_dict.values()]
           json.dump(json_data, f, ensure_ascii=False, indent=2)
           
       with open('ml_training_features.csv', 'w', encoding='utf-8', newline='') as f:
           writer = csv.writer(f)
           writer.writerow(['id', 'name', 'category', 'price', 'popularity'])
           for p in final_inventory_dict.values():
               writer.writerow([p.id, p.name, p.category, p.price, p.popularity])
               
       return final_inventory_dict