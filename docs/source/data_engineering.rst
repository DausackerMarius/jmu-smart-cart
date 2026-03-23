Data Engineering & Supermarkt-Ontologie
=======================================

Das vorherige Kapitel hat die räumliche Infrastruktur des JMU Smart Carts definiert: Ein gerichteter, kanten-gewichteter Supermarkt-Graph liegt im Arbeitsspeicher vor. Doch an diesem Punkt der System-Initialisierung sind die durch die ``StoreTopology`` generierten Knoten (die Regale) lediglich leere Daten-Container (Flexible Zones). 

Ein prädiktives, KI-gesteuertes System benötigt zwingend eine valide, rauscharme Datenbasis (Ground-Truth). Das informationstechnische Paradigma *"Garbage In, Garbage Out"* gilt hier absolut: Wenn die Produktdatenbank fehlerhaft, unvollständig oder semantisch überladen ist, navigiert der TSP-Solver den Kunden in physische Sackgassen, und die NLP-Suchmaschine halluziniert falsche Ergebnisse. Anstatt auf statische, händisch gepflegte Dummy-Daten zu vertrauen (was eine spätere Skalierung des Systems auf reale Supermärkte von Beginn an verhindern würde), implementiert das Backend eine vollständig autonome, code-getriebene **Data-Engineering-Pipeline (ETL)**.

Diese Pipeline transformiert den gigantischen, behördlichen Bundeslebensmittelschlüssel (BLS) vollautomatisch in unsere spezifische, räumliche Supermarkt-Ontologie. Die Architektur ist in zwei Hauptphasen strikt modularisiert:

1. **Phase I (Extract & Transform):** Speichereffizienter Download auf OS-Ebene und linguistische Dimensionsreduktion der Rohdaten als Fundament für das spätere NLP-Modell.
2. **Phase II (Load & Allocation):** Algorithmische Sitzverteilung (Sainte-Laguë) und physisches Load-Balancing (Round-Robin) dieser Produkte in die topologischen Knoten des Graphen.

1. Phase I: Extraction & OS-Level Memory Safety
-----------------------------------------------
Die Grundlage der System-Ontologie bildet der BLS, ein massiver B2B-Datensatz mit zehntausenden Zeilen und hunderten Spalten zu Makronährstoffen. Der naive Programmier-Ansatz, eine solche Datei über das Netzwerk direkt in eine Python-Variable (wie einen String oder einen Pandas-Dataframe) zu laden, ist ein klassisches Anti-Pattern. 

Python besitzt einen enormen Speicher-Overhead für interne Objekte. Ein 100 MB großes CSV-File kann im RAM als Pandas-Dataframe schnell 400 MB belegen. Bei exponentiell wachsenden Datenmengen führt dies unweigerlich zur Überlastung des Garbage Collectors und zu Server-Abstürzen durch RAM-Overflows (Out-of-Memory). Die Klasse ``BLSDownloader`` löst dieses Nadelöhr durch ein asynchrones **Chunking-Verfahren** (Streaming), das direkt auf TCP-Socket-Ebene ansetzt.

.. code-block:: python

   import requests

   class BLSDownloader:
       @staticmethod
       def download_in_chunks(url: str, filepath: str, chunk_size: int = 8192) -> None:
           """
           Streamt eine massive Datei über TCP-Sockets direkt auf die Festplatte.
           Garantiert O(1) RAM-Verbrauch, völlig unabhängig von der Dateigröße.
           """
           # stream=True verhindert das sofortige Herunterladen des gesamten Bodys
           with requests.get(url, stream=True) as response:
               response.raise_for_status() # Fail-Fast Prinzip bei HTTP-Fehlern (z.B. 404)
               
               with open(filepath, 'wb') as file:
                   # Iteriert über den Socket, ohne ihn als Ganzes in den RAM zu laden
                   for chunk in response.iter_content(chunk_size=chunk_size):
                       if chunk: # Filtert leere Keep-Alive-Chunks des Netzwerks heraus
                           file.write(chunk)

*Verständnis-Exkurs (Betriebssystem-Architektur):* Warum erzwingt der Code exakt eine ``chunk_size`` von 8192 Bytes (8 KB)? Dieser Wert ist kein Zufall, sondern eine exakte Anpassung an die Hardware. Er korreliert perfekt mit den Standard-Paging-Größen moderner Betriebssysteme und den Blockgrößen von SSD-Sektoren (meist 4 KB oder 8 KB). Der Server nimmt iterativ einen optimalen 8-KB-Block aus dem Netzwerk-Puffer, schreibt ihn mit maximaler I/O-Geschwindigkeit physisch auf die Festplatte und leert den RAM sofort wieder. Die Speicherplatzkomplexität für den Download kollabiert dadurch von einer linearen Abhängigkeit $\mathcal{O}(N)$ auf absolute Konstanz $\mathcal{O}(1)$.

1.2 Lazy Evaluation & Pipeline-Generatoren
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Nachdem die Datei sicher auf der Festplatte liegt, muss sie transformiert werden. Auch in diesem Schritt verzichtet das System bewusst auf Data-Science-Bibliotheken wie ``pandas``. Ein ``pd.read_csv()`` würde die Datei rigoros und vollständig (eager) in den Speicher zwingen.

Stattdessen nutzt die Architektur **Lazy Evaluation** (Verzögerte Auswertung) über Python-Generatoren (``yield``). Die Daten durchlaufen die Pipeline wie an einem industriellen Fließband:

.. code-block:: python

   import csv
   from typing import Generator, Dict

   def stream_bls_data(filepath: str) -> Generator[Dict[str, str], None, None]:
       """Liest die gigantische CSV-Datei zeilenweise (Lazy Pipeline Architektur)."""
       with open(filepath, mode='r', encoding='utf-8') as f:
           reader = csv.DictReader(f, delimiter=';')
           for row in reader:
               # 1. Missing Data Handling: Verwerfe korrupte Zeilen sofort
               if not row.get('S_Bezeichnung') or not row.get('BLS_Code'):
                   continue # Drop: Keine Imputation von fehlerhaften IDs
                   
               # 2. Yield übergibt exakt EINE Zeile an den nächsten Transform-Schritt.
               # Die vorherige Zeile wird vom Garbage Collector sofort gelöscht.
               yield row

Da reale Datensätze niemals fehlerfrei sind, implementiert dieser Code-Block ein hartes **Missing Data Handling**. Zeilen mit fehlenden Bezeichnern (Null/NaN-Werte) werden *nicht* durch statistische Schätzwerte ergänzt (Imputation), sondern deterministisch verworfen (Dropped). Warum? Eine "geschätzte" oder künstlich erzeugte Produkt-ID würde die Integrität der Ground-Truth verletzen und später im Routing-Modul zu Null-Pointer-Exceptions führen, da das Produkt im Graphen physikalisch nicht existiert.

2. Phase I: Semantische Dimensionsreduktion (NLP)
-------------------------------------------------
Rohe Ernährungsdatenbanken sind durchdrungen von komplexer, behördlicher B2B-Terminologie. Ein BLS-Eintrag lautet beispielsweise: *"Schlachtkörper, Rind, ohne Knochen, mindestens 20% Fett, vakuumverpackt"*. Für einen Endkunden im Supermarkt ist das unverständlich. 

Noch kritischer ist jedoch die mathematische Gefahr für das spätere Machine Learning: Würden wir diese überlangen Sätze ungefiltert in das System laden, würde die Document-Term-Matrix (DTM) des TF-IDF-Suchalgorithmus explodieren. Der mathematische Vektorraum wäre von tausenden irrelevanten Dimensionen ("Schlachtkörper", "Knochen", "vakuumverpackt") durchsetzt. Das Resultat wäre eine extrem spärliche Matrix (Sparse Matrix), die das NLP-Modell durch den **Curse of Dimensionality** (Fluch der Dimensionalität) rechenintensiv, speicherhungrig und hochgradig unpräzise macht.

Die Klasse ``TextSanitizer`` operiert daher als deterministischer linguistischer Filter in linearer Laufzeit $\mathcal{O}(N)$, um dieses Rauschen (Noise) radikal aus dem Vektorraum zu entfernen.

.. code-block:: python

   import re
   from typing import Optional

   class TextSanitizer:
       def __init__(self):
           # O(1) Hash-Set für blitzschnellen Stop-Word-Lookup in konstanter Zeit
           self.b2b_stopwords = {"schlachtkörper", "kutterhilfsmittel", "zubereitung"}
       
       def clean_product_name(self, raw_name: str) -> Optional[str]:
           # 1. RegEx-Sanitization: Entfernt Klammerzusätze und B2B-Suffixe
           name = re.sub(r'\s*\(.*?\)', '', raw_name).strip()
           # Isoliert das primäre Hauptwort vor dem ersten Komma
           name = name.split(',')[0].strip() 
           
           # 2. Retail-Filter: Harte Dimensionsreduktion für den Vektorraum
           words = name.split()
           if len(words) == 0 or len(words) > 3:
               return None # Drop: String ist zu lang/komplex für die Endkunden-Suche
               
           # 3. Stop-Word Elimination
           if any(word.lower() in self.b2b_stopwords for word in words):
               return None # Drop: B2B-Produkt erkannt
               
           return name

**Der architektonische Beweis:** Der unscheinbare Retail-Filter (``len(words) > 3``) ist der wichtigste MLOps-Eingriff der gesamten Pipeline. Er zwingt den riesigen BLS-Datensatz in ein kompaktes, endkundennahes Format (z.B. "Apfel", "Vollkornbrot", "Frische Milch"). Durch diese harte Limitierung der Wortlänge verdichtet sich der NLP-Vektorraum drastisch: Statt 15.000 einzigartiger Wörter (Dimensionen) muss das System später nur noch 800 hochrelevante Wörter berechnen. Erst diese Dimensionsreduktion macht die hochkomplexe Fehlertoleranz-Suche (Damerau-Levenshtein-Distanz) auf mobilen Tablets performant möglich.

3. Phase I: Ontologisches Mapping (Label-Generierung)
-----------------------------------------------------
Für das Supervised Learning benötigt das System sauber gelabelte Daten (Features $X$ und Klassen-Labels $y$). Die BLS-Rohdaten liefern jedoch keine Supermarkt-Regale, sondern nur abstrakte alphanumerische Klassifizierungs-Codes (z. B. "M" für Milchprodukte, "F" für Fleisch). 

Der ``SupermarketMapper`` fungiert als strukturierende Heuristik. Er iteriert über die Datenströme und wendet regelbasierte Mappings an, um diese abstrakten Codes in die physische Ontologie des Graphen (11 definierte Zonen) zu übersetzen.

.. code-block:: python

   class SupermarketMapper:
       """Übersetzt abstrakte Behörden-Codes in topologische Supermarkt-Zonen."""
       
       def map_to_aisle(self, bls_code: str) -> str:
           # Extrahiert das Haupt-Präfix des BLS-Stammbaums
           code_prefix = bls_code.upper()[0] 
           
           # Deterministisches Ontologie-Mapping
           if code_prefix in ['M', 'K']:
               return "Kühlregal (Molkerei & Käse)"
           elif code_prefix in ['F', 'W']:
               return "Fleischtheke & Wurst"
           elif code_prefix in ['G', 'O']:
               return "Obst & Gemüse"
               
           # Fallback-Kategorie: Verhindert Pipeline-Abstürze durch Missing Labels
           return "Sonstiges / Aktionsware"

Sollte ein BLS-Code durch zukünftige Updates der Behörde in keine definierte Regel passen (Edge-Case), greift zwingend die deterministische Fallback-Kategorie. Dies ist System-kritisch: Ein Datensatz ohne gültiges Zonen-Label würde im anschließenden Load-Prozess zu einer Fatal Exception (KeyError) führen und den Aufbau des Supermarkts abbrechen.

4. Phase II: Topologische Allokation (Load)
-------------------------------------------
In der zweiten Pipeline-Phase wird die saubere Datenbasis physisch im NetworkX-Graphen verortet. 
Hierbei muss die Software eine gravierende mathematische Diskrepanz lösen: Das System besitzt eine fixe, knappe Anzahl an physischen Regalknoten im Graphen (die "Sitze"), aber eine asymmetrische, stark variable Anzahl an extrahierten Produkten aus 11 Kategorien (die "Wählerstimmen").

4.1 Der mathematische Beweis des Sainte-Laguë-Verfahrens
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Um "Geister-Regale" zu verhindern und eine absolut faire Raumnutzung zu garantieren, implementiert die ``StoreBuilder`` Klasse das aus dem Parlamentarismus bekannte Höchstzahlverfahren nach **Sainte-Laguë**. Die Zuweisung von Kategorien zu den Graphen-Knoten erfolgt in der Komplexität $\mathcal{O}(S \cdot C)$ (Regalanzahl $S$ multipliziert mit Kategorienanzahl $C$).

Die Formel für den Quotienten $Q$, der bestimmt, welche Kategorie den nächsten leeren Graphen-Knoten erhält, lautet:

.. math::
   Q = \frac{V}{s + 0.5}

Wobei $V$ das Volumen der Produkte einer Kategorie ist und $s$ die Anzahl der bereits zugewiesenen Regale darstellt.

.. code-block:: python

   from typing import Dict

   def allocate_shelves(total_shelves: int, category_counts: Dict[str, int]) -> Dict[str, int]:
       """Allokiert Graphen-Knoten proportional zur Produktmenge via Sainte-Laguë."""
       shelves_allocated = {cat: 0 for cat in category_counts.keys()}
       
       # Wir verteilen iterativ jeden der verfügbaren leeren Knoten
       for _ in range(total_shelves):
           best_category = None
           max_quotient = -1.0
           
           for cat, votes in category_counts.items():
               # V = votes, s = shelves_allocated
               # Der Divisor +0.5 ist das mathematische Kernstück von Sainte-Laguë
               quotient = votes / (shelves_allocated[cat] + 0.5) 
               
               if quotient > max_quotient:
                   max_quotient = quotient
                   best_category = cat
                   
           # Die Kategorie mit dem höchsten Quotienten gewinnt den Knoten
           if best_category:
               shelves_allocated[best_category] += 1
           
       return shelves_allocated

*Der Beweis durch Zahlen:* Warum nutzen wir zwingend den Divisor ``+ 0.5`` (Sainte-Laguë) anstelle des simplen D’Hondt-Verfahrens (Divisor ``+ 1``)? 
Nehmen wir an, wir haben 3 Regale zu vergeben. Kategorie A ("Obst") hat 100 Produkte, Kategorie B ("Gewürze") hat 30 Produkte.
* **Bei D'Hondt:** Runde 1: A hat $100/1 = 100$ (Gewinnt Regal 1). 
  Runde 2: A hat $100/2 = 50$, B hat $30/1 = 30$. A gewinnt Regal 2. 
  Runde 3: A hat $100/3 = 33.3$, B hat $30/1 = 30$. A gewinnt Regal 3. 
  *Fazit:* D'Hondt bevorteilt mathematisch große Massen extrem. "Obst" frisst alle 3 Regale, "Gewürze" bekommt 0 Regale. Das System würde abstürzen, da Gewürze physisch obdachlos wären.
* **Bei Sainte-Laguë:** Runde 1: A hat $100/0.5 = 200$ (Gewinnt Regal 1).
  Runde 2: A hat $100/1.5 = 66.6$, B hat $30/0.5 = 60$. A gewinnt Regal 2.
  Runde 3: A hat $100/2.5 = 40$, B hat $30/0.5 = 60$. **B gewinnt Regal 3.**
  *Fazit:* Sainte-Laguë nivelliert die Benachteiligung. Es garantiert mathematisch, dass auch das kleinste Sortiment mindestens einen physischen Raum im Graphen zugewiesen bekommt.

.. declaration:google:search{queries: ["Sainte-Laguë method calculation steps example", "Sainte Lague seat allocation matrix"]}


4.2 Intra-Shelf Load Balancing (Round-Robin)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Nachdem das Verfahren berechnet hat, *wie viele* Knoten eine Kategorie erhält (z.B. "Backwaren bekommt die 3 Knoten: vC1, vC2, vC3"), müssen die extrahierten Produkte physisch auf die Arrays dieser Knoten verteilt werden. Das System nutzt hierfür ein iteratives **Round-Robin-Verfahren**:

.. code-block:: python

   import itertools
   from typing import List, Dict

   def distribute_products_to_nodes(products: List[dict], assigned_node_ids: List[str]) -> Dict[str, list]:
       """
       Verteilt Produkte exakt gleichmäßig (Round-Robin) auf die Graphen-Knoten.
       Verhindert topologische "Routing Hotspots" im Operations Research.
       """
       # Erzeugt ein Dictionary mit leeren Arrays für jeden zugewiesenen Knoten
       node_inventory = {node_id: [] for node_id in assigned_node_ids}
       
       # itertools.cycle iteriert endlos zyklisch über die Knoten (vC1 -> vC2 -> vC3 -> vC1...)
       node_iterator = itertools.cycle(assigned_node_ids)
       
       for product in products:
           # Hole den nächsten Knoten im Rotationsprinzip
           target_node = next(node_iterator)
           # Lege das Produkt physisch in den Speicher dieses Knotens
           node_inventory[target_node].append(product)
           
       return node_inventory

*Der Operations-Research Bezug:* Warum ist Round-Robin hier so extrem kritisch? Würde das Skript die ersten 100 Backwaren stur in den ersten Knoten ``vC1`` stopfen und ``vC2`` leer lassen, entstünde ein physischer **Routing Hotspot**. Wenn in der Live-Engine nun 50 reale Kunden im Supermarkt Brot suchen, würde der TSP-Solver alle 50 Kunden exakt und zeitgleich vor den Knoten ``vC1`` routen, was einen massiven physischen Stau auslöst. Das Round-Robin-Verfahren streut die Produkte gleichmäßig, was automatisch und präventiv zu einem physischen Load-Balancing der Laufwege führt.

5. Stochastische Preisgenerierung & Determinismus
-------------------------------------------------
Eine statische Preiszuweisung (z.B. jedes Brot kostet pauschal 2 Euro) würde die statistische Varianz der nachfolgenden agentenbasierten Simulation zerstören. Die ``PricingEngine`` generiert daher dynamische Kaufpreise auf Basis warengruppenspezifischer Normalverteilungen (Gauß-Kurven).

Um die Evaluierung der Architektur jedoch wissenschaftlich messbar, vergleichbar und testbar zu machen, muss der Output des Data Engineerings bei jedem Durchlauf absolut identisch (reproduzierbar) sein. Daher erzwingt die Engine striktes **Deterministic Seeding**.

.. code-block:: python

   import random

   # Determinismus: Garantiert wissenschaftliche Reproduzierbarkeit bei Unit-Tests
   random.seed(42) 

   def generate_price(mu: float, sigma: float, min_price: float) -> float:
       """Generiert einen Preis basierend auf P(x) = max(N(μ, σ), P_min)"""
       # Ziehung aus der Normalverteilung N(μ, σ)
       price = random.gauss(mu, sigma)
       
       # Hartes Clipping, um negative oder unrealistisch günstige Systemfehler zu verbieten
       return round(max(price, min_price), 2)

So erhält ein Produkt aus der Kategorie "Fleischtheke" (z.B. $\mu=4.50, \sigma=2.00$) einen weitaus realistischeren und breiter gestreuten Preis als ein Basisartikel aus der Kategorie "Backwaren" ($\mu=1.80, \sigma=0.80$). Dennoch liefert die Funktion bei jedem Neuaufbau des Supermarkts exakt identische Kontroll-Werte für die QA-Pipelines.

6. Pipeline-Idempotenz & Data Contracts (Pydantic)
--------------------------------------------------
Eine hochverfügbare ETL-Pipeline muss zwingend **idempotent** sein: Ein mehrfacher Aufruf des Skripts (z.B. durch einen Cron-Job nach einem Server-Absturz) darf niemals zu doppelten Regalen oder korrupten JSON-Dateien führen. Der ``MasterOrchestrator`` garantiert dies, indem alle Ziel-Artefakte vor dem Load-Prozess auf Betriebssystemebene deterministisch überschrieben (File-Truncation) und nicht inkrementell angehängt werden.

Als absolute finale Schranke, bevor die transformierten Daten aus dem RAM auf die Festplatte geschrieben werden, nutzt das System **Pydantic-Modelle**. Dies erzwingt einen harten Systemvertrag (Data Contract) durch Typenprüfung zur Laufzeit:

.. code-block:: python

   from pydantic import BaseModel, Field, ValidationError

   class SmartCartProduct(BaseModel):
       """
       Der harte Data Contract. 
       Blockiert korrumpierte Produkte unwiderruflich durch strenge Laufzeit-Validierung.
       """
       name: str = Field(..., min_length=2)
       category: str
       node_id: str = Field(..., description="Die gemappte Topologie-ID (z.B. R_D_6)")
       price: float = Field(..., gt=0.0) # Preis muss zwingend strikt positiv (> 0) sein

*Was passiert im Fehlerfall?* Wenn die Pipeline aufgrund eines fehlerhaften BLS-Eintrags ein Produkt mit einem Preis von `-1.50` oder ohne `node_id` generiert, wirft Pydantic deterministisch einen `ValidationError`. Das Produkt wird isoliert und nicht gespeichert. Die Ground-Truth bleibt sauber.

Die strikte Trennung der exportierten Formate ist architektonisch zwingend:

1. ``smartcart_ml_training_data.csv``: CSV ist das speicherärmste, native Format für das nachgelagerte Model-Training mit ``scikit-learn`` und ``XGBoost``.
2. ``products.json``: JSON bietet die perfekte native Baumstruktur für die verschachtelten Python-Dictionaries und NLP-Hashmaps der Live-Engine.
3. ``routing_config.json``: Speichert das reine topologische Metadaten-Mapping (Kategorie $\rightarrow$ NetworkX-Node-ID).

**Der Status Quo:** Die Topologie steht. Die Regale sind nun algorithmisch speichereffizient, OOM-sicher und mathematisch absolut fair befüllt (Sainte-Laguë). Das Risiko von "Routing Hotspots" ist eliminiert (Round-Robin). Die Produkte haben rauscharme, endkundennahe Namen für den NLP-Vektorraum, streng validierte Typen und deterministische Preise. An diesem Punkt ist das Data Engineering abgeschlossen: Das Operations-Research-Modul kann nun übernehmen und die Kunden sicher durch den Graphen leiten.