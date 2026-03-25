Data Engineering & Supermarkt-Ontologie (ETL-Pipeline)
======================================================

Das vorherige Kapitel hat die räumliche Infrastruktur des JMU Smart Carts definiert: Ein Graphen-Modell des Supermarkts liegt als Digitaler Zwilling im Arbeitsspeicher vor. Doch an diesem Punkt der Systemarchitektur sind die Knoten (die physischen Regale) lediglich leere Container. 

Ein prädiktives Routing-System funktioniert nur mit einer absolut validen Datenbasis. Das Grundgesetz der Informatik *"Garbage In, Garbage Out"* (Müll rein, Müll raus) gilt hier rigoros: Wenn die Produktdatenbank fehlerhaft ist, navigiert der Dijkstra-Algorithmus den Kunden im besten Fall vor eine leere Wand und im schlimmsten Fall stürzt das Backend durch Exceptions ab. 

Anstatt rudimentäre Dummy-Daten von Hand zu schreiben, implementiert das System eine vollautonome **Data-Engineering-Pipeline nach dem ETL-Muster (Extract, Transform, Load)**. Gesteuert durch den zentralen ``MasterOrchestrator`` im Skript ``generate_data_driven_store.py`` lädt sie den echten "Bundeslebensmittelschlüssel" (BLS), generiert die Ground Truth für die KI, reinigt den Text von bürokratischem Rauschen und verteilt die Produkte algorithmisch auf die virtuellen Supermarkt-Regale, während sie synchron das Machine-Learning-Modell retrainiert.

Teil I: Extract – Network Resiliency & CI/CD Fallbacks
------------------------------------------------------
Die erste ingenieurtechnische Hürde ist der Umgang mit großen Datenmengen (Big Data). Der Bundeslebensmittelschlüssel ist eine massiv angewachsene B2B-Excel-Datei. 

Ein naiver Architekturansatz wäre es, die Datei über einen simplen HTTP-GET-Request vollständig in den RAM zu laden. Bei langsamen Verbindungen führt dies unweigerlich zu Memory-Spikes (OOM) oder Timeouts. Die Klasse ``BLSDownloader`` implementiert stattdessen einen speicherschonenden Downloader, der die Datei in 8192-Byte-Chunks asynchron über das Netzwerk zieht und in $O(1)$ RAM-Komplexität direkt auf die Festplatte streamt. 

*Graceful Degradation für Docker/CI-CD:* Um zu garantieren, dass die Pipeline auch in minimalen, Headless-Server-Umgebungen (ohne installierte Progress-Bar-Module wie ``tqdm``) nicht mit einem ``ImportError`` abstürzt, nutzt das System einen intelligenten Fallback.

.. code-block:: python

   # Graceful Degradation für Server ohne installiertes tqdm-Modul. 
   try:
       from tqdm import tqdm
       HAS_TQDM = True
   except ImportError:
       HAS_TQDM = False

   class BLSDownloader:
       def download(self) -> bool:
           try:
               # Ein harter Timeout von 60s verhindert Thread-Deadlocks
               response = requests.get(self.url, stream=True, timeout=60)
               response.raise_for_status()
               
               with open(self.dest_path, 'wb') as file:
                   if HAS_TQDM:
                       # ... [tqdm Progress Bar Logik] ...
                   else:
                       log.info("Lade Daten herunter... (Bitte warten)")
                       for chunk in response.iter_content(chunk_size=1024 * 8):
                           if chunk: file.write(chunk)
               return True
           except RequestException as e:
               return False

Teil II: Transform – Auto-Discovery, X-Features & Y-Targets
-----------------------------------------------------------
Nach dem Download übernimmt die ``CSVBuilder``-Klasse und nutzt das C-basierte Backend der Bibliothek ``pandas`` für die Datenverarbeitung. Hier müssen kritische Fehlerquellen von externen B2B-Daten eliminiert und die Daten für das Machine Learning vorbereitet werden.

2.1 Schema-Resilienz (Auto-Discovery der Spalten)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Behördliche Datensätze ändern häufig undeklariert ihr Layout. Wenn die ETL-Pipeline Spalten hartcodiert indiziert (z.B. ``df['Produktname']``), bricht die Architektur beim nächsten Datei-Update unweigerlich zusammen.
Das System implementiert als Gegenmaßnahme eine **Auto-Discovery-Heuristik**. Sie iteriert über die Spalten und analysiert Stichproben probabilistisch via Regular Expressions (RegEx) und einem Scoring-System, um die BLS-Codes und die Bezeichnungsspalte autonom zur Laufzeit zu identifizieren.

2.2 Semantische Reduktion (Das X-Feature für die KI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Behördliche Kataloge sind bilingual und voller Labor-Jargon (z.B. *"Schweineschnitzel, intermuskulär, rohmasse"*). Für das Machine-Learning-Modell (TF-IDF N-Gramme) ist das toxisch. Ein englisches Wort ("beef") oder ein Labor-Begriff ("kutterhilfsmittel") würde die Matrix unnötig aufblähen (Curse of Dimensionality) und zu massivem Data Leakage führen.

Die Klasse ``TextSanitizer`` wendet Vektor-Operationen an, um den Text radikal auf den semantischen Kern (das X-Feature) für den Endkonsumenten zu reduzieren:

.. code-block:: python

   class TextSanitizer:
       @staticmethod
       def clean(name: str) -> Optional[str]:
           name_lower = str(name).lower()
           
           # 1. B2B Filter: Verhindert Overfitting auf toxisches Labor-Vokabular
           if any(noise in name_lower for noise in Config.B2B_NOISE_WORDS): return None
               
           # 2. Englisch Filter: Reduziert Sparse-Matrix Dimensionalität im TF-IDF
           for en_word in Config.EN_KILLER_WORDS:
               if re.search(r'\b' + en_word + r'\b', name_lower): return None
                   
           # 3. String-Kappung: Schneidet alles ab dem ersten Komma ab.
           name = name.split(',')[0]
           
           # 4. RETAIL-FILTER: Maximal 3 Wörter erlaubt.
           # Verhindert toxische Class Imbalance durch überlange B2B-Sätze.
           name = re.sub(r'\s+', ' ', name).strip()
           word_count = len(name.split())
           if word_count == 0 or word_count > 3: return None
               
           return name

2.3 Ground-Truth Generierung (Das Y-Target der KI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ein Klassifikationsmodell benötigt zwingend gelabelte Trainingsdaten (X $\rightarrow$ Y). Die rohe Excel-Datei liefert jedoch nur kryptische Behörden-Codes (z.B. "E1" oder "M"). 

Die Klasse ``SupermarketMapper`` fungiert als elementarer Y-Target Generator. Sie übersetzt diese kryptischen BLS-Codes über eine deterministische Mapping-Tabelle und harte semantische Keyword-Heuristiken in unsere Supermarkt-Ontologie (z.B. "Kühlregal (Molkerei & Veggie)"). Erst durch diesen massiven Data-Engineering-Schritt erhält die Logistische Regression überhaupt valide Ziel-Klassen (Ground Truth), auf die sie trainieren kann.

Teil III: Load – Algorithmische Graphen-Befüllung & Type Safety
---------------------------------------------------------------
Nach der Transformation und dem KI-Training (TF-IDF Matrix) mappt der ``MasterOrchestrator`` die bereinigten Produkte auf die topologischen Container-Knoten des Graphen. Hier müssen vier komplexe Probleme der Operations Research gelöst werden.

3.1 Die Verhinderung von Null-Allokationen (Sainte-Laguë)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Das System muss hunderte Produkte fair auf eine limitierte Anzahl flexibler Regale (``FLEXIBLE_ZONES``) verteilen. Würde das System die Regale per normaler Prozentrechnung verteilen und abrunden, bekäme eine kleine Kategorie mathematisch $0.48 \rightarrow 0$ Regale zugewiesen. Ein Routing-Versuch in "null" Regale würde das Backend abstürzen lassen.

Die Architektur nutzt daher einen Algorithmus aus dem parlamentarischen Wahlrecht: Das **Sainte-Laguë-Verfahren**. Es nutzt den Divisor ``(allocation[cat] + 0.5)``. Dieser Divisor garantiert mathematisch, dass selbst die kleinste Kategorie zwingend *mindestens ein* physisches Regal zugewiesen bekommt, bevor riesige Kategorien ihr sechstes Regal erhalten.

3.2 Quengelware (Topologische Determiniertheit an Kassen)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Während Sainte-Laguë für flexible Sortimente greift, gibt es Produkte, die sich einer proportionalen Fairness entziehen. Sogenannte **Quengelware** (Impulskäufe wie Batterien oder Kaugummis) wird strategisch starr in den Wartezonen der Kassen platziert.

Das System implementiert dies durch die Methode ``_allocate_kassen()``. Diese bypassiert die dynamische Zuteilung vollständig und injiziert die Impuls-Produkte deterministisch in die hartcodierten Warteschlangen-Knoten (``vW1``, ``vW2``, ``vW3``). Dies beweist, dass der Digitale Zwilling reale verhaltensökonomische Verkaufskonzepte auf Code-Ebene adaptiert.

3.3 Greedy Capacity Balancing & Der Numpy-Serialization Crash
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Die Produkte werden durch **Greedy Capacity Balancing** über ``min()`` deterministisch immer in das Regal sortiert, das aktuell am leersten ist. Dadurch wird das Kundeninteresse automatisch homogen auf die Zone gestreut (präventives Load-Balancing).

*Architektonischer Deep-Dive (Type-Safety):* Ein tückischer Bug, der in klassischen Pipelines zu fatalen Produktionsausfällen führt, ist die Inkompatibilität von C-basierten Datentypen. Scikit-Learn operiert intern auf Numpy-C-Strukturen (z.B. ``np.float64``). Versucht man, diese KI-Konfidenzwerte direkt in die JSON-Datenbank zu schreiben, wirft die Standard-Bibliothek den fatalen Fehler ``TypeError: Object of type float64 is not JSON serializable``.

Die Pipeline behebt dies durch einen rigorosen Type-Cast (``float(...)``), der den C-Pointer zwingend in einen nativen CPython-Float umwandelt:

.. code-block:: python

   for _, row in cat_items.iterrows():
       available = [n for n in nodes if self.capacity[n] < StoreOntology.MAX_CAPACITY]
       if not available: break 
       
       best_node = min(available, key=lambda n: self.capacity[n])
       
       self.stock[best_node].append({
           'name': row['Name'], 'category': cat,
           'is_frozen': row['IsFrozen'], # Cold-Chain Injection
           
           # TYPE-SAFETY FLEX: Verhindert den fatalen JSON Serialization Crash
           # Konvertiert np.float64 hart in den nativen Python-Float
           'ai_confidence': float(round(row['Conf'], 3)), 
           
           'suggested_slot': best_node
       })
       self.capacity[best_node] += 1

3.4 Das Emergency Fallback (Graceful Degradation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ein reales Produktionsrisiko: Was passiert, wenn die KI aufgrund strenger Konfidenz-Schwellenwerte nicht genügend Produkte einer bestimmten Kategorie findet, um die zugewiesenen Regale zu füllen? Ein leeres Regal würde das Dashboard topologisch zerstören.

Die Pipeline nutzt hier das architektonische Prinzip der **Graceful Degradation**. Schlägt das Befüllen fehl, greift das System auf das hartcodierte Dictionary ``StoreOntology.EMERGENCY_FALLBACKS`` zurück. Das System injiziert vollautomatisch extrem stark nachgefragte synthetische Notfall-Daten (z.B. "Standard Milch"), um das Backend-Routing und die Frontend-Darstellung abzusichern.

Teil IV: Cold-Chain Injection & MLOps Sync
------------------------------------------
Ein elementarer Business-Case für den Smart Cart ist die Einhaltung der Kühlkette (Cold Chain). Anstatt diese Logik später rechenaufwändig im Backend zu parsen, injiziert die ETL-Pipeline das topologische Metadatum ``is_frozen`` direkt in die Datenstruktur. Das Operations-Research-Modul kann Tiefkühlware so in $O(1)$ identifizieren und durch mathematische Strafen ans Ende der Einkaufsroute verschieben.

*Prävention von Concept Drift:* Da das Skript ``generate_data_driven_store.py`` alle Phasen orchestriert, wird das Problem des Concept Drifts im Keim erstickt. Wenn die Pipeline neue Produkte in den Supermarkt lädt, erzwingt der ``MasterOrchestrator`` ein kompromissloses **Synchronous Retraining**, indem er das alte ML-Artefakt (``.pkl``) hart löscht und das Modell zwingend synchron auf der neu generierten Ground Truth trainiert.

Am Ende der Pipeline werden die generierten JSON-Artefakte über Thread-Locks (``inv_manager._lock``) atomar materialisiert. Das Backend verfügt nun über einen vollständig validierten, physikalisch balancierten und maschinenlesbaren Digitalen Zwilling.