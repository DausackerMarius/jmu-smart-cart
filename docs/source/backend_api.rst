Backend-Architektur & System-Design (Live-Engine & model.py)
============================================================

Die theoretische Konzeption von Graphenalgorithmen und Machine-Learning-Modellen ist in der modernen Informatik oftmals nur das mathematische Fundament. Die weitaus größere ingenieurtechnische Herausforderung (Software Engineering & MLOps) besteht in der Praxis jedoch darin, diese extrem rechenintensiven Modelle in eine serverseitige Architektur zu gießen, die in der realen Welt eines Supermarkts auch unter extremer Last standhält. 

Ein solches Backend muss gigantische Datenmengen verarbeiten, fehlerhafte Benutzereingaben der Tablets auf unterster Netzwerkebene abfangen, das berüchtigte Python-GIL umgehen und dabei konstante Antwortzeiten im Millisekundenbereich garantieren. 

Dieses Kapitel widmet sich dem „Maschinenraum“ des JMU Smart Cart Projekts. Es dokumentiert den lückenlosen Datenfluss und die Design Patterns der Live-Engine – von der speichersicheren Initialisierung (State Hydration) über die NLP-Suchalgorithmen bis hin zur kybernetischen Auslieferung der fertigen Route.

1. Architektonisches Paradigma: Zero-Disk I/O & State Hydration
---------------------------------------------------------------
Das oberste architektonische Gebot für den Live-Betrieb des Backends lautet: **Zero Disk I/O**. Sobald ein Kunde im Supermarkt auf dem Tablet auf "Route berechnen" tippt, darf der Server keine einzige Millisekunde mehr mit dem Lesen von langsamen SSDs oder dem Warten auf relationale SQL-Datenbanken verschwenden. Die Latenz muss konstant bei $\mathcal{O}(1)$ für den Datenzugriff liegen.

Um dies code-technisch sauber zu lösen, nutzt die Architektur das asynchrone ``lifespan``-Konzept des **FastAPI**-Frameworks. Bevor der Webserver überhaupt die ersten HTTP-Anfragen auf Port 8000 akzeptiert, werden alle topologischen Modelle und Machine-Learning-Gewichte exakt einmalig in den Arbeitsspeicher (RAM) geladen. Dieser Prozess nennt sich **State Hydration**.

.. code-block:: python

   import pickle
   import json
   from contextlib import asynccontextmanager
   from fastapi import FastAPI
   import networkx as nx

   # Globale, In-Memory Container für den Graphen und die KI.
   # Durch das Pre-Fork-Worker Modell (siehe Kap. 2.2) ist dieser State thread-safe.
   app_state = {}

   @asynccontextmanager
   async def lifespan(app: FastAPI):
       """
       Der Boot-Prozess des Servers (Pre-Flight). Hydriert alle statischen Daten 
       in den RAM, bevor der erste Smart Cart den Supermarkt betritt.
       """
       try:
           # 1. JSON-Parsing (C-optimiert in Python) für topologische Metadaten
           with open("routing_config.json", "r") as f:
               app_state["topology_data"] = json.load(f)
               
           # 2. Binäre Entpackung des ML-Modells via Pickle
           # Pickle lädt den exakten C-Memory-State des XGBoost-Entscheidungsbaums 
           # in Bruchteilen einer Sekunde direkt in den L3-Cache der CPU.
           with open("traffic_model.pkl", "rb") as f:
               app_state["traffic_predictor"] = pickle.load(f)
               
           # 3. Graphen-Rekonstruktion
           # Der Graph wird einmalig im RAM als NetworkX-Objekt materialisiert
           app_state["base_graph"] = nx.node_link_graph(app_state["topology_data"])
           
       except FileNotFoundError as e:
           # Fail-Fast-Prinzip: Der Server crasht absichtlich und lautstark, 
           # falls Ground-Truth-Daten fehlen, statt später ins Leere zu routen.
           raise RuntimeError(f"CRITICAL BOOT FAILURE: Missing artifact - {e}")
           
       # Yield gibt die Kontrolle an den Uvicorn-Server zurück: System ist nun LIVE.
       yield 
       
       # Teardown: Clean-Up Ressourcen bei regulärem Server-Shutdown zur Vermeidung von Memory Leaks
       app_state.clear()

   app = FastAPI(lifespan=lifespan)

**Die architektonische Message:** Durch diesen Ansatz ist die Applikation zur Laufzeit komplett in-memory. Das Backend agiert fortan als zustandsloser (stateless) Microservice, der Anfragen isoliert verarbeitet.

2. Der API-Vertrag: Cyber Security & Typensicherheit
----------------------------------------------------
Das Backend muss zwingend davon ausgehen, dass der eingehende Netzwerk-Traffic der Tablets fehlerhaft oder durch einen Man-in-the-Middle manipuliert ist. Ein Angreifer könnte versuchen, durch modifizierte HTTP-Requests hunderte unsinnige Artikel oder gigantische Zeichenketten als Zielprodukte zu senden. Da Python eine dynamisch typisierte Sprache ist, würde das Routing-Modul ohne Schutzmaßnahmen erst tief im Code abstürzen (z.B. durch eine `RecursionError` bei exponentiellen TSP-Berechnungen).

2.1 Harte Schema-Validierung mit Pydantic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Um das System auf Netzwerkebene abzusichern, fungiert das Framework **Pydantic** als unbestechlicher Türsteher. Wir nutzen nicht nur statische Typen, sondern implementieren *Custom Validators*, die bösartige Muster filtern, noch bevor der CPU-intensive Routing-Code aufgerufen wird.

.. code-block:: python

   from pydantic import BaseModel, Field, field_validator
   from typing import List
   import re

   class RouteRequest(BaseModel):
       """
       Der strikte Datenvertrag (Data Contract). Blockiert fehlerhafte Payloads 
       deterministisch vor der Ausführung der Business-Logik.
       """
       # Hard-Limit auf max 50 Items verhindert "Algorithmische DDoS-Angriffe" 
       # auf den exponentiellen Held-Karp TSP-Solver.
       target_items: List[str] = Field(..., min_items=1, max_items=50)
       
       # Verfolgt den Fortschritt des Einkaufs für das stochastische Checkout-Timing
       cart_progress: float = Field(..., ge=0.0, le=1.0)
       
       @field_validator('target_items')
       def validate_strings(cls, items: List[str]) -> List[str]:
           """Verhindert Code-Injections und filtert unlesbare Sonderzeichen."""
           clean_items = []
           for item in items:
               # Trimmen und Längenlimitierung schützt den Damerau-Levenshtein-Algorithmus
               item = item.strip()
               if len(item) > 40:
                   raise ValueError(f"Security Warning: Suchbegriff '{item[:10]}...' zu lang.")
                   
               # Erlaubt ausschließlich alphanumerische Zeichen und deutsche Umlaute
               if not re.match(r"^[a-zA-Z0-9äöüÄÖÜß\s\-_]+$", item):
                   raise ValueError(f"Security Warning: Ungültige Zeichen in '{item}'")
               clean_items.append(item)
           return clean_items

Schlägt die Validierung fehl, wirft Pydantic deterministisch einen Error `422 Unprocessable Entity`. Die Server-CPU wird gar nicht erst belastet, und das Frontend erhält eine saubere Fehlerbeschreibung.

2.2 Das GIL-Problem und WSGI-Concurrency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ein Backend für Supermärkte muss hunderte smarte Einkaufswagen zeitgleich bedienen (Concurrency). Moderne Web-Frameworks nutzen hierfür oft asynchrone Event-Loops (``async/await``). Für dieses System wäre das jedoch ein katastrophaler Architektur-Fehler.

Der Grund liegt im **Global Interpreter Lock (GIL)** von Python, das echtes Multithreading auf CPU-Ebene verhindert. Python kann pro Prozess immer nur *einen* Befehl gleichzeitig ausführen. Unsere TSP-Routenberechnung ist jedoch pure, *CPU-gebundene* Mathematik im RAM. Ein schwerer, dreisekündiger Routing-Task für Kunde A würde den asynchronen Event-Loop komplett blockieren. Alle anderen Kunden im Supermarkt würden in dieser Zeit "einfrieren", da der Server nicht antwortet.

**Die Lösung:** Die Architektur nutzt einen **Gunicorn-Server** mit einem **Pre-Fork-Worker-Modell**. Anstelle von sich blockierenden Threads weist der Master-Prozess das Betriebssystem (OS) an, sich zu klonen (Forking). Es entstehen völlig autarke Python-Prozesse (Worker), die das GIL umgehen. Führt Worker A schwere Mathematik aus, bedient Worker B latenzfrei den nächsten Kunden.

3. Komponenten-Architektur: model.py
------------------------------------
Um unwartbaren "Spaghetti-Code" zu vermeiden, folgt die Architektur der ``model.py`` strikt dem *Single Responsibility Principle*. 

3.1 Die SearchEngine: Damerau-Levenshtein & Memoization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Wenn ein Nutzer im Gehen "Kaffe" statt "Kaffee" tippt, muss der ``SearchKernel`` diesen Fehler abfangen und auf die harte Knoten-ID des Graphen mappen. Eine SQL-Abfrage (``LIKE '%Kaffe%'``) wäre hierfür zu I/O-intensiv. 

Das Inventar liegt als Hash-Map im RAM (Latenz $\mathcal{O}(1)$). Findet das System keinen exakten Treffer, nutzt es die **Damerau-Levenshtein-Distanz**. Im Gegensatz zur Standard-Levenshtein-Distanz, die einen simplen Buchstabendreher ("ei" zu "ie") als *zwei* Fehler wertet (Löschen + Einfügen), wertet die Damerau-Erweiterung (Transposition) dies als nur *einen* Fehler. Da Buchstabendreher auf Touchscreens extrem häufig sind, steigt die Toleranz massiv.

**Der Code-Beweis (Dynamische Programmierung):**

.. code-block:: python

   from functools import lru_cache

   class FuzzySearchEngine:
       @staticmethod
       @lru_cache(maxsize=2048) # Memoization: Hält die letzten 2048 Suchen im schnellen Cache
       def damerau_levenshtein(s1: str, s2: str) -> float:
           """
           Berechnet die topologische Editier-Distanz via Dynamischer Programmierung.
           Laufzeitkomplexität: O(n * m), Speicherkosten: O(n * m).
           """
           len1, len2 = len(s1), len(s2)
           # 2D-Matrix Initialisierung
           d = [[0] * (len2 + 1) for _ in range(len1 + 1)] 
           
           for i in range(len1 + 1): d[i][0] = i
           for j in range(len2 + 1): d[0][j] = j
           
           for i in range(1, len1 + 1):
               for j in range(1, len2 + 1):
                   cost = 0 if s1[i-1] == s2[j-1] else 1
                   
                   # Standard Levenshtein: Löschen, Einfügen, Ersetzen
                   d[i][j] = min(
                       d[i-1][j] + 1,      
                       d[i][j-1] + 1,      
                       d[i-1][j-1] + cost  
                   )
                   # Damerau-Erweiterung: Überprüfung auf benachbarte Vertauschungen
                   if i > 1 and j > 1 and s1[i-1] == s2[j-2] and s1[i-2] == s2[j-1]:
                       d[i][j] = min(d[i][j], d[i-2][j-2] + cost)
                       
           # Umrechnung der absoluten Distanz in prozentuale Übereinstimmung
           match_percentage = (1.0 - (d[len1][len2] / max(len1, len2))) * 100.0
           return match_percentage

**Die Message:** Diese $\mathcal{O}(n \cdot m)$ Matrix-Traversierung ist rechenintensiv. Durch den ``@lru_cache``-Dekorator (**Memoization**) muss dieser Code für denselben Tippfehler (z.B. "Kaffe") jedoch nur ein einziges Mal pro Worker berechnet werden. Folgt eine zweite Anfrage, wird das Ergebnis aus dem RAM serviert, was die Server-CPU massiv entlastet.

3.2 Traffic Prediction (Das Pass-by-Reference Problem)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Der statische Graph kennt nur Meter. Die Klasse ``TrafficPredictor`` mutiert die Kanten "on-the-fly" mit prädiktiven Zeitstrafen der KI.

*Ein architektonisch kritischer Code-Beweis:* In Python werden komplexe Objekte (Graphen, Arrays) nicht als isolierte Werte, sondern als **Speicher-Referenzen** (Pointer) übergeben. Würde das System den Master-Graphen aus `app_state` direkt mutieren, blieben Staus für immer im globalen RAM. Die Routen des nächsten Morgens wären fälschlicherweise vom abendlichen Stau betroffen. Der tiefe ``.copy()``-Aufruf ist daher zwingend.

.. code-block:: python

   import numpy as np
   import networkx as nx

   def apply_dynamic_traffic_penalties(base_graph: nx.DiGraph, ml_model, current_time) -> nx.DiGraph:
       """
       Mutiert den Graphen mit KI-Strafen, strikt isoliert auf einem temporären Kontext.
       """
       # TIEFE KOPIE: Schützt den Master-Graphen zwingend vor permanenter Memory-Corruption
       G_temp = base_graph.copy() 
       
       hour = current_time.hour
       is_weekend = 1 if current_time.weekday() >= 5 else 0
       
       for u, v in G_temp.edges():
           base_weight = G_temp[u][v]['weight']
           
           # Feature Tensor für XGBoost: (Stunde, Wochenende, Basisdistanz)
           features = np.array([[hour, is_weekend, base_weight]])
           
           # ML-Prädiktion. Clamping (max(0)) verhindert unmögliche, negative Staus.
           predicted_load = max(0.0, ml_model.predict(features)[0])
           
           # Algorithmische Translation: Stau = Künstliche Verlängerung der Graphen-Kante
           G_temp[u][v]['weight'] += (predicted_load * SystemConfig.PENALTY_FACTOR)
           
       return G_temp

4. Pipeline-Orchestrierung: calculate_hybrid_route()
----------------------------------------------------
Diese Hauptfunktion ist der Dirigent des Systems. Ein monolithischer Code voller ``if-else`` Blöcke für die hochkomplexen Algorithmen (Dijkstra, Held-Karp, SA) wäre unwartbar. Stattdessen isoliert das System jeden TSP-Algorithmus über das **Strategy Pattern** in einer eigenen Klasse.

**Das eiserne Timing-Gesetz der Stochastik:**
Zudem wird in dieser Schicht eine fundamentale Systemvorgabe durchgesetzt: Die stochastische Kassen-Vorhersage (welche Kasse am schnellsten abfertigt) **muss zwingend vor Erreichen der Routen-Halbzeit** in die Distanzmatrix injiziert werden. Wartet das System zu lange, fehlt dem TSP-Solver auf den letzten Metern der topologische Manövrierraum, um den Kunden ohne abrupte Kehrtwenden zur besten Kasse zu leiten. 

.. code-block:: python

   from core.interfaces import RoutingStrategy
   from core.strategies import HeldKarpStrategy, SimulatedAnnealingStrategy
   from core.stochastics import CheckoutFacade
   import networkx as nx

   def synthesize_final_route(
       shopping_nodes: List[str], 
       cart_progress: float, 
       dist_matrix: dict
   ) -> dict:
       """
       Der zentrale Orchestrator. Verschmilzt Graphentheorie, Stochastik und Routing.
       """
       n = len(shopping_nodes)
       
       # 1. Zwingender Halftime-Trigger für Stochastik-Vorhersagen
       optimal_checkout_node = "DEFAULT_EXIT"
       
       # VERIFIKATION: Die Vorhersage ist ausschließlich in der ersten Hälfte des 
       # Einkaufs erlaubt (cart_progress <= 0.5). Dies ist ein hartes Architektur-Gesetz,
       # um Deadlocks vor der Kassenzone algorithmisch auszuschließen.
       if cart_progress <= 0.5:
           facade = CheckoutFacade()
           optimal_checkout_node = facade.predict_optimal_checkout()
       else:
           # Fallback, falls die Applikation das Halftime-Window verpasst hat
           optimal_checkout_node = determine_nearest_physical_checkout(dist_matrix)
       
       # 2. Polymorphe Algorithmus-Zuweisung (Strategy Pattern)
       solver: RoutingStrategy = None
       
       if n <= SystemConfig.DP_EXACT_LIMIT: # Limit 15 Produkte
           # O(n^2 * 2^n) - Deterministische DP-Suche für absolute Exaktheit
           solver = HeldKarpStrategy()      
       else:
           # Thermodynamische Metaheuristik zur OOM-Vermeidung bei Großeinkäufen
           solver = SimulatedAnnealingStrategy() 
           
       # 3. Ausführung (Der Dirigent ruft dynamisch nur das abstrakte Interface auf)
       route, exec_time = solver.solve(dist_matrix, shopping_nodes, optimal_checkout_node)
       
       return {"route": route, "compute_time_ms": exec_time}

5. Quality Assurance (QA), Mocking & Resilienz
----------------------------------------------
Ein Live-System im Retail darf nicht zur Laufzeit abstürzen. Die Codebase ist via ``mypy`` statisch durchtypisiert (PEP 484). 

Für die isolierten Unit-Tests (via ``pytest``) nutzt das System konsequent **Mocking**. 
*Die Message:* Beim Mocking ersetzt man reale, langsame Systemkomponenten durch Attrappen. Anstatt das echte, 50 MB große XGBoost-Modell bei jedem kurzen Testlauf in den RAM zu pressen, liefert das ``unittest.mock`` Modul feste Dummy-Werte. Dies entkoppelt die Graphen-Tests vollständig von der ML-Pipeline und hält die CI/CD-Prozesse pfeilschnell.

.. code-block:: python

   from unittest.mock import patch

   @patch('model.TrafficPredictor.predict', return_value=[15.0]) # Zwingt das Modell, "15 Personen" zurückzugeben
   def test_graph_penalty_logic(mock_predict):
       """
       Beweist, dass die Graphen-Mutation funktioniert, ohne das ML-Modell zu laden.
       """
       G_mutated = apply_dynamic_traffic_penalties(dummy_graph, mock_model, mock_time)
       
       # Assert-Beweis: Das Kantengewicht im RAM muss nach der Mutation 
       # strikt größer sein als die rein physische Basis-Distanz.
       assert G_mutated["A"]["B"]["weight"] > dummy_graph["A"]["B"]["weight"]

6. System-Evaluation: Ablationsstudie (Baseline vs. KI-Agent)
-------------------------------------------------------------
In der akademischen Informatik beweist man die Leistungsfähigkeit eines Systems durch eine **Ablationsstudie** (Ablation Study). Man fragt sich: *Was passiert, wenn wir die KI-Komponenten abschalten und das Supermarkt-Problem mit klassischer Basis-Informatik lösen?*

6.1 Mechanik der deterministischen Baseline (Der "blinde" Algorithmus)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Die Baseline repräsentiert klassische Navigationssysteme. Die Architektur ist rein *geometrisch*. Das Kantengewicht $W_{edge}$ entspricht in der Distanzmatrix stur der metrischen Länge des Ganges (z.B. Hauptgang = 10m, Umweg = 25m). Ein Standard-Dijkstra wählt hier zwingend den 10m Gang.

**Das systematische Versagen:** Die Baseline ist "stau-blind". Wenn sich im kurzen Hauptgang zur Rushhour 40 Menschen stauen, schickt die Baseline den Kunden deterministisch in diesen Flaschenhals, weil 10 Meter algorithmisch immer kürzer sind als 25 Meter. An der Kasse wählt die Baseline am Ende des Einkaufs per *Greedy-Heuristik* die räumlich nächste Kasse, selbst wenn dort 15 Kunden warten.

6.2 Mechanik des intelligenten KI-Agenten
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unser Backend verwirft die reine Längenmessung. Stattdessen mutiert es den Graphen und rechnet in der Kostenfunktion der **echten Zeit**. 

Das Machine Learning generiert Straf-Metriken für überlastete Gänge. Für den Routing-Algorithmus erscheint der physikalisch 10 Meter kurze Gang im Arbeitsspeicher plötzlich so, als wäre er zeitlich 100 Meter lang. Der Agent leitet den Kunden völlig autonom über den physisch längeren, aber zeitlich signifikant schnelleren 25-Meter-Umweg. Gepaart mit dem Stochastik-Modul, das wie oben im Code bewiesen zwingend *vor Erreichen der Halbzeit* die optimale Kasse prädiziert, wird die kognitive Belastung des Kunden minimiert.

.. list-table:: Architektur-Vergleich: Deterministische Baseline vs. KI-Agent
   :widths: 20 40 40
   :header-rows: 1

   * - Kriterium
     - Baseline-Architektur (Der Standard)
     - JMU Smart Cart KI-Agent
   * - **Routing-Metrik**
     - $W_{edge} = d(x,y)$ (Minimiert physikalische Meter)
     - $W_{edge} = d(x,y) + f_{ml}(x,y)$ (Minimiert reale Zeitlatenz)
   * - **Graphen-Zustand**
     - Statisch. Keine Mutationen im RAM.
     - Dynamisch. Gänge werden "on-the-fly" virtuell verlängert.
   * - **Kassen-Auswahl**
     - Greedy-Heuristik. Erfolgt blind am Ende des Einkaufs.
     - Stochastisch (M/M/1/K). Früher Halftime-Trigger garantiert den perfekten Exit.
   * - **Nutzer-Erlebnis**
     - Führt zielsicher in physische Staus.
     - Navigiert reibungslos und unsichtbar um jeden Flaschenhals herum.

7. Big-O Komplexitätsmatrix
---------------------------
Die abschließende Matrix beweist die theoretische Laufzeiteffizienz der Architektur im Worst-Case.

.. list-table:: Laufzeitkomplexitäten der System-Komponenten
   :widths: 30 45 25
   :header-rows: 1

   * - System-Komponente
     - Algorithmus / Datenstruktur
     - Zeitkomplexität (Big-O)
   * - Such-Engine (Best-Case)
     - Hash-Map Lookup (In-Memory Index)
     - $\mathcal{O}(1)$
   * - Such-Engine (Worst-Case)
     - Damerau-Levenshtein (DP-Matrix)
     - $\mathcal{O}(n \cdot m)$
   * - Topologie-Mutation
     - Feature-Vector Tensor Mapping
     - $\mathcal{O}(|E|)$
   * - Distanzmatrix (Clique)
     - Heuristikfreier Multiple Dijkstra
     - $\mathcal{O}(V \cdot (V+E)\log V)$
   * - Routing (Kleine Listen)
     - Held-Karp (Dynamische Programmierung)
     - $\mathcal{O}(n^2 \cdot 2^n)$
   * - Routing (Mittlere Listen)
     - Simulated Annealing (Thermodynamik)
     - $\mathcal{O}(\text{Iterationen})$
   * - Checkout-Routing
     - Queuing Model Facade (M/M/1/K)
     - $\mathcal{O}(\text{Kassenanzahl})$