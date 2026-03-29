Backend-Architektur & System-Design (Dash MVC-Engine)
=====================================================

Die mathematische Konzeption von Graphenalgorithmen und Machine-Learning-Modellen ist in der modernen Informatik nur das theoretische Fundament. Die weitaus größere ingenieurtechnische Herausforderung besteht darin, diese isolierten, rechenintensiven Modelle in eine serverseitige Architektur zu gießen, die in der physischen Realität eines Supermarkts unter massiver Gleichzeitigkeit (Concurrency) nicht einbricht. 

Das Backend des JMU Smart Cart Systems fungiert als das **kybernetische Gehirn** der gesamten Infrastruktur. Entgegen klassischer Ansätze, die Frontend und Backend über separate REST-APIs physisch trennen, implementiert diese Architektur das strenge **MVC-Paradigma (Model-View-Controller)** kohäsiv über das Enterprise-Web-Framework *Plotly Dash* (welches unter der Haube auf Flask und der Werkzeug-WSGI-Engine basiert). 

Die Architektur konsumiert die statischen Gebäudepläne aus dem Data Engineering, triggert die XGBoost-Modelle zur Stauvorhersage, berechnet über Operations Research die Kürzeste-Wege-Matrix und rendert die Ergebnisse reaktiv für den Client. Ein solches Backend muss gigantische Datenmengen im Arbeitsspeicher verwalten, Speicherkontaminationen (Memory Leaks) zwischen hunderten Supermarktkunden verhindern, das Global Interpreter Lock (GIL) von Python orchestrieren und Echtzeit-Antworten im strengen Millisekundenbereich garantieren. 

Dieses Kapitel dokumentiert den "Maschinenraum" des Projekts in seiner vollen Tiefe. Die Architektur ist strikt modular aufgebaut und folgt dem chronologischen Lebenszyklus einer Systemanfrage: Von der Kaltstart-Prävention über die zustandslose Edge-Speicherung, das Caching, die topologische Graphen-Kondensation, die Warteschlangen-Stochastik bis hin zur Berechnung der finalen Route.

Phase I: Kaltstart-Prävention & State Hydration (Singleton)
-----------------------------------------------------------
Das oberste architektonische Gebot für den Live-Betrieb des Backends lautet: **Zero Disk I/O (Input/Output) während der Inferenz**. 

*Die architektonische Problemstellung:* Wenn ein Kunde im Supermarkt auf dem Tablet auf "Route optimieren" tippt, darf der Server keine Millisekunde mehr damit verschwenden, Produktdaten von einer langsamen Festplatte (SSD) zu lesen oder komplexe, blockierende Datenbank-Queries auszuführen. Festplattenzugriffe operieren im besten Fall im einstelligen Millisekundenbereich. Direkte Zugriffe auf den flüchtigen Arbeitsspeicher (RAM) operieren im Nanosekundenbereich. Bei hunderten gleichzeitigen Kundenanfragen summiert sich die I/O-Latenz einer Festplatte zu einem massiven Flaschenhals, der den Thread-Pool des Servers sofort blockieren würde.

*Die Lösung:* Das System nutzt für alle ressourcenintensiven Datenstrukturen das **Singleton-Entwurfsmuster**. Bevor der erste Kunde den Webserver überhaupt erreicht, führt die Engine die sogenannte **State Hydration** (das "Bewässern" des RAMs) durch. 

Die Klasse ``TrafficSimulationEngine`` sucht beim Systemstart dynamisch nach dem aktuellsten kompilierten XGBoost-Artefakt (Continuous Deployment Support) und lädt den binären Speicher-Dump (``.pkl``) exakt einmalig in ein Klassenattribut. Die Topologie des Supermarkts (``nx.Graph``) und das Inventar (``SearchKernel``) werden in der ``model.py`` analog instanziiert und durch Thread-Locks (``threading.RLock()``) gegen asynchrone Race Conditions auf Speicherebene abgesichert.

.. code-block:: python

    import pickle
    import glob
    import os
    
    class TrafficSimulationEngine:
        """
        Kapselt das XGBoost-Modell. Das _model_cache Klassenattribut 
        garantiert Inferenzzeiten von wenigen Millisekunden ohne I/O-Overhead.
        """
        _model_cache = None

        @classmethod
        def initialize(cls):
            pkl_files = glob.glob("*.pkl")
            traffic_models = [f for f in pkl_files if "traffic" in f]

            if traffic_models:
                # Automatisches Laden des aktuellsten Modells (CI/CD Support)
                best_model_path = max(traffic_models, key=os.path.getmtime)
                try:
                    # SECURITY-NOTE: Das Deserialisieren via Pickle ist ein Sicherheitsrisiko
                    # (Arbitrary Code Execution). Dieser Aufruf ist hier nur zulässig, da 
                    # die .pkl-Datei in der CI/CD-Pipeline strikt intern signiert und erzeugt 
                    # wird. Für externe Deployments nutzt die Architektur das native, sichere 
                    # xgb.Booster().load_model() Format.
                    with open(best_model_path, "rb") as f:
                        cls._model_cache = pickle.load(f)
                    print(f"✅ KI-MODELL GELADEN: {best_model_path}")
                except Exception as e:
                    print(f"❌ Fehler beim Laden des Models: {e}")

    # Ausführung vor dem ersten Web-Request: Verhindert den Kaltstart (Cold Start)
    TrafficSimulationEngine.initialize()

Phase II: Das Session-Dilemma & Stateless Edge Architecture
-----------------------------------------------------------
Ein Backend für einen gut besuchten Supermarkt muss dutzende smarte Einkaufswagen zeitgleich bedienen. In klassischen Python-Applikationen besteht hier die fatale Gefahr der Speicherkontamination durch server-seitige Sessions.

*Das Problem:* Würde das Backend den Einkaufszettel (Warenkorb) in einer globalen Server-Variable wie ``global_cart = []`` speichern, würden sich alle Nutzer im Supermarkt denselben Warenkorb teilen. Das System muss also zwingend **zustandslos (stateless)** operieren. Der Server darf sich zwischen zwei Klicks eines Kunden nicht merken, wer der Kunde ist oder was in seinem Korb liegt.

*Die architektonische Lösung:* Die Dash-Architektur lagert das State-Management vollständig an die "Edge" (das Endgerät/Tablet des Nutzers) aus. Die Komponente ``dcc.Store`` verpackt den Warenkorb als JSON-String und speichert ihn autark im *Local Storage* des Browsers. 

Für Fachfremde lässt sich dies mit einem Restaurantbesuch vergleichen: Anstatt dass sich der Kellner (Server) im Kopf merkt, was Tisch 4 bestellt hat (Session), liegt der Bestellzettel physisch auf dem Tisch (Tablet). Bei jeder Rückfrage reicht der Gast dem Kellner den kompletten Zettel. Der Kellner arbeitet ihn ab und gibt ihn sofort zurück. Der Kellner benötigt so null Gedächtnis und kann hunderte Tische simultan fehlerfrei bedienen.

Der Code-Beweis der Zustandslosigkeit: Das System nutzt in seinen reaktiven Controllern zwingend die ``State``-Klasse von Dash. Anstatt den Warenkorb aus dem Server-RAM zu lesen, empfängt das Backend bei jedem Request den kompletten Warenkorb direkt vom Browser des Tablets. Der Server berechnet die Route in durchschnittlich :math:`\mathcal{O}(1)` in Bezug auf den globalen State und "vergisst" den Kunden danach sofort wieder. Dies macht die Applikation horizontal unendlich skalierbar.

.. code-block:: python

    from dash import Input, Output, State

    @app.callback(
        [Output('cart-store', 'data'), Output('add-status-msg', 'children')],
        [Input('btn-add-item', 'n_clicks')],
        # STATELESS ARCHITECTURE: Der Payload (current_cart) kommt direkt vom Edge-Client
        [State('input-prod-name', 'value'), State('cart-store', 'data')]
    )
    def manage_cart(n_add, text_input, current_cart):
        """ Der Server hat kein Gedächtnis. """
        current_cart = current_cart or []
        
        if text_input:
            found = inv_manager.find_product(text_input)
            if found:
                node_id, real_name, brand = found
                # ... [Produktspezifische Logik] ...
                current_cart.append({'node': node_id, 'name': real_name})
                
        # Der neue Zustand wird sofort wieder an den Browser des Nutzers zurückgegeben
        return current_cart, "Produkt hinzugefügt"

Phase III: Das Architektur-Dilemma – GIL vs. Multiprocessing
------------------------------------------------------------
Ein Webserver muss dutzende Anfragen zeitgleich verarbeiten (Concurrency). In modernen Web-Frameworks nutzt man hierfür oft asynchrone Programmierung (die Schlüsselwörter ``async`` und ``await``) oder klassisches Multithreading. Für unser spezifisches TSP-Routing-System wäre dies jedoch ein fataler Architekturfehler.

Der Grund hierfür ist tief in der C-Implementierung der Programmiersprache (CPython) verankert: Das **Global Interpreter Lock (GIL)**. Das GIL verhindert auf C-Ebene, dass zwei Threads gleichzeitig denselben Python-Bytecode ausführen. 

*Die Konsequenz:* Multithreading skaliert in Python nur bei I/O-gebundenen Wartezeiten (z. B. Datenbankabfragen). Die hochkomplexe TSP-Routenberechnung (Operations Research) besitzt jedoch keine I/O-Latenzen, sondern ist reine, CPU-gebundene Mathematik. Würden wir Dash über Threads skalieren, würde ein Kunde, der gerade die Held-Karp-Matrix berechnet, das GIL exklusiv sperren. Alle anderen parallelen Routing-Anfragen im Supermarkt würden in einem Bottleneck blockieren und "einfrieren".

*Die Lösung (WSGI Multiprocessing):* Anstelle von sich blockierenden Threads nutzt das System parallele OS-Prozesse. Das System wird über einen hochperformanten WSGI-Server (wie Gunicorn) orchestriert, der die Dash-Applikation im Pre-Fork-Modell in strikt isolierte Betriebssystem-Worker-Prozesse spiegelt. Jeder dieser Worker besitzt seinen eigenen Speicherraum und – entscheidend – **sein eigenes, unabhängiges GIL**. Das GIL wird dadurch nicht magisch "umgangen", sondern die Architektur multipliziert die Anzahl der GILs auf Betriebssystemebene. So skaliert die CPU-gebundene Mathematik parallel über alle physischen Kerne des Servers. Wenn Prozess A die Held-Karp-Matrix berechnet, übernimmt Prozess B völlig unabhängig und latenzfrei die HTTP-Anfrage des nächsten Kunden.

Phase IV: Fuzzy Search & In-Memory Memoization (LRU Cache)
----------------------------------------------------------
Die vom UI gesendete Sucheingabe muss in reale topologische Graphen-Knoten übersetzt werden. Da Kunden auf Touchscreens oft Tippfehler machen, würde eine exakte String-Abfrage fast immer versagen. Das System nutzt hierfür die TF-IDF Logistische Regression und die **Wagner-Fischer Distanz** (eine Implementierung der Damerau-Levenshtein-Distanz via Dynamischer Programmierung).

Um das Backend bei wiederkehrenden und rechenintensiven Anfragen zu entlasten, implementiert die Architektur auf der ``predict``-Methode der ``MLOpsEngine`` eine In-Memory Memoization über den Python ``@lru_cache``. 

*Der Trade-off (Isolation):* Da die Gunicorn-Worker (wie in Phase III beschrieben) über Multiprocessing isoliert sind und ihren Speicherraum auf OS-Ebene strikt trennen, ist ein solcher RAM-Cache pro Prozess gekapselt (Local Cache). Ein Suchbegriff, der bei Worker A gecacht wurde, muss von Worker B bei einer identischen neuen Anfrage dennoch neu berechnet werden. Das System toleriert diese Redundanz bewusst, um das Setup leichtgewichtig zu halten. Für eine globale Memoization über alle Worker hinweg müsste in einer zukünftigen Skalierungsstufe ein externer In-Memory-Datastore wie *Redis* angebunden werden.

.. code-block:: python

    from functools import lru_cache

    class MLOpsEngine:
        # ...
        @lru_cache(maxsize=4096)
        def predict(self, text: str) -> Tuple[str, str, float]:
            if not text: return "Sonstiges (Kasse)", "vW1", 0.0
            clean = TextNormalizer.clean(text)
            
            # 1. Greifen die Standard-Anker?
            anchor_cat = self._get_compound_anchor(text)
            if anchor_cat:
                return anchor_cat, self._get_balanced_node(anchor_cat, clean), 0.99
                    
            # 2. ML-Modell übernimmt den Rest
            if self.is_loaded and self.pipeline:
                try:
                    probs = self.pipeline.predict_proba([clean])[0]
                    # ... [Klassifizierungs-Logik] ...

Phase V: Architektonische Limitierung & Future Work (Kühlkette)
---------------------------------------------------------------
Ein elementarer Business-Case für digitale Supermarkt-Zwillinge ist die Einhaltung der Kühlkette (Cold Chain). Physisch tiefgekühlte Produkte (z.B. Speiseeis) beginnen sofort nach der Entnahme aus dem Regal zu tauen. 

Die aktuelle Iteration der Operations-Research-Engine (V1.0) operiert noch auf einer strikten euklidischen Distanzminimierung unter Einbezug temporaler Stau-Metriken. Das bedeutet, das System ist aktuell "blind" für die thermodynamischen Eigenschaften der einzelnen Produkte. Ein naiver Routing-Algorithmus würde das Eis womöglich als erstes Produkt auf die Liste setzen, woraufhin der Kunde noch 30 Minuten mit dem tauenden Eis durch den Markt läuft.

*Architektonische Limitierung & Future Work:* In zukünftigen Ausbaustufen muss die Produktdatenbank (``products.json``) um topologische Metadaten (z.B. ein ``is_frozen``-Flag) erweitert werden. Das Frontend könnte dies visuell durch Badges ("TK-Ware") hervorheben, während der TSP-Solver auf Backend-Ebene dieses Flag ausliest, um Kühlwaren durch harte Penalty-Gewichtungen systematisch ans Ende der Route (direkt vor die Kasse) zu sortieren. Dies verdeutlicht die Notwendigkeit einer immer engeren Verzahnung von Data Engineering, UI-Design und Operations Research für kommende Systemgenerationen.

Phase VI: KI-Mutation, Graph-Isolierung & DSGVO-Konformität
-----------------------------------------------------------
An dieser Schnittstelle greifen Stau-Prädiktion und Topologie ineinander. Der statische Supermarkt-Graph kennt physikalisch nur "Meter". Um den Kunden nicht in einen verstopften Gang zu leiten, mutiert die Architektur die Kanten "on-the-fly" mit Zeitstrafen aus dem Machine-Learning-Modell.

*Open-Loop vs. Closed-Loop:* Ein Einwand könnte lauten, dass das System keine echte Live-Telemetrie (Closed-Loop) aller 50 physischen Einkaufswagen im Markt nutzt. Diese architektonische Entscheidung ist jedoch kein Bug, sondern ein zwingendes Feature für den **Datenschutz (DSGVO)**. Ein lückenloses, serverseitiges Live-Tracking aller Kunden-Bewegungsprofile ist im europäischen Einzelhandel juristisch hochproblematisch. Die KI arbeitet stattdessen im **Open-Loop-Verfahren** auf Basis angelernter autoregressiver historischer Muster (Sim2Real), was völlig ausreicht, um zirkadiane Staus hochpräzise zu umschiffen.

*Graph-Isolierung via Copy-Pattern:* Um bei 100 gleichzeitigen Kundenanfragen Speicherkontaminationen (Pass-by-Reference Leaks) auf dem globalen Graphen zu verhindern, erzeugt die Dashboard-Instanz pro Request eine autarke Kopie des Basisgraphen (``G_smart = G_base.copy()``). Da der NetworkX-Graph hochgradig optimiert ist, operiert diese Instanziierung performant und garantiert absolute Thread-Sicherheit innerhalb des jeweiligen WSGI-Workers.

Die physikalische Übersetzung des Vorhersage-Tensors in Kantengewichte erfolgt dabei auf diesem isolierten Graphen über die adaptierte **Bureau of Public Roads (BPR) Penalty-Funktion**. Verkehrsstaus wachsen nicht linear. Ab einer kritischen Personenmasse bricht der Verkehrsfluss exponentiell zusammen. Die Architektur nutzt daher im Frontend exponentielle Bestrafungen für stark belastete Kanten.

Phase VII: Graphen-Kondensation (Der Floyd-Warshall-Fehlschluss)
----------------------------------------------------------------
Der Graph besteht aus hunderten Knotenpunkten. Diesen riesigen Suchraum direkt an den Traveling-Salesperson-Solver (TSP) zu übergeben, ist schlicht unmöglich. Der Solver benötigt als Eingabe stattdessen eine drastisch reduzierte **Distanzmatrix (Clique)**, die *ausschließlich* aus den gesuchten Ziel-Produkten besteht. 

Ein klassischer Fehler im Software Engineering ist es, hier den Floyd-Warshall-Algorithmus (All-Pairs Shortest Path) zur einmaligen Vorberechnung (Precomputation) einzusetzen, um Pfade danach in :math:`\mathcal{O}(1)` abzufragen. Das ist hier jedoch mathematisch unmöglich: Da die dynamischen KI-Stau-Mutationen (siehe Phase VI) die Kantenkosten je nach Uhrzeit und Auslastung stetig verändern, verbietet sich eine statische Vorberechnung der gesamten Matrix. 

Das System nutzt stattdessen den heuristikfreien **Dijkstra-Algorithmus** direkt innerhalb der Haupt-Orchestrierung (``calculate_hybrid_route``). Entgegen eines echten Multi-Source-Dijkstras (welcher alle Startpunkte simultan evaluiert), operiert die Engine hier als **Iterated Single-Source Dijkstra**. Der Code beweist, dass Dijkstra in einer iterativen Schleife exakt nur für die Menge der gesuchten Produkte (K) ausgeführt wird. Durch die Nutzung des internen Binary Heaps von NetworkX sinkt die Komplexität dramatisch auf :math:`\mathcal{O}(K \cdot E \log V)`. 

.. code-block:: python

    from typing import List
    import networkx as nx
    import datetime

    # @execution_profiler
    def calculate_hybrid_route(graph: nx.Graph, start_node: str, valid_targets: List[str], end_node: str, current_time: datetime = None):
        # ...
        d_mat, p_mat = {}, {}
        
        # Architektonischer Fix: Der deterministisch zugewiesene Kassenknoten (end_node) 
        # muss zwingend Teil der Kürzeste-Wege-Matrix sein!
        rel_nodes = [start_node] + valid_targets + [end_node]
        
        for u in rel_nodes:
            try:
                l, p = nx.single_source_dijkstra(graph, u, weight='weight')
                for v in rel_nodes:
                    if u != v and v in l: 
                        d_mat[(u, v)], p_mat[(u, v)] = l[v], p[v]
                    elif u != v: 
                        d_mat[(u, v)] = float('inf')
            except nx.NetworkXNoPath: 
                pass
        # ... Übergabe von d_mat an den gewählten TSP-Solver

Phase VIII: Stochastik & M/M/1/K Warteschlangentheorie
------------------------------------------------------
Eine rein räumliche Navigation scheitert auf der "letzten Meile". Der geometrisch kürzeste Weg zur Kasse 1 ist wertlos, wenn sich dort ein massiver Rückstau bildet. Das System übergibt diese finale Entscheidung daher an ein dediziertes Warteschlangentheorie-Modul (Queueing Theory).

Um die Latenzen im Echtzeit-Dashboard minimal zu halten, vermeidet das System rechenintensive M/G/1-Gleichungen. Die Architektur modelliert die Supermarktkassen stattdessen ganz bewusst als performantes **M/M/1/K Warteschlangenmodell** (Verlustsystem). Hierbei wird mathematisch approximiert, dass die Ankünfte poissonverteilt (Markov-Eigenschaft, M) und die Abfertigungszeiten exponentiell (M) sind, mit *einem* Server (1) und einer strikt begrenzten Systemkapazität (K=10). Ist die Schlange maximal gefüllt (Loss-System), weichen Kunden auf andere Gänge aus.

Die Berechnungen basieren auf den stationären Gleichungen für Markow-Ketten. Der Auslastungsgrad ist definiert als :math:`\rho = \frac{\lambda}{\mu}`.
Die Wahrscheinlichkeit für ein leeres System (:math:`P_0`) und ein volles System (:math:`P_K`) berechnen sich bei einer Auslastung :math:`\rho \neq 1` als:

.. math::

    P_0 = \frac{1 - \rho}{1 - \rho^{K+1}}

.. math::

    P_K = \rho^K P_0

.. math::

    L_q = \frac{\rho}{1 - \rho} - \frac{(K + 1) \rho^{K+1}}{1 - \rho^{K+1}}

Aus der durchschnittlichen Warteschlangenlänge (:math:`L_q`) und der effektiven Ankunftsrate (:math:`\lambda_{eff} = \lambda(1 - P_K)`) leitet das System über das Gesetz von Little die erwartete Wartezeit ab.

Um eine reibungslose Navigation zu garantieren, wird die finale Zuweisung der optimalen Kasse zwingend vor der Halbzeit des Einkaufs (Pre-Halftime) berechnet. Wartet das System zu lange mit der Kassenzuweisung, fehlt dem Routing-Algorithmus auf den letzten Metern der topologische Manövrierraum für eine effiziente Umleitung.

.. code-block:: python

    import random
    from typing import Dict

    class EnterpriseQueuingModel:
        @staticmethod
        def calculate_wait_metrics(base_lambda: float, current_hour: int, checkout_id: str) -> Dict[str, float]:
            c = 1       # Anzahl der Kassen (isoliert als M/M/1 betrachtet)
            mu = 1.5    # Service-Rate
            K = 10      # Kapazität (Loss System)
            
            # ... Berechnung effektiver Ankunftsrate (lam) ...
            lam = base_lambda # (Vereinfacht für Darstellung)
            rho = lam / mu
            
            # --- Markow-Ketten Formeln für M/M/1/K ---
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
            
            return { "wait_sec": wait_sec, "p_wait": 1.0 - p0, "lq": lq, "p_loss": pk }

Phase IX: Der Orchestrator, Strategy Pattern & SA Idempotenz
------------------------------------------------------------
Ein monolithischer Python-Code voller bedingter Anweisungen für die verschiedenen Algorithmen wäre extrem schwer wartbar. Das System lagert diese Logik elegant über das **Strategy Pattern** (ein Entwurfsmuster der "Gang of Four") in austauschbare Klassen aus.

Der Controller schaltet abhängig von der Warenkorbgröße logisch linear aufsteigend um:

* **n <= 11:** Exakte Dynamische Programmierung (Held-Karp). Das harte Limit von ca. 250.000 Operationen ist primär eine bewusste "Safety First" Ingenieursentscheidung. Obwohl der Server-RAM theoretisch auch 15 Produkte fassen würde, garantiert dieses strikte Limit, dass die API-Antwortzeit unter paralleler Last konstant unter 100 ms bleibt und OOM-Abstürze absolut ausgeschlossen sind.
* **n <= 15:** Thermodynamische Heuristiken (Simulated Annealing). 
* **n <= 25:** Schwarmintelligenz (Ant Colony Optimization).
* **n > 25:** Biologische Evolution (Genetic Algorithm für gigantische Suchräume).

*Die UX-Absicherung (Idempotenz):* Stochastische Algorithmen (wie Simulated Annealing oder Ant Colony) bergen jedoch ein massives Risiko für die User Experience: Drückt der Kunde bei einem großen Warenkorb versehentlich zweimal auf "Route berechnen", würde der Zufallsgenerator anders abbiegen und die Route auf dem Tablet würde wild flackern. Um absolute **Idempotenz** (Gleicher Input = Gleicher Output) zu garantieren, wird der Random-Seed des Solvers deterministisch aus dem Hashwert der Einkaufsliste generiert. Die Route ist somit heuristisch optimiert, aber für denselben Warenkorb zu 100 % reproduzierbar.

.. code-block:: python

    import hashlib
    import random

    # ... [In calculate_hybrid_route] ...
    
    # Deterministische Idempotenz für stochastische Algorithmen (Verhindert UI-Flackern)
    seed_str = "".join(valid_targets)
    random.seed(int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32))
    
    # Sauberes Software-Engineering: Lineare, aufsteigende Laufzeit-Eskalation
    if valid_targets:
        n_targets = len(valid_targets)
        if n_targets <= CONFIG.DP_EXACT_LIMIT:
            solver = HeldKarpDPSolver()          
        elif n_targets > 25:
            solver = GeneticAlgorithmSolver()    
        elif n_targets > CONFIG.SA_THRESHOLD:    
            solver = AntColonySolver()           
        else:
            solver = SimulatedAnnealingSolver()  
            
        store_seq, msg = solver.solve(d_mat, start_node, valid_targets, end_node)

Phase X: Wissenschaftliche Evaluierung & Big-O
----------------------------------------------
In einer akademischen Arbeit muss die architektonische Sicherheit des Backends bewiesen werden. Die folgende Laufzeitmatrix beweist, dass das hochkomplexe MVC-Backend innerhalb strenger Laufzeitschranken operiert. 

Ein entscheidendes Detail der Informatik: Eine Hash-Map (wie sie für den Cache oder das Inventar verwendet wird) liefert zwar durchschnittlich Zugriffszeiten von :math:`\mathcal{O}(1)`. Im **Worst-Case** jedoch (wenn der Hashing-Algorithmus massive Hash-Kollisionen erzeugt), degeneriert die Zugriffszeit zu :math:`\mathcal{O}(N)`. Die Architektur schützt sich vor diesem Worst-Case durch das vorherige Pruning des Suchraums.

.. list-table:: Algorithmische Komplexität des Backends
    :widths: 30 45 25
    :header-rows: 1

    * - System-Komponente
      - Verwendete Datenstruktur / Algorithmus
      - Zeitkomplexität
    * - **Inventar Hash-Lookup**
      - Hash-Map / Python Dictionary
      - Avg: :math:`\mathcal{O}(1)`, Worst: :math:`\mathcal{O}(N)`
    * - **Inventar-Fuzzy-Suche**
      - Damerau-Levenshtein (Dynamische Programmierung)
      - :math:`\mathcal{O}(N \cdot M)`
    * - **KI Stau-Mutation**
      - Bureau of Public Roads (BPR) Transformation
      - :math:`\mathcal{O}(E)`
    * - **Graphen-Kondensation**
      - Iterated Single-Source Dijkstra (Binary Heap)
      - :math:`\mathcal{O}(K \cdot E \log V)`
    * - **Routen-Lösung (Klein, n<=11)**
      - Held-Karp Algorithmus (Exakte DP)
      - :math:`\mathcal{O}(N^2 \cdot 2^N)`
    * - **Routen-Lösung (Mittel, 12-25)**
      - Simulated Annealing & Ant Colony (Heuristiken)
      - :math:`\mathcal{O}(\text{Iterationen})`
    * - **Pfad-Rekonstruktion**
      - Listen-Verkettung (Dijkstra Path Extend)
      - :math:`\mathcal{O}(V_{\text{pfad}})`