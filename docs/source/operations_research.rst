Operations Research & Routing-Strategien
========================================

Das mathematische Herzstück der Navigation im Smart Supermarket ist die Lösung des Traveling Salesperson Problems (TSP) – des Problems der Handlungsreisenden. Die Aufgabe klingt für den Endanwender intuitiv simpel: Finde den zeitlich kürzesten Weg, um alle Produkte auf dem Einkaufszettel genau einmal zu besuchen und am Ende an der Kasse zu bezahlen. 

In der Komplexitätstheorie der Informatik wird dieses Problem jedoch als NP-schwer (nondeterministic polynomial-time hard) eingestuft. Das bedeutet: Es existiert weltweit kein deterministischer Algorithmus, der dieses Problem für beliebig große Einkaufslisten in akzeptabler (polynomieller) Zeit perfekt lösen kann. 

Die mathematische Realität der kombinatorischen Explosion: Bei einer kleinen Einkaufsliste von 10 Produkten gibt es bereits über 3,6 Millionen mögliche Wege ($10!$). Ein moderner Backend-Server berechnet dies in Millisekunden. Stehen jedoch 25 Produkte auf der Liste, wächst der Suchraum auf über $1.5 \times 10^{25}$ mögliche Routen an – eine kombinatorische Explosion, die die Brute-Force-Rechenkapazität jedes modernen Supercomputers übersteigt. Der naive Versuch, dies auf einem Server exakt zu berechnen, würde im Live-Betrieb in einem sofortigen Out-of-Memory-Error (OOM) resultieren.

Die Backend-Architektur des JMU Smart Cart Systems erzwingt daher keinen starren "One-Size-Fits-All"-Ansatz. Stattdessen implementiert sie ein intelligentes, dynamisch skalierendes Framework über das **Strategy Pattern**. Abhängig von der Größe des Warenkorbs schaltet das System autonom zwischen mathematischer Exaktheit, Thermodynamik, Schwarmintelligenz und biologischen Evolutionsverfahren um.

1. Architektonische Disruption: Die Demontage der akademischen Baseline
-----------------------------------------------------------------------
In der klassischen Fachliteratur wird das Routing oft durch den A*-Algorithmus und die Christofides-Approximation gelehrt. Im Kontext einer performanten Live-Engine mit dynamischen Verkehrsstaus wurde diese architektonische Entscheidung mathematisch rigoros revidiert.

1.1 Warum A* ausscheidet: Heuristische Degeneration & Point-to-Point Limitierung
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Der A*-Algorithmus ist primär für Point-to-Point Navigation (Ein Start, ein Ziel) konzipiert. Um eine dichte $\mathcal{O}(K \times K)$ Distanzmatrix für $K$ Zielprodukte aufzubauen, müsste der A*-Algorithmus $K \cdot (K-1)$ mal initialisiert werden. 

Zudem garantiert A* nur dann Effizienzvorteile, wenn seine Heuristik (z. B. die reine Laufzeit ohne Stau) nah an den realen Kosten liegt. Durch die Integration des prädiktiven Traffic-Modells (BPR-Zeitstrafen) entsteht eine gewaltige Diskrepanz: Die Heuristik bleibt zwar mathematisch zulässig (da sie die wahren, staugeplagten Kosten stark unterschätzt), sie wird jedoch extrem "schwach". Eine zu optimistische Heuristik zwingt A* dazu, seinen gerichteten Such-Fokus zu verlieren und fast den gesamten Graphen zu expandieren. Der Algorithmus degeneriert de facto zu einem regulären Dijkstra, trägt dabei aber den unnötigen Rechen-Overhead der kontinuierlichen Heuristik-Evaluierung mit sich.

1.2 Unsere gewählte Baseline: Der Repeated Single-Source Dijkstra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Das System nutzt stattdessen den heuristikfreien Dijkstra-Algorithmus. Im Gegensatz zu A*, ist Dijkstra ein Single-Source-All-Destinations-Verfahren. Er breitet sich im Graphen wie eine Wasserwelle aus und berechnet in einem einzigen Durchlauf die exakten Zeitkosten von einem Startknoten zu *allen* relevanten Zielknoten simultan. 

Anstatt $K \cdot (K-1)$ Anfragen zu feuern, wird der Algorithmus iterativ nur exakt $K$-mal aufgerufen (Repeated Single-Source Dijkstra) und berücksichtigt dabei strikt die dynamischen BPR-Strafen. Die Zeitkomplexität für den Aufbau der Matrix sinkt dadurch drastisch auf $\mathcal{O}(K \cdot (|V| \log |V| + |E|))$. 

Um die theoretischen Performance-Nachteile von Dijkstra (die ungerichtete Expansion) zu kompensieren, operiert der Algorithmus auf einem adjazenz-optimierten Teilgraphen. Da der Supermarkt-Graph stark dünnbesetzt ist (low degree), konvergiert Dijkstra auch ohne Heuristik in unter 10 Millisekunden. Der Algorithmus generiert dabei den sogenannten *metrischen Abschluss* (Metric Closure): Er überbrückt die physischen Regalreihen und erzeugt eine vollständig verbundene Clique, in der jede Kante den exakten, stau-gewichteten kürzesten Pfad zwischen zwei Zielprodukten repräsentiert.

.. code-block:: python

    import networkx as nx

    def build_condensed_matrix(G_smart: nx.Graph, start_node: str, shopping_nodes: list, end_node: str):
        """
        Kondensiert den 500-Knoten Graphen in eine kompakte K-x-K Distanzmatrix (Clique).
        Nutzt iterativen Dijkstra zur Ermittlung des Metrischen Abschlusses.
        """
        d_mat, p_mat = {}, {}
        
        # Die Matrix muss zwingend den Start, alle Ziele und die finale Kasse enthalten
        rel_nodes = [start_node] + shopping_nodes + [end_node]

        for u in rel_nodes:
            try:
                # Dijkstra expandiert heuristikfrei und berücksichtigt BPR-Strafen (weight)
                # Laufzeit: O(V log V + E) pro iteriertem Zielknoten
                lengths, paths = nx.single_source_dijkstra(G_smart, u, weight='weight')
                
                for v in rel_nodes:
                    if u != v and v in lengths:
                        d_mat[(u, v)] = lengths[v]
                        p_mat[(u, v)] = paths[v]
            except nx.NetworkXNoPath:
                pass # Graceful Degradation für physisch blockierte Regal-Inseln
                
        return d_mat, p_mat

1.3 Das Umweg-Paradoxon (Temporale Arbitrage)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Eine kritische Systemfrage lautet: *Leitet die KI den Kunden auf einen 50-Meter-Umweg, nur um einem winzigen Stau auszuweichen, der den Kunden eigentlich nur 5 Sekunden Wartezeit gekostet hätte? Wäre die KI dann nicht schlechter als eine "dumme" statische Baseline?*

Die Antwort liegt in der physikalischen Einheit des Basis-Graphen. Da die Kantengewichte der Topologie nicht in Metern, sondern von Beginn an in **Sekunden (Transit Time)** kodiert sind und die BPR-Strafen der KI ebenfalls **Sekunden** ausgeben, führt der Dijkstra-Algorithmus eine exakte **Temporale Arbitrage (Kosten-Nutzen-Analyse)** durch. 

Der Algorithmus minimiert das globale Integral der Zeit. Ein 50-Meter-Umweg kostet bei normaler Schrittgeschwindigkeit ca. 35 Sekunden ($t_{umweg} = 35$). Ein direkter Gang kostet 10 Sekunden plus 5 Sekunden Stau ($t_{direkt} = 15$). Der Dijkstra-Algorithmus vergleicht 15 < 35 und leitet den Kunden deterministisch **durch** den Stau. Die KI weicht einem Hindernis also niemals blind aus, sondern exakt nur dann, wenn die Zeitstrafe des Staus mathematisch größer ist als die Zeitstrafe des physischen Umwegs.

1.4 Warum Christofides scheitert: Die Dreiecksungleichung
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Der Christofides-Algorithmus ist das berühmteste Approximationsverfahren für das TSP. Er erfordert für seinen mathematischen Beweis jedoch zwingend einen metrischen Raum und die Einhaltung der Dreiecksungleichung. In unserem Modell ist dieser metrische Raum durch die BPR-Penalty-Funktion zerstört: Ein extremer Stau macht den direkten Weg von A nach C oft teurer als die Summe der Umwege über B. Christofides würde hier physikalisch unmögliche "Shortcuts" (z.B. durch Regale) generieren, da der Algorithmus die Zeitstrafen geometrisch nicht korrekt abbilden kann.

2. Paradigmenwechsel: Das Asymmetric TSP (ATSP)
-----------------------------------------------
Ein häufiger und fataler Modellierungsfehler in der Literatur ist die Klassifizierung von Supermarkt-Routen als symmetrisches Problem (Symmetric TSP). Ein symmetrisches Problem impliziert, dass die Distanz (oder Zeit) von Knoten A nach B identisch mit dem Rückweg von B nach A ist: $d(A,B) = d(B,A)$.

Diese Annahme ist für das JMU Smart Cart System physikalisch und mathematisch falsch:

* **Topologische Einbahnstraßen:** Wie in der Architektur definiert, besitzen Kassenzonen und Eingangsschranken ausschließlich gerichtete Vorwärtskanten. Ein Zurücknavigieren ist topologisch ausgeschlossen ($d(\text{Kasse}, \text{Regal}) = \infty$).
* **Dynamische KI-Zeitstrafen:** Das XGBoost-Modell verhängt Stau-Strafen gerichtet. Wenn sich ein Stau am Ende von Gang 3 bildet, kostet der Weg in den Gang hinein massive Zeitstrafen. Der Weg aus dem Gang heraus (gegen den Stau) ist jedoch völlig frei. 

Infolgedessen gilt im System zwingend $d(A,B) \neq d(B,A)$. Die Engine modelliert die Wegfindung daher strikt als **Asymmetric Traveling Salesperson Problem (ATSP)**. Der resultierende Lösungsraum ist mathematisch nochmals deutlich restriktiver und komplexer als beim regulären TSP.

Anstatt das offene Problem durch das künstliche Hinzufügen von Dummy-Nodes aufzublähen, erzwingt die Architektur eine strikte **Separation of Concerns**. Der ATSP-Solver optimiert ausschließlich die Raum-Geometrie der Regale. Die Auswahl der optimalen Kasse wird bewusst vom Routing entkoppelt und erst in der System-Synthese durch das Stochastik-Modul evaluiert.

3. Das Strategy-Pattern: Dynamische Algorithmen-Eskalation
----------------------------------------------------------
Um die Laufzeitschranken des Servers zu schützen, lagert die Hauptfunktion ``calculate_hybrid_route`` die Logik in das abstrakte Interface ``RoutingStrategy`` aus. Das System implementiert eine vierstufige Eskalationsstrategie:

* **K <= 11: Held-Karp (Exakte DP).** Garantiert das absolute globale Zeit-Optimum.
* **12 <= K <= 15: Simulated Annealing.** Thermodynamische Suche zur Überwindung lokaler Minima. Der hierbei implementierte 2-Opt-Swap-Mechanismus validiert nach jedem Kantentausch die neue Pfadrichtung strikt gegen die Adjazenzmatrix. Dies garantiert Asymmetrie-Bewusstsein (ATSP), sodass keine verbotenen Einbahnstraßen-Regelungen im Supermarkt verletzt werden.
* **16 <= K <= 25: Ant Colony Optimization (ACO).** Schwarmbasierte Pfadfindung.
* **K > 25: Genetischer Algorithmus.** Biologische Evolution für gigantische Suchräume.

.. code-block:: python

    # Auszug aus dem Controller (calculate_hybrid_route):
    # Dynamische Zuweisung des Solvers zur Vermeidung von OOM-Abstürzen
    if shopping_nodes:
        n_targets = len(shopping_nodes)
        if n_targets <= CONFIG.DP_EXACT_LIMIT:
            solver = HeldKarpDPSolver()          
        elif n_targets > 25:
            solver = GeneticAlgorithmSolver()    
        elif n_targets > CONFIG.SA_THRESHOLD: 
            solver = AntColonySolver()           
        else:
            solver = SimulatedAnnealingSolver()  
            
        # Polymorpher Aufruf über das gemeinsame Interface
        store_seq, msg = solver.solve(d_mat, start_node, shopping_nodes, optimal_exit_node)

4. Exakte Dynamische Programmierung: Der Pythonic Way
-----------------------------------------------------
Für kleine Warenkörbe ($K \le 11$) verlangt das System absolute Exaktheit. Die Klasse ``HeldKarpDPSolver`` reduziert die Brute-Force-Laufzeit $\mathcal{O}(K!)$ durch Dynamische Programmierung auf $\mathcal{O}(K^2 \cdot 2^K)$. 

Das harte Limit von $K \le 11$ ist primär durch die exponentielle Raumkomplexität von $\mathcal{O}(K \cdot 2^K)$ motiviert. Über diesen Schwellenwert hinaus würde der Memory-Footprint der zu speichernden DP-Tabelle (insbesondere bei 64-Bit Floats) die zustandslosen Backend-Worker unter paralleler Last instabil machen.

*Der Architektonische Kniff (Frozensets statt Bitmasking):* In Hardware-nahen Sprachen (C++) wird Held-Karp über 32-Bit-Integer und Bit-Shifting gelöst. In einer High-Level-Sprache wie Python führt manuelles Bit-Shifting jedoch zu Performance-Verlusten. Die Architektur nutzt daher den "Pythonic Way": **Frozensets**. Ein ``frozenset`` ist eine unveränderliche (immutable) Menge und damit im Gegensatz zu normalen Listen hashbar. 

Dies erlaubt es dem Algorithmus, die bereits besuchten Knoten (``unvisited: frozenset``) direkt als Schlüssel in einem Dictionary (der Memoization-Tabelle) abzulegen. Der Python-Interpreter löst dies intern über hochoptimierte C-Hashmaps in $\mathcal{O}(1)$, was den Server-RAM massiv entlastet.

.. code-block:: python

    class HeldKarpDPSolver(RoutingStrategy):
        def solve(self, dist_matrix: dict, start: str, targets: List[str], end: Optional[str]):
            memo = {} # Das Cache-Dictionary
            
            # Die innere Rekursionsfunktion
            def dp(curr: str, unvisited: frozenset) -> Tuple[float, List[str]]:
                if not unvisited:
                    # Basisfall: Alle Produkte gefunden. Addiere die Distanz zur finalen Kasse.
                    cost = dist_matrix.get((curr, end), 0.0) if end else 0.0
                    return cost, [end] if end else []
                
                # Der Zustand (State) wird aus dem aktuellen Knoten und dem Frozenset gebildet.
                # Da Frozensets immutable sind, dienen sie als perfekten Hash-Key.
                state = (curr, unvisited)
                
                # O(1) Lookup: Wurde dieser Sub-Graph bereits berechnet?
                if state in memo: 
                    return memo[state]
                
                # ... [Berechnung des Minimums über alle unbesuchten Nachbarn] ...
                
                memo[state] = (min_cost, best_path)
                return memo[state]
                
            # Initialer Aufruf mit einem frozenset aller Zielprodukte
            _, path = dp(start, frozenset(targets))
            return [start] + path, "Held-Karp Exakt-Optimum"

5. Schwarmintelligenz: Ant Colony Optimization (ACO)
----------------------------------------------------
Für große Warenkörbe (bis 25 Artikel) nutzt die Architektur die kollektive Intelligenz von Ameisenvölkern (Ant Colony Optimization). 

Virtuelle Ameisen schwärmen parallel über den ATSP-Graphen aus. Die Entscheidung einer Ameise $k$, vom aktuellen Produkt $i$ zum nächsten Produkt $j$ zu navigieren, basiert nicht auf reinem Zufall, sondern auf der hochkomplexen stochastischen Übergangswahrscheinlichkeit $p_{ij}^k$:

$$p_{ij}^k = \frac{[\tau_{ij}]^\alpha [\eta_{ij}]^\beta}{\sum_{l \in N_i^k} [\tau_{il}]^\alpha [\eta_{il}]^\beta}$$

Wobei $\tau_{ij}$ die auf der Kante liegende Pheromonspur (Erfahrungswert früherer Ameisen) und $\eta_{ij}$ die heuristische Sichtbarkeit (der Kehrwert der Entfernung, $\frac{1}{d_{ij}}$) darstellt. $N_i^k$ ist die Menge der noch unbesuchten Knoten.

Nach jeder Epoche verdunstet ein Teil der Pheromone (Evaporation), um Konvergenz in suboptimalen Routen zu verhindern, während Ameisen mit besonders kurzen Gesamtwegen massiv neue Pheromone ausschütten. Die Ameisen "lernen" kollektiv den perfekten Weg durch das asymmetrische Labyrinth.

.. code-block:: python

    # Auszug aus der AntColonySolver Klasse:
    # prob berechnet sich exakt nach der ACO-Wahrscheinlichkeitsformel
    prob = pheromones.get((curr, v), 1.0) ** CONFIG.ACO_ALPHA * ((1.0 / dist) ** CONFIG.ACO_BETA)
    probs.append((v, prob))
    
    tot_p = sum(pr for _, pr in probs)
    if tot_p > 0:
        # Roulette-Wheel-Selection: Stochastische Auswahl basierend auf den Gewichten.
        # Verhindert, dass Ameisen deterministisch immer den gleichen Weg laufen, 
        # und fördert die emergente Exploration des Suchraums.
        nxt = random.choices([v for v, _ in probs], weights=[pr/tot_p for _, pr in probs])[0] 

6. Biologische Evolution: Genetic Algorithm (GA) & PMX
------------------------------------------------------
Für gigantische Großeinkäufe (über 25 Artikel) oder Inventur-Routen der Mitarbeiter bricht selbst die Schwarmintelligenz unter der Flut der Permutationen ein. Das System eskaliert in die höchste Instanz: Evolutionäre Algorithmen.

Eine Population aus Hunderten potenzieller Routen wird generiert. Über Fitness-Funktionen (Survival of the Fittest) werden die besten Routen selektiert und rekombiniert. 

*Das Crossover-Paradoxon:* Man kann zwei Supermarkt-Routen nicht einfach in der Mitte zerschneiden und neu zusammensetzen (klassisches 1-Point-Crossover), da sonst Artikel doppelt auf der Liste auftauchen und andere komplett verschwinden würden. Die Engine löst dieses Problem über das komplexe **Partially Mapped Crossover (PMX)**. Ein Sub-Array der Eltern-DNA wird kopiert, fehlende Produkte werden durch eine zyklische Mapping-Tabelle iterativ aufgefüllt, um Duplikate mathematisch absolut auszuschließen.

.. code-block:: python

    # Auszug aus der GeneticAlgorithmSolver Klasse:
    def _partially_mapped_crossover(self, parent1: List[str], parent2: List[str]) -> List[str]:
        """
        Spezielle Crossover-Technik für das TSP-Problem.
        Ein normales Durchschneiden von Arrays würde dazu führen, dass Regale doppelt 
        besucht werden oder fehlen. PMX repariert diese Duplikate intelligent.
        """
        size = len(parent1)
        child = [None] * size
        
        # 1. Tausche einen zufälligen Gen-Block zwischen den Elternteilen aus
        c1, c2 = sorted(random.sample(range(size), 2))
        child[c1:c2] = parent1[c1:c2]
        
        # 2. Fülle den Rest mit Elementen von Elternteil 2 auf (Vermeidung von Duplikaten)
        for i in range(c1, c2):
            if parent2[i] not in child:
                pos = i
                # Mapping-Schleife: Sucht den korrekten legalen Platz im Array
                while c1 <= pos < c2:
                    pos = parent2.index(parent1[pos])
                child[pos] = parent2[i]
                
        # 3. Restliche leere Stellen direkt übernehmen
        for i in range(size):
            if child[i] is None:
                child[i] = parent2[i]
                
        return child

7. System-Synthese: Separation of Concerns & M/M/1/K-Integration
----------------------------------------------------------------
Ein fataler Architekturfehler klassischer Routing-Systeme ist die Vermischung von Zuständigkeiten. Der TSP-Solver darf niemals die Komplexität der dynamischen Kassenauslastung oder der Wartezeit-Stochastik berechnen. 

Die Architektur erzwingt daher eine strikte **Separation of Concerns**. Um topologische Deadlocks im Endspurt an der Kasse sicher zu unterbinden, ruft der Orchestrator zwingend noch **vor der Halbzeit des Einkaufs** die ``CheckoutOptimizationFacade`` aus dem Stochastik-Modul auf. Diese Fassade nutzt das **M/M/1/K-Warteschlangenmodell**, um den exakten, optimalen Zielknoten auf Basis der Echtzeit-IoT-Daten stochastisch zu prädizieren.

Der ATSP-Solver erhält diesen finalisierten Kassenknoten vom Orchestrator lediglich als injizierten, deterministischen Parameter übergeben. Dadurch bleibt das Metaheuristik-Modul völlig blind (und damit hochperformant) gegenüber der schwankenden Komplexität der Kassenzonen und kann sich ausschließlich auf die Lösung des Graphen-Labyrinths konzentrieren.

8. Der Kybernetische Fallback-Schutz (Shadow-Routing)
-----------------------------------------------------
Das finale Meisterstück der Architektur ist die Absicherung des Systems gegen KI-Halluzinationen. Um das System absolut "Bulletproof" zu machen, berechnet der Orchestrator die finale Route im Hintergrund zwingend parallel im **Shadow-Mode**. Das System generiert eine Route auf dem normalen, staufreien Graphen (deterministische Baseline) und zeitgleich eine Route auf dem durch die KI mutierten Graphen. 

Anschließend unterzieht der Code beide Routen einem strikten Integritäts-Check auf dem nackten Basis-Graphen: Wenn die von der KI vorgeschlagene Ausweichroute in der *reinen physischen Laufzeit* (ohne Stau-Strafen) einen vordefinierten Toleranz-Schwellenwert im Vergleich zur Baseline überschreitet (die KI den Kunden also auf einen völlig absurden Umweg schicken will), legt die System-Synthese ein Veto ein. Das Backend verwirft die KI-Route und serviert dem Tablet als Fail-Safe die Baseline-Route. Dies beweist mathematisch, dass das KI-Routing das Nutzererlebnis in Edge-Cases niemals schlechter machen kann als ein klassisches Navigationssystem.

.. code-block:: python

    # Auszug aus update_visuals (app.py):
    # tb = Zeit der Baseline-Route (Dummer Agent)
    # tm = Zeit der KI-Route (Smarter Agent)
    
    # SANITY CHECK: Der kybernetische Fallback-Schutz
    # Ist die KI in der harten Simulation (tm) langsamer als die sture Baseline (tb)?
    if tm > tb or (px_b == px_m and py_b == py_m):
        
        # Veto der System-Synthese: Verwirft die KI-Halluzination und 
        # zwingt den Agenten deterministisch auf die Baseline zurück.
        walk_m, queue_m = walk_b, queue_b
        tm = tb
        seq_m = seq_b
        px_m, py_m = px_b, py_b
        optimized_order = seq_b