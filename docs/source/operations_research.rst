Operations Research & Routing-Strategien
========================================

Das mathematische Herzstück der Navigation im Smart Supermarket ist die Lösung des Traveling Salesperson Problems (TSP) – des Problems der Handlungsreisenden. Die Aufgabe klingt für den Endanwender intuitiv simpel: Finde den zeitlich kürzesten Weg, um alle Produkte auf dem Einkaufszettel genau einmal zu besuchen und am Ende an der Kasse zu bezahlen. 

In der Komplexitätstheorie der Informatik wird dieses Problem jedoch als NP-schwer (nondeterministic polynomial-time hard) eingestuft. Das bedeutet: Es existiert weltweit kein deterministischer Algorithmus, der dieses Problem für beliebig große Einkaufslisten in akzeptabler (polynomieller) Zeit perfekt lösen kann. 

Die mathematische Realität der kombinatorischen Explosion: Bei einer kleinen Einkaufsliste von 10 Produkten gibt es bereits über 3,6 Millionen mögliche Wege (10 Fakultät). Ein moderner Backend-Server berechnet dies in Millisekunden. Stehen jedoch 40 Produkte auf der Liste, existieren mehr mögliche Routen (40 Fakultät), als es Atome im sichtbaren Universum gibt. Der naive Versuch, dies auf einem Server exakt zu berechnen, würde in einem sofortigen Out-of-Memory-Error (OOM) resultieren.

Die Backend-Architektur des JMU Smart Cart Systems erzwingt daher keinen starren "One-Size-Fits-All"-Ansatz. Stattdessen implementiert sie ein intelligentes, dynamisch skalierendes Framework über das **Strategy Pattern**. Abhängig von der Größe des Warenkorbs schaltet das System autonom zwischen mathematischer Exaktheit, Thermodynamik, Schwarmintelligenz und biologischen Evolutionsverfahren um.

1. Architektonische Disruption: Die Demontage der akademischen Baseline
-----------------------------------------------------------------------
In der klassischen Fachliteratur wird das Routing oft durch den A*-Algorithmus und die Christofides-Approximation gelehrt. Im Kontext einer performanten Live-Engine mit dynamischen Verkehrsstaus wurde diese architektonische Entscheidung mathematisch rigoros revidiert.

1.1 Warum A* scheitert: Die unzulässige Heuristik
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A* garantiert nur dann den kürzesten Weg, wenn seine Heuristik (z.B. die Luftlinie) zulässig (admissible) ist, also die wahren Kosten niemals überschätzt. Durch die Integration unseres Prädiktiven Traffic-Modells, welches Staus als Bureau of Public Roads (BPR) Zeitstrafen auf die Gänge addiert, wird diese Regel gebrochen. Ein geometrisch kurzer Weg kann durch einen Stau plötzlich massiv teurer sein als ein physikalischer Umweg. Die klassische räumliche Heuristik würde A* in die Irre führen.

1.2 Unsere gewählte Baseline: Der Multi-Source Dijkstra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Das System nutzt stattdessen den heuristikfreien Dijkstra-Algorithmus. Dijkstra breitet sich im Graphen wie eine Wasserwelle aus und findet die exakten Zeitkosten unter strikter Berücksichtigung der dynamischen BPR-Strafen. Für eine Zielmenge K (die gesuchten Produkte) starten wir Dijkstra iterativ genau K-Mal. Die Komplexität sinkt dramatisch auf $O(K \cdot (V \log V + E))$. Da K extrem viel kleiner als V (Gesamtknoten) ist, ist dieser Ansatz hochperformant.

Der folgende Code-Beweis zeigt, wie die Architektur den riesigen Supermarkt-Graphen für den TSP-Solver in eine winzige, dichte Distanzmatrix (Clique) kondensiert:

.. code-block:: python

   import networkx as nx

   def build_condensed_matrix(G_smart: nx.Graph, start_node: str, valid_targets: list):
       """
       Kondensiert den 500-Knoten Graphen in eine kompakte K-x-K Distanzmatrix.
       Nutzt Multi-Source Dijkstra, um exakte Zeitkosten (inkl. KI-Staus) zu finden.
       """
       d_mat, p_mat = {}, {}
       rel_nodes = [start_node] + valid_targets

       for u in rel_nodes:
           try:
               # Dijkstra expandiert heuristikfrei und berücksichtigt BPR-Strafen (weight)
               # Laufzeit: O(V log V + E) pro Produkt in der Liste
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

Der Algorithmus minimiert das globale Integral der Zeit. Ein 50-Meter-Umweg kostet bei normaler Schrittgeschwindigkeit ca. 35 Sekunden (t_umweg = 35s). Ein direkter Gang kostet 10 Sekunden plus 5 Sekunden Stau (t_direkt = 15s). Der Dijkstra-Algorithmus vergleicht 15s < 35s und leitet den Kunden deterministisch **durch** den Stau. Die KI weicht einem Hindernis also niemals blind aus, sondern exakt nur dann, wenn die Zeitstrafe des Staus mathematisch größer ist als die Zeitstrafe des physischen Umwegs.

1.4 Warum Christofides scheitert: Die Dreiecksungleichung
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Der Christofides-Algorithmus ist das berühmteste Approximationsverfahren für das TSP. Er erfordert für seinen mathematischen Beweis jedoch zwingend einen metrischen Raum und die Einhaltung der Dreiecksungleichung. In unserem Modell ist dieser metrische Raum durch die BPR-Penalty-Funktion zerstört: Ein extremer Stau macht den direkten Weg von A nach C oft teurer als die Summe der Umwege über B. Christofides würde hier physikalisch unmögliche "Shortcuts" (z.B. durch Regale) generieren, da der Algorithmus die Zeitstrafen geometrisch nicht korrekt abbilden kann.

2. Das Symmetric Open TSP (Non-Metric)
--------------------------------------
Ein klassisches TSP sucht eine geschlossene Rundreise. Die Realität im Markt diktiert ein offenes Pfad-Problem: Start am Eingang, Ende an einer dynamisch berechneten Kasse. 

Da unser Basis-Graph ungerichtet ist und die Verkehrsstaus die Fluiddynamik der Menschenmassen in beide Laufrichtungen gleichermaßen blockieren, handelt es sich mathematisch um ein **Symmetric Open TSP**. Dennoch versagen Standard-Approximationen (wie Christofides), da die dynamischen Strafen das System zu einem sogenannten Non-Metric TSP machen (Verletzung der Dreiecksungleichung).

Anstatt das offene Problem durch das künstliche Hinzufügen von Dummy-Nodes aufzublähen, erzwingt die Architektur eine strikte **Separation of Concerns**. Der TSP-Solver wird mit einem freien Endknoten (``end=None``) aufgerufen und optimiert ausschließlich die Raum-Geometrie der Regale. Die Auswahl der optimalen Kasse wird bewusst vom Routing entkoppelt und erst in der anschließenden System-Synthese (Post-Processing) durch das Stochastik-Modul evaluiert.

3. Das Strategy-Pattern: Dynamische Algorithmen-Eskalation
----------------------------------------------------------
Um die O(N)-Laufzeitschranken des Servers zu schützen, lagert die Hauptfunktion ``calculate_hybrid_route`` die Logik in das abstrakte Interface ``RoutingStrategy`` aus. Das System implementiert eine vierstufige Eskalationsstrategie:

* **n <= 11: Held-Karp (Exakte DP).** Garantiert das absolute globale Zeit-Optimum.
* **12 <= n <= 15: Simulated Annealing.** Thermodynamische Suche gegen lokale Minima.
* **16 <= n <= 25: Ant Colony Optimization (ACO).** Schwarmbasierte Pfadfindung.
* **n > 25: Genetischer Algorithmus.** Biologische Evolution für gigantische Suchräume.

.. code-block:: python

   # Auszug aus dem Controller (calculate_hybrid_route):
   # Dynamische Zuweisung des Solvers zur Vermeidung von OOM-Abstürzen
   if n_targets <= CONFIG.DP_EXACT_LIMIT:
       solver = HeldKarpDPSolver()          # Exakt (O(N^2 * 2^N))
   elif n_targets > 25:
       solver = GeneticAlgorithmSolver()    # Biologische Evolution
   elif n_targets > CONFIG.SA_THRESHOLD: 
       solver = AntColonySolver()           # Schwarmintelligenz
   else:
       solver = SimulatedAnnealingSolver()  # Thermodynamik
       
   # Polymorpher Aufruf über das gemeinsame Interface (Open TSP mit end=None)
   store_seq, msg = solver.solve(d_mat, start_node, store_t, None)

4. Exakte Dynamische Programmierung: Der Pythonic Way
-----------------------------------------------------
Für kleine Warenkörbe (n <= 11) verlangt das System absolute Exaktheit. Die Klasse ``HeldKarpDPSolver`` reduziert die Brute-Force-Laufzeit O(N!) durch Dynamische Programmierung auf $O(N^2 \cdot 2^N)$. 

*Der Architektonische Kniff (Frozensets statt Bitmasking):* In Hardware-nahen Sprachen (C++) wird Held-Karp über 32-Bit-Integer und Bit-Shifting gelöst. In einer High-Level-Sprache wie Python führt manuelles Bit-Shifting jedoch zu Performance-Verlusten. Die Architektur nutzt daher den "Pythonic Way": **Frozensets**. Ein ``frozenset`` ist eine unveränderliche (immutable) Menge und damit im Gegensatz zu normalen Listen hashbar. 

Dies erlaubt es dem Algorithmus, die bereits besuchten Knoten (``unvisited: frozenset``) direkt als Schlüssel in einem Dictionary (der Memoization-Tabelle) abzulegen. Der Python-Interpreter löst dies intern über hochoptimierte C-Hashmaps in O(1), was den Server-RAM massiv entlastet.

.. code-block:: python

   class HeldKarpDPSolver(RoutingStrategy):
       def solve(self, dist_matrix: dict, start: str, targets: List[str], end: Optional[str]):
           memo = {} # Das Cache-Dictionary
           
           # Die innere Rekursionsfunktion
           def dp(curr: str, unvisited: frozenset) -> Tuple[float, List[str]]:
               if not unvisited:
                   return 0.0, [] # Basisfall (Open TSP: Keine Rückkehrkosten zum Start)
               
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
In der Stufe bis 25 Produkte nutzt die Architektur die ``AntColonySolver`` Klasse. Dieser Algorithmus simuliert das Verhalten von Ameisen bei der Futtersuche. 

Agenten ("Ameisen") durchlaufen den Graphen stochastisch und hinterlassen virtuelle Pheromone auf zeitlich effizienten Kanten. Durch die Verdunstungsrate (Evaporation) verschwinden schlechte Pfade, während sich auf dem optimalen Weg eine stabile Pheromonspur bildet. Dies erlaubt es, komplexe Stau-Situationen durch emergente Gruppenintelligenz zu umgehen. Die mathematische Entscheidung der Ameise für das nächste Regal berechnet sich aus dem Integral von Pheromonstärke $\tau^\alpha$ und Sichtbarkeit $\eta^\beta$:

.. code-block:: python

   # Auszug aus der AntColonySolver Klasse:
   # Eine Ameise steht an einem Regal (curr) und überlegt, wohin sie als nächstes geht.
   # Die Wahrscheinlichkeit für das nächste Regal (v) wird bestimmt durch das Integral aus:
   # Pheromonstärke (Alpha) * Sichtbarkeit/Kürzeste Distanz (Beta)
   
   prob = pheromones.get((curr, v), 1.0) ** CONFIG.ACO_ALPHA * ((1.0 / dist) ** CONFIG.ACO_BETA)
   probs.append((v, prob))
   
   tot_p = sum(pr for _, pr in probs)
   if tot_p > 0:
       # Roulette-Wheel-Selection: Stochastische Auswahl basierend auf den Gewichten.
       # Verhindert, dass Ameisen deterministisch immer den gleichen Weg laufen, 
       # und fördert die emergente Exploration des Suchraums.
       nxt = random.choices([v for v, _ in probs], weights=[pr/tot_p for _, pr in probs])[0] 

6. Evolutionäre Biologie: Der Genetische Algorithmus (GA)
---------------------------------------------------------
Für extreme Warenkörbe (n > 25) ist der Genetische Algorithmus das Mittel der Wahl. Er modelliert die Darwinsche Evolution über Generationen hinweg. Ein kritischer architektonischer Aspekt ist die Vermeidung von Duplikaten während der "Paarung" von Routen (Crossover). Hierzu implementiert die Architektur den **Partially Mapped Crossover (PMX)**.

Beim PMX wird ein zufälliger Gen-Abschnitt (Sub-Route) zwischen zwei Elternteilen getauscht. Die restlichen Positionen werden über eine Mapping-Tabelle so angepasst, dass jedes Produkt exakt einmal in der neuen Route vorkommt. Dies schützt die topologische Integrität der Einkaufsliste.

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

7. Checkout-Stochastik: M/M/1/K Approximation
---------------------------------------------
Eine räumliche Route scheitert ohne Einbeziehung der Kassenwartezeit. Unsere Architektur verzichtet bewusst auf die rechenintensive Pollaczek-Khintchine-Formel für log-normalverteilte Servicezeiten ($M/G/1$) und approximiert die Realität stattdessen performant als **M/M/1/K-Warteschlangenmodell**.

Die Klasse ``EnterpriseQueuingModel`` berechnet die Wartezeit unter Berücksichtigung der Ankunftsrate $\lambda$ (moduliert durch Sinus-Tageszeitkurven zur Abbildung der Rush-Hour) und einer Kapazitätsgrenze K = 10.

.. code-block:: python

   import math
   import random

   class EnterpriseQueuingModel:
       @staticmethod
       def calculate_wait_metrics(base_lambda: float, current_hour: int, checkout_id: str) -> dict:
           """ Approximiert die Wartezeit performant über M/M/1/K Markow-Ketten. """
           c, mu, K = 1, 1.5, 10
           
           # Sinus-Modulation für den Tageszeit-Faktor (Abbildung der Rush-Hour)
           time_factor = math.sin((current_hour - 8) / 12 * math.pi) * 1.5
           lam = max(0.2, base_lambda + time_factor)
           rho = lam / mu
           
           if rho == 1.0:
               lq = (K * (K - 1)) / (2 * (K + 1))
               pk = 1.0 / (K + 1)
           else:
               p0 = (1 - rho) / (1 - rho**(K + 1))
               pk = (rho**K) * p0
               # Erwartete Kundenanzahl Lq in der Schlange (geometrische Reihe)
               lq = (rho / (1 - rho)) - ((K + 1) * rho**(K + 1) / (1 - rho**(K + 1)))
               
           lambda_eff = lam * (1 - pk)
           wait_minutes = lq / lambda_eff if lambda_eff > 0 else 0.0
           
           # Rückgabe inkl. leichtem stochastischem Jitter
           return {"wait_sec": (wait_minutes * 60.0) + random.uniform(1.0, 5.0)}

8. System-Synthese: Das Decoupling von Raum und Zeit & Shadow-Routing
---------------------------------------------------------------------
Das finale Meisterstück der Architektur ist die zustandslose Synthese im Controller (``calculate_hybrid_route``). Das System entkoppelt die räumliche TSP-Routenfindung (Regale) strikt von der temporalen Stochastik (Kassenwartezeit).

Sobald der TSP-Solver die ideale Regal-Reihenfolge mit freiem Endknoten berechnet hat, extrahiert der Orchestrator das letzte Regal der Route. Von diesem Punkt aus berechnet er für alle verfügbaren Kassen das physikalische Integral aus Laufzeit und Stochastik: 

$$\text{Kosten}=\text{Dijkstra-Laufweg}+\text{M/M/1/K Wartezeit}+\text{Weg zum Ausgang}$$

Das absolute Minimum dieser Funktion bestimmt die Zielkasse, welche nahtlos an die Array-Liste des TSP-Solvers angefügt (stitched) wird. Da dieser Prozess zustandslos in exakt der Millisekunde iteriert, in der der Nutzer den Callback in der UI auslöst, fließen Echtzeit-Sensorik und Warteschlangentheorie deterministisch in die Routenplanung ein, ohne die Such-Komplexität des TSP-Solvers unnötig aufzublähen.

**Der Kybernetische Fallback-Schutz (Shadow-Routing):**
Um das System absolut "Bulletproof" gegen KI-Halluzinationen zu machen, berechnet der Orchestrator die finale Route im Hintergrund zwingend parallel im **Shadow-Mode**. Das System generiert eine Route auf dem normalen, staufreien Graphen (deterministische Baseline) und zeitgleich eine Route auf dem durch die KI mutierten Graphen. 

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