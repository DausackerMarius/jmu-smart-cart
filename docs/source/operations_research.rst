Operations Research & Routing-Strategien
========================================

Das mathematische Herzstück der Navigation im Smart Supermarket ist die Lösung des Traveling Salesperson Problems (TSP) – des Problems der Handlungsreisenden. Die Aufgabe klingt für den Endanwender intuitiv simpel: Finde den zeitlich kürzesten Weg, um alle Produkte auf dem Einkaufszettel genau einmal zu besuchen und am Ende an der Kasse zu bezahlen. 

In der Komplexitätstheorie der Informatik wird dieses Problem jedoch als NP-schwer (nondeterministic polynomial-time hard) eingestuft. Das bedeutet: Es existiert weltweit kein deterministischer Algorithmus, der dieses Problem für beliebig große Einkaufslisten in akzeptabler (polynomieller) Zeit perfekt lösen kann. 

Die mathematische Realität der kombinatorischen Explosion: Bei einer kleinen Einkaufsliste von 10 Produkten gibt es bereits über 3,6 Millionen mögliche Wege (10 Fakultät). Ein moderner Backend-Server berechnet dies in Millisekunden. Stehen jedoch 40 Produkte auf der Liste, existieren mehr mögliche Routen (40 Fakultät), als es Atome im sichtbaren Universum gibt. Der naive Versuch, dies auf einem Server exakt zu berechnen, würde in einem sofortigen Out-of-Memory-Error (OOM) oder einer jahrelangen Laufzeit resultieren.

Die Backend-Architektur des JMU Smart Cart Systems erzwingt daher keinen starren "One-Size-Fits-All"-Ansatz. Stattdessen implementiert sie ein intelligentes, dynamisch skalierendes Framework. Abhängig von der Größe des Warenkorbs schaltet das System autonom zwischen absoluter mathematischer Exaktheit (für kleine Listen) und thermodynamischen sowie evolutionären Schätzverfahren (für große Listen) um.

1. Architektonische Disruption: Die Demontage der akademischen Baseline
-----------------------------------------------------------------------
In der klassischen Fachliteratur und in universitären Vorlesungen wird das Routing-Problem auf 2D-Graphen oft durch eine feste, standardisierte Pipeline gelehrt: 
1. Der A*-Suchalgorithmus berechnet die Distanzen zwischen den Knoten.
2. Der Christofides-Algorithmus berechnet daraus die finale TSP-Route. 

Das JMU Smart Cart System verwirft diesen akademischen Standardansatz jedoch vollständig. Im Kontext einer performanten, cyber-physischen Live-Engine muss diese architektonische Entscheidung mathematisch rigoros verteidigt werden.

1.1 Warum A* scheitert: Die unzulässige Heuristik
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Der A*-Algorithmus ist extrem effizient, um den Weg von einem Punkt A nach einem Punkt B zu finden. Er nutzt eine räumliche Heuristik (meist die Manhattan-Distanz) als "Kompass", um nicht blind in die falsche Richtung zu suchen. 

Das architektonische Problem: A* garantiert nur dann den kürzesten Weg, wenn seine Heuristik zulässig (admissible) ist. Der Kompass darf die wahren Kosten zum Ziel niemals überschätzen (h(n) <= d(n, target)). Durch die Integration unseres Prädiktiven Traffic-Modells (welches prognostizierte Menschen-Staus als Zeitstrafen auf die Gänge addiert) wird diese eiserne Regel gebrochen. Ein geometrisch kurzer Weg (10 Meter) kann durch einen Stau plötzlich 50 Straf-Sekunden kosten. Eine freie Umleitung (20 Meter) dauert nur 20 Sekunden. Die klassische räumliche Heuristik würde den A*-Algorithmus hier in die Falle locken und ineffiziente Wege produzieren.

1.2 Der Floyd-Warshall-Fehlschluss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Für das TSP benötigen wir zwingend eine Distanzmatrix (Clique) – wir müssen die Distanz von *jedem* Zielprodukt zu *jedem* anderen Zielprodukt kennen. Ein gängiger Fehler im Software Engineering ist es, hier blind den Floyd-Warshall-Algorithmus einzusetzen, da dieser als Standardlösung für das All-Pairs Shortest Path (APSP) Problem gilt.

Der mathematische Beweis des Scheiterns: Floyd-Warshall berechnet die Pfade zwischen *allen* Knoten des Supermarkt-Graphen (V = 500). Die Laufzeitkomplexität beträgt zwingend O(V^3). Bei 500 Knoten sind das 125 Millionen Operationen, selbst wenn der Kunde nur 10 Produkte sucht. Das ist für eine Echtzeit-API absolut inakzeptabel.

1.3 Unsere gewählte Baseline: Der Multi-Source Dijkstra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Das System nutzt stattdessen den heuristikfreien Dijkstra-Algorithmus als Single-Source Shortest Path Variante. Dijkstra breitet sich im Graphen wie eine Wasserwelle aus und orientiert sich strikt an den echten Kantengewichten (inklusive Staus). Wir haben eine kleine Zielmenge K (die Produkte auf der Liste, z. B. K=15). Wir starten Dijkstra exakt 15 Mal. Die Komplexität sinkt auf O(K * (V log V + E)). Da K extrem viel kleiner als V ist, ist unser Ansatz um mehrere Größenordnungen schneller als Floyd-Warshall.

.. code-block:: python

   import networkx as nx
   from typing import List, Dict

   def build_distance_matrix(G_congested: nx.DiGraph, targets: List[str]) -> Dict[str, Dict[str, float]]:
       """
       Baut die Distanzmatrix (Clique) effizient in O(K * (V log V + E)) auf.
       Fokussiert sich ausschließlich auf die 'time_penalty' Kantengewichte.
       """
       matrix = {}
       for start_node in targets:
           # Dijkstra expandiert heuristikfrei und findet die exakten Zeitkosten 
           # zu ALLEN anderen Knoten des Graphen in einem einzigen Durchlauf.
           lengths = nx.single_source_dijkstra_path_length(
               G_congested, source=start_node, weight='time_penalty'
           )
           # Wir extrahieren nur die Distanzen zu unseren Ziel-Produkten
           matrix[start_node] = {
               end_node: lengths[end_node] for end_node in targets if end_node != start_node
           }
       return matrix

1.4 Warum Christofides in der Praxis scheitert: Die Dreiecksungleichung
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Der Christofides-Algorithmus ist das berühmteste Approximationsverfahren für das TSP (garantierte Approximationsgüte von 1.5). 
Das fatale Praxis-Problem: Er erfordert für seinen mathematischen Beweis zwingend einen ungerichteten Graphen und die Einhaltung der Dreiecksungleichung:

Distanz(A, C) <= Distanz(A, B) + Distanz(B, C)

Der direkte Weg von A nach C muss also immer kürzer oder gleich lang sein wie ein Umweg über B. In unserem Modell ist dieser metrische Raum jedoch völlig zerstört: Wir haben Einbahnstraßen vor den Kassen und asymmetrische KI-Strafen (ein Stau gilt oft nur für eine Laufrichtung). Durch diese Asymmetrie (ATSP - Asymmetric TSP) kollabiert die Dreiecksungleichung. Christofides würde direkte "Shortcuts" generieren, die physikalisch durch Regale führen oder gegen Einbahnstraßen verstoßen. Das Resultat wären invalide Geisterfahrer-Routen. 

Das System zwingt uns daher zu asymmetrischen Lösungsverfahren wie Held-Karp.

2. Das Asymmetric Open TSP & Der Dummy-Node-Trick
-------------------------------------------------
Ein klassisches TSP sucht eine geschlossene Rundreise (Start = Ziel). Die Realität im Supermarkt diktiert jedoch ein offenes Pfad-Problem: Der Kunde startet am Eingang, sammelt Produkte und endet an einer Kasse. Da die Kanten gerichtet sind, sprechen wir vom Asymmetric Hamiltonian Path Problem.

Um hochoptimierte TSP-Solver nutzen zu können, wendet die Architektur einen genialen Operations-Research-Kniff an: Die Injektion eines Dummy-Nodes (Geister-Knoten).

.. code-block:: python

   def inject_dummy_node(dist_matrix: dict, start_node: str, end_node: str) -> dict:
       """ Verwandelt ein offenes Pfad-Problem in ein geschlossenes ATSP. """
       dummy = "DUMMY_NODE"
       dist_matrix[dummy] = {}
       
       for node in list(dist_matrix.keys()):
           if node == dummy: continue
           
           # 1. Magische Brücke: Von der Kasse führt ein Weg (Kosten 0.0) zum Dummy.
           dist_matrix[end_node][dummy] = 0.0
           
           # 2. Vom Dummy führt ein Weg (Kosten 0.0) zurück zum Start.
           # Der Solver denkt nun, er könne eine kostenfreie Rundreise machen.
           dist_matrix[dummy][start_node] = 0.0
           
           # 3. Verbotene Wege: Der Dummy darf NIEMALS mitten im Einkauf besucht werden.
           if node != start_node: dist_matrix[dummy][node] = float('inf')
           if node != end_node:   dist_matrix[node][dummy] = float('inf')
           
       return dist_matrix

Der Ablauf: Der Solver berechnet nun ahnungslos eine geschlossene Rundreise. Da der Weg von der Kasse über den Dummy zum Eingang exakt 0.0 Sekunden kostet, legt der Solver diesen Weg zwingend an das Ende der Route. Das Backend durchtrennt die Route danach exakt am Dummy-Knoten und wirft diesen weg. Übrig bleibt der perfekte, lineare Weg vom Eingang zur Kasse.

3. Das Strategy-Pattern (Architektonische Kapselung)
----------------------------------------------------
Da kein einzelner Algorithmus alle Warenkorbgrößen von 5 bis 80 Produkten effizient bedienen kann, lagert die Architektur die Logik über das Strategy Pattern (ein Entwurfsmuster der Gang of Four) aus. Der Controller-Code ruft nur ein einheitliches Interface (solve) auf:

.. code-block:: python

   from core.interfaces import RoutingStrategy
   from core.strategies import HeldKarpDPSolver, SimulatedAnnealingSolver, GeneticAlgorithmSolver

   n = len(shopping_nodes)
   
   # Dynamische Zuweisung basierend auf der Listen-Größe
   if n <= 15: 
       # Garantiert das absolute mathematische Optimum. Limit = 15 Produkte.
       solver: RoutingStrategy = HeldKarpDPSolver()
   elif n <= 40: 
       # Thermodynamische Metaheuristik für mittlere Einkäufe. Limit = 40 Produkte.
       solver: RoutingStrategy = SimulatedAnnealingSolver()
   else: 
       # Biologisch inspirierte Schwarm-Evolution als Fallback für Großeinkäufe.
       solver: RoutingStrategy = GeneticAlgorithmSolver() 
       
   route, compute_time = solver.solve(dist_matrix, start_node, shopping_nodes, end_node)

4. Unsere Baseline: Dynamische Programmierung (Held-Karp)
---------------------------------------------------------
Für kleine Warenkörbe (n <= 15) verlangt das System zwingend die globale Optimallösung. Die Klasse HeldKarpDPSolver reduziert die Brute-Force-Laufzeit O(N!) durch das Konzept der Dynamischen Programmierung (DP) massiv auf O(N^2 * 2^N).

Verständnis-Exkurs: Anstatt den Baum der Möglichkeiten immer wieder von vorne abzugehen, zerlegt Held-Karp die Route in Teil-Routen. Hat das System den besten Weg für die ersten 5 Produkte gefunden, merkt es sich diese Kosten in einer Tabelle (Memoization). Wenn ein anderer Ast des Algorithmus später wieder bei diesen 5 Produkten ankommt, wird das Zwischenergebnis direkt aus dem RAM abgerufen.

Die Herausforderung: Speichereffizienz & Backtracking
Die Architektur nutzt Bitmasking auf Hardware-Ebene. Ein 32-Bit Integer (z. B. 0101) repräsentiert binär den Zustand: "Produkt 1 und Produkt 3 sind besucht". 
Ein weiteres massives Problem: Der Algorithmus liefert primär nur die Kosten. Das System muss daher zwingend den "Parent-Knoten" (Vorgänger) in der Matrix speichern, um den Pfad am Ende rückwärts aufrollen zu können (Backtracking). Ohne Backtracking gäbe es keine Koordinaten für das Tablet.

.. code-block:: python

   def solve_held_karp(dist_matrix: list, n: int) -> list:
       """
       Löst das ATSP exakt und gibt den Pfad durch Backtracking zurück.
       """
       # memo speichert Tupel: (Minimale_Kosten, Vorgänger_Knoten_ID)
       # 1 << n nutzt Bitshifting, um 2^N extrem schnell auf CPU-Ebene zu berechnen.
       memo = [[(float('inf'), -1)] * n for _ in range(1 << n)]
       memo[1][0] = (0.0, -1) # Startknoten (Bit 0 ist 1) hat Kosten 0
       
       # 1. DP-Tabellenaufbau (Vorwärts-Phase)
       for mask in range(1, 1 << n):
           for i in range(n):
               if mask & (1 << i): # Bit-AND: Ist Knoten i besucht?
                   for j in range(n):
                       if not (mask & (1 << j)): # Ist Knoten j unbesucht?
                           next_mask = mask | (1 << j) # Aktiviere das j-te Bit
                           
                           # Bellman-Gleichung: Kosten bis i + Distanz von i nach j
                           new_cost = memo[mask][i][0] + dist_matrix[i][j]
                           
                           # Speichere das Minimum inkl. Vorgänger für Backtracking
                           if new_cost < memo[next_mask][j][0]:
                               memo[next_mask][j] = (new_cost, i)
                               
       # 2. Backtracking Phase (Rückwärts-Phase zur Pfad-Rekonstruktion)
       last_mask = (1 << n) - 1 # Alle Bits auf 1
       
       # Finde den letzten Knoten der Rundreise mit den geringsten Gesamtkosten
       last_node = min(range(1, n), key=lambda x: memo[last_mask][x][0] + dist_matrix[x][0])
       
       route = []
       current_mask, current_node = last_mask, last_node
       
       while current_node != -1:
           route.append(current_node)
           prev_node = memo[current_mask][current_node][1]
           current_mask ^= (1 << current_node) # Bit-XOR schaltet das aktuelle Bit ab
           current_node = prev_node
           
       return route[::-1] # Pfad umdrehen (Start -> Ziel)

Das Limit bei n=15 schützt die CPU. Bei n=20 explodiert 2^20 auf über 419 Millionen Zustände und würde den Server sofort lahmlegen.

5. Thermodynamische Metaheuristik: Simulated Annealing (SA)
-----------------------------------------------------------
Für mittlere Listen (16 bis 40 Artikel) nutzt die SimulatedAnnealingSolver Klasse einen Algorithmus, der das langsame thermodynamische Abkühlen von Metallen simuliert.

Das Problem lokaler Minima: Ein simpler Suchalgorithmus (Hill-Climbing) tauscht Routen-Stationen nur so lange aus, bis er keine Verbesserung mehr findet. Dabei verharrt er oft in einem lokalen Minimum (er verpasst die perfekte Route, weil er dafür kurzzeitig einen schlechteren Umweg akzeptieren müsste).

Die Lösung: Simulated Annealing akzeptiert temporär auch schlechtere Routen. Bei initial hoher Temperatur (T) testet der Algorithmus wild Umwege, um den Suchraum zu erkunden. Je kühler das System wird, desto strenger akzeptiert es nur Verbesserungen. Die Wahrscheinlichkeit P für Verschlechterungen basiert auf der Metropolis-Hastings-Funktion: P = exp(-Delta E / T).

Um die Route im Code zu mutieren, nutzt die Architektur die asymmetrische 2-Opt-Heuristik. Ein einfaches Umdrehen des Arrays (naiver 2-Opt via Python-Slice) würde bei Einbahnstraßen zu unendlichen Geisterfahrer-Kosten führen. Der Algorithmus berechnet stattdessen die Kosten der neu zusammengesetzten Kanten explizit neu.

.. code-block:: python

   import random
   import math

   def simulated_annealing(dist_matrix, initial_route, initial_temp=1000.0, cooling_rate=0.995):
       current_route = initial_route
       current_distance = calculate_total_dist(current_route, dist_matrix)
       temp = initial_temp
       
       while temp > 1.0:
           # 2-Opt Swap: Wähle zwei zufällige Schnittpunkte im Array
           i, j = sorted(random.sample(range(1, len(current_route)-1), 2))
           
           # Generierung des neuen Zustands
           new_route = current_route[:i] + current_route[i:j][::-1] + current_route[j:]
           new_distance = calculate_total_dist(new_route, dist_matrix)
           
           delta = new_distance - current_distance
           
           # Akzeptanz-Bedingung: 
           # Wenn besser (Delta < 0) ODER wenn die Metropolis-Funktion zuschlägt
           if delta < 0 or math.exp(-delta / temp) > random.random():
               current_route = new_route
               current_distance = new_distance
               
           temp *= cooling_rate # Thermodynamisches Abkühlen
           
       return current_route

6. Evolutionäre Biologie: Der Genetische Algorithmus (GA)
---------------------------------------------------------
Für extreme Großeinkäufe (n > 40) versagt Simulated Annealing, da der Suchraum gigantisch wird. Der ultimative Fallback ist die GeneticAlgorithmSolver Klasse. Sie modelliert die Darwinsche Evolution.

Ein Genetischer Algorithmus besteht aus vier Phasen: Initialisierung, Selektion, Crossover (Paarung) und Mutation. Um das beste genetische Material nicht zu zerstören, nutzt das System Elitismus: Die besten 5% der Routen einer Generation werden unangetastet übernommen.

.. code-block:: python

   import random

   def genetic_algorithm(dist_matrix, nodes, pop_size=150, generations=500):
       # 1. Initialisierung: Erschaffe 150 zufällige Routen (Chromosomen)
       population = [generate_random_route(nodes) for _ in range(pop_size)]
       
       for generation in range(generations):
           # Bewerte Fitness (1 / Gesamtdistanz). Je kürzer, desto fitter.
           fitness_scores = [1.0 / calculate_total_dist(route, dist_matrix) for route in population]
           
           new_population = []
           
           # Elitismus: Die besten 5% überleben unmutiert
           elite_count = int(pop_size * 0.05)
           elites = get_best_routes(population, fitness_scores, elite_count)
           new_population.extend(elites)
           
           # 2. Selektion (Roulette Wheel Selection) & 3. Crossover
           while len(new_population) < pop_size:
               parent_a = roulette_wheel_selection(population, fitness_scores)
               parent_b = roulette_wheel_selection(population, fitness_scores)
               
               # Order 1 Crossover (OX1): Verhindert Duplikate bei der Fortpflanzung
               child = order_1_crossover(parent_a, parent_b)
               
               # 4. Mutation: 5% Chance auf zufälligen Swap, schützt vor Inzucht
               if random.random() < 0.05:
                   child = mutate_swap(child)
                   
               new_population.append(child)
               
           population = new_population
           
       return get_best_routes(population, fitness_scores, 1)[0]

Das Crossover-Problem: Wenn sich zwei Eltern-Routen paaren, kann man sie nicht einfach in der Mitte durchschneiden und zusammenfügen. Das Resultat wäre eine Route, auf der Produkte doppelt vorkommen und andere fehlen (Verletzung der Permutations-Regel). Die Architektur implementiert daher zwingend den deterministischen Order 1 Crossover (OX1):

.. code-block:: python

   def order_1_crossover(parent_a: list, parent_b: list) -> list:
       """ Vererbt topologische Eigenschaften, ohne Duplikate zu erzeugen. """
       size = len(parent_a)
       child = [None] * size
       
       # Wähle ein zufälliges Sub-Array von Parent A und vererbe es exakt 
       # an die gleiche physische Position im Kind-Array.
       start, end = sorted(random.sample(range(size), 2))
       child[start:end] = parent_a[start:end]
       
       # Fülle leere Plätze mit fehlenden Elementen von Parent B auf.
       # Wir starten direkt NACH dem vererbten Block, um die relative 
       # Reihenfolge (Graphen-Wegrichtung) von Parent B zu bewahren.
       pointer = end
       for item in parent_b:
           if item not in child:
               if pointer == size: pointer = 0 # Array Wrap-around
               child[pointer] = item
               pointer += 1
               
       return child

7. Checkout-Stochastik: M/M/1/K & Pre-Halftime Prädiktion
---------------------------------------------------------
Ein rein metrisches TSP ignoriert eine der volatilsten Variablen im Supermarkt: Die Wartezeit an den Kassen. 
Ein fundamentaler Architektur-Fehler statischer Navigationssysteme ist das späte Routing: Erfährt der TSP-Solver erst am Ende des Einkaufs, welche Schlange die kürzeste ist, fehlt ihm der topologische Manövrierraum. Der Kunde müsste abrupt umdrehen und gegen den Strom navigieren.

Der Stochastik-Algorithmus (M/M/1/K):
Die QueuingModelFacade berechnet die Wartezeit nicht durch simple Längenmessung via Kameras, sondern prädiktiv. Sie nutzt das M/M/1/K-Warteschlangenmodell (Kendall-Notation). 
Warum M/M/1/K und nicht M/M/c? Am Flughafen bilden alle Kunden eine einzige Schlange für mehrere Schalter (M/M/c). Im Supermarkt jedoch wählt der Kunde eine dedizierte Kasse mit genau einem Kassierer (M/M/1). Die Kapazität ist räumlich limitiert (K). 
Das Modell basiert auf Little's Law (L = lambda * W), welches besagt, dass die durchschnittliche Anzahl der Kunden (L) gleich der Ankunftsrate (lambda) multipliziert mit der durchschnittlichen Verweildauer (W) ist.

.. code-block:: python

   def predict_checkout_wait_time(lam: float, mu: float, k_capacity: int) -> float:
       """
       Berechnet die stochastische Wartezeit an einer dedizierten Supermarktkasse.
       lam = Ankunftsrate (Kunden pro Minute)
       mu = Bedienrate (Kassiervorgänge pro Minute)
       """
       if lam == 0: return 0.0
       rho = lam / mu # Auslastungsgrad (Traffic Intensity)
       
       if rho == 1.0:
           # Sonderfall L'Hôpital: Genau so viele Ankünfte wie Abfertigungen
           expected_customers = k_capacity / 2
       else:
           # M/M/1/K Formel zur Bestimmung der erwarteten Systemlänge
           numerator = 1 - (k_capacity + 1) * (rho ** k_capacity) + k_capacity * (rho ** (k_capacity + 1))
           denominator = (1 - rho) * (1 - (rho ** (k_capacity + 1)))
           expected_customers = rho * (numerator / denominator)
           
       # Wartezeit W = L / lambda (Abgeleitet aus Little's Law)
       return expected_customers * (1 / mu)

Der architektonische Pre-Halftime-Trigger:
Das System lauscht asynchron auf den Warenkorbfortschritt des Nutzers. Die Vorhersage muss zwingend vor der Halbzeit (bei 40% bis 45% Fortschritt) berechnet werden. 

Die Kasse mit der kürzesten prognostizierten Wartezeit wird frühzeitig als fixierter End-Knoten in den Dummy-Node-Trick injiziert. Dies garantiert, dass der TSP-Solver in der zweiten Einkaufshälfte den Graphen strategisch nutzt, um den Kunden fließend und ohne kognitive Brüche in Richtung des besten Ausgangs zu routen.