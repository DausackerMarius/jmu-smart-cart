Operations Research & Routing-Strategien
========================================

Das mathematische Herzstück der Navigation im Smart Supermarket ist die Lösung des **Traveling Salesperson Problems (TSP)** – des Problems der Handlungsreisenden. Die Aufgabe klingt trivial: Finde den kürzesten Weg, um alle Produkte auf dem Einkaufszettel genau einmal zu besuchen und an der Kasse zu enden. 

In der Komplexitätstheorie der Informatik wird dieses Problem jedoch als **NP-schwer** (nondeterministic polynomial-time hard) eingestuft. Das bedeutet: Es existiert weltweit kein deterministischer Algorithmus, der dieses Problem für große Einkaufslisten in akzeptabler Zeit perfekt lösen kann. 

*Verständnis-Exkurs:* Bei 10 Produkten auf der Liste gibt es bereits über 3,6 Millionen mögliche Wege ($10!$). Ein Server rechnet das in Millisekunden. Bei 50 Produkten ($50!$) gibt es jedoch mehr mögliche Routen, als es Atome im sichtbaren Universum gibt. Der Server würde bis an das Ende der Zeit rechnen und mit einem Out-of-Memory-Error abstürzen.

Die Backend-Architektur des JMU Smart Cart Systems erzwingt daher keinen starren "One-Size-Fits-All"-Ansatz. Stattdessen implementiert sie ein dynamisch skalierendes Framework. Abhängig von der "Entropie" (der Größe des Warenkorbs) schaltet das System autonom zwischen absoluter mathematischer Exaktheit und speichereffizienten Schätzverfahren (Metaheuristiken) um.

1. Architektonische Disruption: Die Demontage der akademischen Baseline
-----------------------------------------------------------------------
In der klassischen Fachliteratur und in universitären Arbeiten wird das Routing-Problem auf Gitter-Graphen (Grid-Graphs) standardmäßig durch eine feste Pipeline gelöst: Der **A*-Suchalgorithmus** berechnet die Distanzen zwischen den Produkten, und der **Christofides-Algorithmus** berechnet daraus die finale Route. 

Das JMU Smart Cart System verwirft diesen akademischen Standardansatz jedoch vollständig aus zwei tiefgreifenden mathematischen Gründen:

1.1 Der Verlust der A*-Zulässigkeit (Admissibility)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Der A*-Algorithmus ist extrem beliebt, da er nicht blind in alle Richtungen sucht, sondern eine "Heuristik" (einen Kompass) nutzt. In einem Supermarkt ist das meist die *Manhattan-Distanz* (die Zählung der Raster-Blöcke). 
*Das mathematische Problem:* A* garantiert nur dann den kürzesten Weg, wenn seine Heuristik **zulässig (admissible)** ist. Das bedeutet, der Kompass darf die wahren Kosten zum Ziel niemals überschätzen: $h(n) \le d(n, \text{target})$.

Durch die Integration unseres **Prädiktiven Traffic-Modells** (welches prognostizierte Menschen-Staus als Zeitstrafen auf die Gänge addiert) wird diese eiserne Regel gebrochen. Ein geometrisch kurzer Weg (10 Meter) kann durch einen Stau plötzlich 50 Straf-Sekunden kosten. Eine freie Umleitung (20 Meter) dauert nur 20 Sekunden. Die klassische Manhattan-Heuristik würde den A*-Algorithmus in Stausituationen in *lokale Minima* zwingen (er würde sich verheddern, weil der Kompass lügt). 

**Die Lösung (Multiple Dijkstra):** Das System nutzt stattdessen den heuristikfreien **Dijkstra-Algorithmus**. Dijkstra tastet das Netzwerk wie eine sich ausbreitende Wasserwelle ab. Da er keinen geometrischen Kompass nutzt, lässt er sich von Stau-Strafen nicht täuschen und operiert auch auf dynamisch gewichteten Graphen zu 100 % exakt.

.. code-block:: python

   def calculate_distance_matrix(G_congested: nx.Graph, targets: List[str]) -> dict:
       """
       Baut die Matrix mit Dijkstra auf. Ignoriert die Geometrie, fokussiert sich 
       ausschließlich auf das Kantengewicht (welches die KI-Staus enthält).
       """
       matrix = {}
       for start_node in targets:
           # Dijkstra expandiert heuristikfrei und findet die ECHTEN Zeitkosten
           # weight='time_penalty' zwingt den Algorithmus, Staus zu umgehen
           lengths = nx.single_source_dijkstra_path_length(
               G_congested, source=start_node, weight='time_penalty'
           )
           matrix[start_node] = {end_node: lengths[end_node] for end_node in targets}
       return matrix

1.2 Das Scheitern von Christofides an der Dreiecksungleichung
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Der Christofides-Algorithmus ist berühmt, weil er eine garantierte Approximationsgüte von 1.5 liefert. 
*Das mathematische Problem:* Er setzt zwingend die Gültigkeit der **Dreiecksungleichung** voraus ($d(A, C) \le d(A, B) + d(B, C)$). In einem echten Supermarkt ist dieser metrische Raum jedoch nicht gegeben. Durch Einbahnstraßen-Regelungen vor den Kassen und unsere asymmetrischen KI-Strafen (ein Stau in Gang 3 gilt oft nur für eine Laufrichtung) kollabiert die Dreiecksungleichung. Ein statischer Algorithmus wie Christofides würde hier physikalisch invalide Routen (Geisterfahrer-Wege) erzeugen. 

2. Pre-Processing & Das "Open ATSP" Problem
-------------------------------------------
Bevor eine Route berechnet wird, abstrahiert das System das physische Supermarkt-Labyrinth. Der TSP-Solver darf keine Wände oder leeren Gänge sehen. Er erhält einen vollständig verbundenen Graphen (eine *Clique*), der ausschließlich die Zielprodukte enthält (Metrischer Abschluss).

**Das Asymmetric Open TSP & Der Dummy-Node-Trick:**
Ein klassisches TSP sucht eine geschlossene Rundreise (Start = Ziel). Die Realität im Supermarkt diktiert jedoch ein *Asymmetric Hamiltonian Path Problem*: Der Kunde startet am Eingang, läuft die Produkte ab und endet an der Kasse. Er läuft *nicht* zurück zum Eingang. Zudem machen Einbahnstraßen die Wege asymmetrisch (Weg A $\rightarrow$ B $\neq$ B $\rightarrow$ A).

Um dieses Problem effizient zu lösen, wendet die Architektur einen genialen Code-Kniff an: Den **Dummy-Node** (einen unsichtbaren Geister-Knoten).

.. code-block:: python

   def inject_dummy_node(dist_matrix: dict, start_node: str, end_node: str) -> dict:
       """
       Verwandelt ein offenes Pfad-Problem in ein geschlossenes TSP.
       """
       dummy = "DUMMY_NODE"
       dist_matrix[dummy] = {}
       
       for node in dist_matrix.keys():
           if node == dummy: continue
           
           # Kosten vom Dummy zum Eingang sind 0 (Brücke)
           dist_matrix[dummy][start_node] = 0.0
           
           # Kosten von der Kasse zum Dummy sind 0 (Kreis wird geschlossen)
           dist_matrix[end_node][dummy] = 0.0
           
           # Alle anderen Wege zum/vom Dummy sind verboten (Kosten = Unendlich)
           if node != start_node: dist_matrix[dummy][node] = float('inf')
           if node != end_node:   dist_matrix[node][dummy] = float('inf')
           
       return dist_matrix

Der Algorithmus berechnet nun gezwungenermaßen eine geschlossene Rundreise. Da der Rückweg von der Kasse zum Eingang über den Dummy exakt `0` Sekunden kostet, nutzt der Solver diese Brücke. Am Ende schneidet das System die Liste genau an diesem Dummy-Knoten programmatisch auf und erhält einen perfekten linearen Pfad.

3. Architektonisches Fundament: Das Strategy-Pattern
----------------------------------------------------
Da kein Algorithmus alle Warenkorbgrößen bedienen kann, ist das Routing softwaretechnisch über das **Strategy-Pattern** (Gang of Four) gekapselt. Die Basisklasse zwingt alle Solver zur Implementierung der Methode ``solve()``. Der Orchestrator in ``model.py`` entscheidet zur Laufzeit anhand harter Schwellenwerte:

.. code-block:: python

   n = len(shopping_nodes)
   
   # Dynamische Laufzeit-Zuweisung des optimalen Solvers
   if n <= SystemConfig.DP_EXACT_LIMIT:          # Limit = 15 (OOM Protection)
       solver = HeldKarpDPSolver()
   elif n <= SystemConfig.SA_THRESHOLD:          # Limit = 40
       solver = SimulatedAnnealingSolver()
   else:                                         # n > 40
       solver = GeneticAlgorithmSolver() 
       
   route = solver.solve(dist_matrix, start_node, shopping_nodes, end_node)

4. Exakte Lösung: Dynamische Programmierung (Held-Karp)
-------------------------------------------------------
Für Warenkörbe mit $n \le 15$ liefert das System deterministisch die **globale Optimallösung**. Die Klasse ``HeldKarpDPSolver`` reduziert die Fakultätslaufzeit $\mathcal{O}(n!)$ durch Dynamische Programmierung auf $\mathcal{O}(n^2 2^n)$.

*Verständnis-Exkurs (Dynamische Programmierung):* Anstatt alle Millionen Wege von vorne bis hinten durchzuprobieren, zerlegt Held-Karp die Route in Teil-Routen. Hat das System den besten Weg für die ersten 5 Produkte gefunden, merkt es sich diesen in einer Tabelle. Spätere Berechnungen greifen einfach auf dieses Zwischenergebnis zurück.

**Komplexitäts-Mathematik & Bitmasking:**
Um einen Out-of-Memory-Error beim Speichern der Rekursionsbäume zu verhindern, nutzt der Code **Bitmasking**. Ein 32-Bit-Integer (z. B. `01011`) repräsentiert binär den Zustand, welche Produkte bereits im Wagen liegen.

.. code-block:: python

   def solve_held_karp(dist_matrix: list, n: int) -> float:
       # Initialisierung der DP-Tabelle (2^n Zeilen, n Spalten) mit Unendlich
       # 1 << n nutzt Bitshifting, um rasend schnell 2^n zu berechnen
       memo = [[float('inf')] * n for _ in range(1 << n)]
       memo[1][0] = 0 # Startknoten (Bit 0) hat Kosten 0
       
       # Iteration über alle möglichen Teilmengen (Bitmasks)
       for mask in range(1, 1 << n):
           for i in range(n):
               # Prüfen via bitweisem AND: Ist Knoten i in der Menge 'mask' enthalten?
               if mask & (1 << i):
                   for j in range(n):
                       # Ist Knoten j noch NICHT in der Menge? Dann füge ihn hinzu!
                       if not (mask & (1 << j)):
                           next_mask = mask | (1 << j) # Bitweises OR aktiviert das j-te Bit
                           
                           # Bellman-Gleichung: Bisherige Kosten + Distanz zu j
                           new_cost = memo[mask][i] + dist_matrix[i][j]
                           
                           # Speichere das Minimum im RAM
                           if new_cost < memo[next_mask][j]:
                               memo[next_mask][j] = new_cost
                               
       # Rückgabe der kürzesten geschlossenen Tour
       return min(memo[(1 << n) - 1]) 

Das Limit von $n=15$ ist zwingend: Bei $n=15$ benötigt die CPU ca. 7,3 Millionen Iterationen (< 0.2 Sekunden). Bei $n=20$ wären es bereits 419 Millionen, was den Server-Worker einfrieren würde.

5. Thermodynamische Metaheuristiken: Simulated Annealing (SA)
-------------------------------------------------------------
Für mittlere Listen (16 bis 40 Artikel) nutzt die ``SimulatedAnnealingSolver`` Klasse einen physik-inspirierten Algorithmus, der das thermodynamische Abkühlen von Metallen simuliert. 

Ein simpler lokaler Suchalgorithmus (Hill-Climbing) würde in einem lokalen Optimum steckenbleiben. Simulated Annealing akzeptiert temporär auch *schlechtere* Routen. Bei hoher "Temperatur" testet der Algorithmus wild Umwege. Je kühler das System wird, desto strenger akzeptiert es nur noch Verbesserungen. Die Akzeptanzwahrscheinlichkeit $P$ wird durch die Metropolis-Hastings-Funktion gesteuert:

.. math::
   P(\Delta E, T) = \exp\left(-\frac{\Delta E}{T}\right)

**Code-Implementierung (Asymmetrischer 2-Opt-Swap):**
Um neue Pfade zu generieren, wendet der Algorithmus die 2-Opt-Heuristik an (Kreuzungen werden aufgelöst). Da wir ein ATSP (Einbahnstraßen) lösen, würde ein naiver Python-Slice ``[::-1]`` die Route umdrehen und zu unendlichen Geisterfahrer-Kosten führen. Der Algorithmus validiert daher jeden Swap.

.. code-block:: python

   def simulated_annealing(dist_matrix, route, initial_temp=1000.0, cooling_rate=0.995):
       current_route = route
       current_distance = calculate_total_distance(current_route, dist_matrix)
       temp = initial_temp
       
       while temp > 1.0:
           # 2-Opt Swap: Wähle zwei zufällige Schnittpunkte im Array
           i, j = sorted(random.sample(range(1, len(current_route)-1), 2))
           
           # Zerschneiden und teilweises Umkehren der Sub-Route
           new_route = current_route[:i] + current_route[i:j][::-1] + current_route[j:]
           new_distance = calculate_total_distance(new_route, dist_matrix)
           
           delta = new_distance - current_distance
           
           # Akzeptanz-Bedingung: Wenn besser (<0) ODER Metropolis-Funktion zuschlägt
           if delta < 0 or math.exp(-delta / temp) > random.random():
               current_route = new_route
               current_distance = new_distance
               
           # Abkühlen des Metalls
           temp *= cooling_rate 
           
       return current_route

6. Schwarmintelligenz & Evolutionäre Algorithmen (GA)
-----------------------------------------------------
Für extreme Großeinkäufe (über 40 Artikel) versagt auch Simulated Annealing, da der Suchraum gigantisch wird. Als extrem robuster Fallback dient die ``GeneticAlgorithmSolver`` Klasse. Sie modelliert eine Population von 150 "Chromosomen" (zufälligen Routen) und simuliert Darwinsche Evolution.

*Das Problem:* Wenn zwei gute Routen ("Eltern") sich paaren, kann man im Code nicht einfach beide Arrays in der Mitte zerschneiden und mit ``parent_a[:5] + parent_b[5:]`` zusammenkleben. Das Resultat wäre ein invalider Graph, bei dem Produkte doppelt vorkommen und andere völlig fehlen.

Die Architektur implementiert daher den deterministischen **Order 1 Crossover (OX1)**:

.. code-block:: python

   def order_1_crossover(parent_a: list, parent_b: list) -> list:
       """
       Vererbt topologische Eigenschaften, ohne Duplikate zu erzeugen.
       """
       size = len(parent_a)
       child = [None] * size
       
       # 1. Wähle ein zufälliges Sub-Array von Parent A und vererbe es exakt
       start, end = sorted(random.sample(range(size), 2))
       child[start:end] = parent_a[start:end]
       
       # 2. Fülle die leeren Plätze mit den fehlenden Elementen von Parent B auf
       pointer = end
       for item in parent_b:
           if item not in child:
               if pointer == size: pointer = 0 # Array Wrap-around
               child[pointer] = item
               pointer += 1
               
       return child

Ein Sub-Array von Elternteil A wird exakt an dieselbe Stelle an das Kind vererbt. Die verbleibenden leeren Plätze werden mit den Produkten von Elternteil B aufgefüllt, exakt in der Reihenfolge, in der sie dort auftauchen. So bleibt die DNA einer guten Route erhalten, ohne die Graphen-Physik zu verletzen. Eine Swap-Mutation (5 %) schützt die Herde vor Inzucht (Premature Convergence).

7. Checkout-Stochastik: M/M/1/K & Halftime-Prädiktion
-----------------------------------------------------
Das rein metrische TSP ignoriert eine der dynamischsten Variablen im Supermarkt: Die Wartezeit an den Kassen. 

Die finale Entscheidung, an welcher Kasse die Route endet, basiert auf der Warteschlangentheorie. Ein fundamentaler architektonischer Fehler statischer Navigationssysteme ist das späte Routing: Erfährt der TSP-Solver erst am Ende des Einkaufs, welche Schlange die kürzeste ist, fehlt ihm der topologische Manövrierraum. Der Kunde müsste abrupt umdrehen und bereits passierte Kreuzungen erneut kreuzen.

Um diese topologischen Sackgassen zu vermeiden, berechnet die Systemarchitektur die optimale Kasse nicht erst im letzten Schritt. 

**Der Stochastik-Algorithmus (M/M/1/K):**
Die ``QueuingModelFacade`` berechnet die voraussichtliche Wartezeit nicht durch simple Längenmessung der Schlange, sondern stochastisch. Sie nutzt das M/M/1/K-Modell: 
Ankünfte ($\lambda$) und Kassiervorgänge ($\mu$) sind poisson-verteilt (Markov-Prozess), es gibt einen Kassierer (1) und eine maximale Schlangenkapazität (K).

.. code-block:: python

   def predict_checkout_wait_time(lam: float, mu: float, k_capacity: int) -> float:
       """
       Berechnet die stochastische Wartezeit an einer Kasse via M/M/1/K.
       lam = Ankunftsrate (Kunden pro Minute)
       mu = Bedienrate (Kassiervorgänge pro Minute)
       """
       if lam == 0: return 0.0
       rho = lam / mu # Auslastungsgrad (Traffic Intensity)
       
       if rho == 1.0:
           # Sonderfall: Genau so viele Ankünfte wie Abfertigungen
           expected_customers = k_capacity / 2
       else:
           # Standard M/M/1/K Formel für die erwartete Schlangenlänge (Little's Law)
           numerator = 1 - (k_capacity + 1) * (rho ** k_capacity) + k_capacity * (rho ** (k_capacity + 1))
           denominator = (1 - rho) * (1 - (rho ** (k_capacity + 1)))
           expected_customers = rho * (numerator / denominator)
           
       # Wartezeit = Erwartete Kunden in der Schlange * Zeit pro Kunde
       return expected_customers * (1 / mu)

**Der architektonische Pre-Halftime-Trigger:**
Das System lauscht asynchron auf den Warenkorbfortschritt des Nutzers. Ein fataler Fehler wäre es, die Vorhersage erst bei oder nach Überschreiten der Halbzeit anzufordern (Race Condition durch Netzwerklatenz). Die Architektur erzwingt daher, dass die Kassen-Prädiktionen zwingend **vor der Halbzeit** (beispielsweise bei 40 % bis 45 % Einkaufsfortschritt) berechnet und ausgeliefert werden. 

Die Kasse mit der kürzesten prognostizierten Wartezeit wird so frühzeitig als verifizierter ``end_node`` in die Distanzmatrix des laufenden TSP-Solvers injiziert, noch bevor die zweite Einkaufshälfte überhaupt beginnt. Dies garantiert, dass die TSP-Matrix dem Algorithmus den maximalen topologischen Freiraum gibt, um die verbleibenden Produkte fließend und ohne kognitive Brüche in Richtung des optimalen Ausgangs zu priorisieren.