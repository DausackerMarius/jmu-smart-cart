Supermarkt-Topologie & Spatial Computing
========================================

Dieses Kapitel dokumentiert die fundamentale In-Memory-Datenstruktur der Live-Engine. Um die im Operations Research (Kapitel 7) definierten deterministischen und stochastischen Routing-Probleme performant zu lösen, muss die physische Verkaufsfläche in einen maschinenlesbaren, topologischen Raum übersetzt werden. Das System modelliert den Supermarkt hierfür als gerichteten, kanten-gewichteten Attribut-Graphen $G = (V, E, W)$.

Diese Architektur geht jedoch weit über einfache planare Knotendarstellungen hinaus. Als "Spatial Computing"-Engine schlägt sie die komplexe algorithmische Brücke zwischen drei völlig unterschiedlichen physikalischen Dimensionen:
1. Der 2D-Pixel-Ebene des React/Dash-Frontends (Touch-Eingaben und UI-Rendering).
2. Der metrischen, euklidischen Ebene der Supermarkt-Architektur (physische Laufwege in Metern).
3. Der temporalen Ebene der Machine-Learning-Stauprognose (Zeitkosten in Sekunden).

1. Graphentheoretische Fundierung: Der Directed Graph (DiGraph)
---------------------------------------------------------------
Anstatt eines starren Raster-Modells (Grids), welches den Raum in diskrete, rechenintensive Kacheln unterteilt, nutzt das System die in C optimierte Graphen-Bibliothek ``NetworkX``. Ein Supermarkt ist mathematisch ein dünnbesetzter Graph (Sparse Graph): Ein Regal ist physisch nur mit seinen unmittelbaren Kreuzungen verbunden, nicht mit allen anderen Regalen im Markt. 

Würde die Architektur eine klassische Adjazenzmatrix verwenden, entstünde bei V = 500 eine Datenstruktur mit 250.000 Speicherzellen, die fast ausschließlich mit Nullen gefüllt wäre. Um eine RAM-Explosion zu verhindern und L1/L2-CPU-Cache-Misses zu minimieren, forciert die Architektur intern ein **Adjazenzlisten-Modell**. Hierbei speichert jeder Knoten nur reale Kanten, was die Speicherkomplexität auf das theoretische Minimum von O(|V| + |E|) drückt.

Die fundamentale Architektur-Entscheidung liegt in der Wahl eines **gerichteten Graphen** (``nx.DiGraph``). Während sich Kunden in regulären Gängen bidirektional bewegen (was durch das Generieren zweier gerichteter Kanten $u \rightarrow v$ und $v \rightarrow u$ abgebildet wird), existieren im Supermarkt strikte topologische Singularitäten: Kassenzonen und Eingangsschranken.

2. Automatische Graphen-Generierung (Floorplan Skeletonization)
---------------------------------------------------------------
Ein System mit 500 Regalen manuell zu pflegen, widerspricht dem Prinzip der Skalierbarkeit. Wenn das Filialmanagement den Grundriss ändert (z.B. durch temporäre Aktionsflächen), muss sich der Graph autonom anpassen. 

Die Architektur generiert die Basis-Wegpunkte daher nicht händisch, sondern extrahiert sie algorithmisch aus dem CAD-Grundriss (Floorplan) des Supermarkts über die **Mediale Achsentransformation (Medial Axis Transform)**. Durch die Berechnung von Voronoi-Diagrammen zwischen den soliden Regalblöcken ermittelt der ETL-Prozess das mathematische "Skelett" der freien Gehflächen. Die Kanten (Skelettlinien) garantieren dabei automatisch den maximalen äquidistanten Abstand zu allen umgebenden Hindernissen. Der Endnutzer wird somit deterministisch exakt in der sicheren Mitte des Ganges geroutet.

3. Bipartite Topologie: Die Demontage des Geisterfahrer-Paradoxons
------------------------------------------------------------------
Ein naiver Modellierungsfehler ist es, die Regale selbst als befahrbare Knoten in den Navigationsgraphen zu integrieren. In diesem Fall würde der Dijkstra-Algorithmus die kürzesten Pfade *durch* das physische Regal hindurch berechnen. Die Engine löst dies durch eine strikte **Bipartite Topologie-Trennung**:

* **Navigable Nodes (Wege-Knoten):** Virtuelle Wegpunkte auf der medialen Achse der Gänge. Der Suchalgorithmus operiert *ausschließlich* auf diesem Subgraphen.
* **POI Nodes (Points of Interest / Regale):** Diese Knoten repräsentieren die Ware. Sie besitzen exakt eine gerichtete Kante (Distanz 0.0) von und zu ihrem nächstgelegenen Anker-Knoten im Gang.

Die bipartite Trennung reduziert den Suchraum des Algorithmus dramatisch. Bei insgesamt 1000 Knoten (davon 800 Regale und 200 Wegpunkte) operiert der Dijkstra-Algorithmus nicht auf der Gesamtmenge. Die Zeitkomplexität für den Fibonacci-Heap-basierten Dijkstra sinkt von O(V_total * log(V_total) + E) auf O(V_way * log(V_way) + E_way). Die Suchlatenz auf dem Server verringert sich dadurch exponentiell.

.. code-block:: json

   // Architektur-Vertrag: Das JSON-Schema der ETL-Pipeline
   {
     "navigable_nodes": {
       "WAY_01": {"coords": [10.5, 5.2]},
       "WAY_02": {"coords": [10.5, 15.2]}
     },
     "navigable_edges": [
       {"u": "WAY_01", "v": "WAY_02", "distance_m": 10.0, "bidirectional": true},
       {"u": "CHECKOUT_IN", "v": "CHECKOUT_OUT", "distance_m": 2.0, "bidirectional": false}
     ],
     "poi_nodes": {
       "SHELF_MILK": {"anchor_node": "WAY_01", "category": "Molkerei"}
     }
   }

.. code-block:: python

   import networkx as nx
   import json
   import threading
   from typing import Dict, Any

   class StoreTopology:
       def __init__(self, config_path: str = "routing_config.json"):
           # Mutex Lock für Thread-Sicherheit bei asynchronen ML-Updates
           self._lock = threading.Lock() 
           # DiGraph erzwingt saubere Einbahnstraßen-Logik nativ im Speicher
           self.G = nx.DiGraph() 
           self._hydrate_from_etl(config_path)

       def _hydrate_from_etl(self, config_path: str) -> None:
           """ Bipartiter Aufbau des Graphen aus der Single Source of Truth. """
           with open(config_path, "r") as file:
               data: Dict[str, Any] = json.load(file)
               
           for node_id, attr in data["navigable_nodes"].items():
               self.G.add_node(node_id, type="WAYPOINT", coords=attr["coords"])
               
           for edge in data["navigable_edges"]:
               u, v, dist = edge["u"], edge["v"], edge["distance_m"]
               self.G.add_edge(u, v, base_distance_m=dist)
               
               # Automatische Erzeugung der Rück-Kante für reguläre Gänge
               if edge.get("bidirectional", True):
                   self.G.add_edge(v, u, base_distance_m=dist)

           for poi_id, attr in data["poi_nodes"].items():
               self.G.add_node(poi_id, type="SHELF", category=attr["category"])
               # Bidirektionales Snapping an den Gang mit physikalischen Kosten von 0.0
               self.G.add_edge(poi_id, attr["anchor_node"], base_distance_m=0.0)
               self.G.add_edge(attr["anchor_node"], poi_id, base_distance_m=0.0)

4. Spatial Indexing & Affine Transformationen
---------------------------------------------
Tippt ein Kunde auf das Frontend-Display (z. B. Koordinate X=1920, Y=1080), liefert der Browser rohe Pixelwerte. Der Graph operiert jedoch streng im metrischen Raum. Bevor ein Such-Algorithmus greifen kann, muss das System diese Dimensionen übersetzen. 

*Stellen Sie sich vor, Sie legen eine transparente, verkleinerte Bauplan-Skizze (Pixel) über eine echte Landkarte (Meter). Um beide exakt übereinanderzulegen, müssen Sie die Skizze verschieben (Translation) und passend zoomen (Skalierung).*

Mathematisch erfolgt dies durch eine **Affine Transformation** über eine homogene Matrix:

$$ \begin{pmatrix} X_{meter} \\ Y_{meter} \\ 1 \end{pmatrix} = \begin{pmatrix} S_x & 0 & T_x \\ 0 & S_y & T_y \\ 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} X_{pixel} \\ Y_{pixel} \\ 1 \end{pmatrix} $$

Wobei $S$ der Skalierungsfaktor und $T$ der Translations-Offset ist. Nach der Konvertierung nutzt das System einen **K-d Baum (K-Dimensional Tree)**. Dieser Binärbaum wird in O(N log N) konstruiert und ermöglicht Nearest-Neighbor-Suchanfragen in exakt O(log N). Der K-d Baum kompensiert das "Fat-Finger-Syndrom" auf Touch-Displays deterministisch.

.. code-block:: python

   from scipy.spatial import cKDTree
   import numpy as np

   def build_spatial_index(self) -> None:
       """ Kompiliert den Suchbaum. Beschränkt auf POI-Knoten. """
       self.poi_ids = [n for n, d in self.G.nodes(data=True) if d.get('type') == 'SHELF']
       coords = [self.G.nodes[n]["coords"] for n in self.poi_ids]
       self.kd_tree = cKDTree(np.array(coords))

   def resolve_touch_event(self, x_pixel: float, y_pixel: float) -> str:
       """ Affine Transformation und K-d Baum Lookup in O(log N). """
       x_meter = (x_pixel * CONFIG.SCALE_X) + CONFIG.OFFSET_X
       y_meter = (y_pixel * CONFIG.SCALE_Y) + CONFIG.OFFSET_Y
       
       _, index = self.kd_tree.query([x_meter, y_meter])
       return self.poi_ids[index]

5. Sensor Fusion & Orthogonales Map-Matching
--------------------------------------------
Der Smart Cart lokalisiert sich via IoT-Sensorik. Rohdaten unterliegen jedoch Gaußschem Rauschen und Multipath-Interferenzen. Würde das UI diese 2D-Koordinaten unbereinigt rendern, sähe der Kunde den Wagen durch Regale gleiten. 

Die Architektur glättet das Signal zunächst temporal über einen asynchronen **Kalman-Filter**, der die physikalische Trägheit des Wagens als Prädiktionsmodell nutzt. Das System führt danach eine mathematische Dimensionsreduktion durch: Die Koordinate im Raum der Ebene wird orthogonal auf einen 1D-Unterraum (das Liniensegment des Ganges) projiziert (Edge-Snapping).

.. code-block:: python

   def orthogonal_projection(self, point: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
       """ 
       Vektorisierte orthogonale Projektion eines Punktes auf das Segment AB.
       Verhindert visuelles Clipping des Wagens mit Regal-Polygonen.
       """
       AB = B - A
       AP = point - A
       len_sq = np.dot(AB, AB)
       
       if len_sq == 0.0:
           return A
           
       # Skalarprodukt für die Projektion, hart begrenzt auf das Intervall [0, 1]
       t = np.clip(np.dot(AP, AB) / len_sq, 0.0, 1.0)
       return A + t * AB

6. Temporale Arbitrage & Concurrency (Thread-Safety)
----------------------------------------------------
Das Machine-Learning-Modell prognostiziert Staus in Sekunden. Ein fataler Architekturfehler wäre es, diese Zeitstrafen stur auf die Basisdistanz in Metern zu addieren (Einheiten-Konflikt). Das System erzwingt das Prinzip der **Temporalen Arbitrage**:

$$ W(e, t) = \frac{d_{base}(e)}{v_{walk}} + P_{traffic}(e, t) $$

Da in einem ASGI-Environment hunderte UI-Requests parallel auf den Graphen zugreifen, während ein Background-Worker die ML-Gewichte aktualisiert, sichert ein **Mutex-Lock** (``threading.Lock``) die In-Memory-Struktur vor korrupten Race Conditions.

.. code-block:: python

   def update_temporal_weights(self, ml_time_penalties: dict, walk_speed_mps: float = 1.2) -> None:
       """ Konvertiert Raum in Zeit. Thread-Safe durch Mutex-Locking. """
       with self._lock:
           for u, v in self.G.edges():
               dist_m = self.G[u][v]['base_distance_m']
               base_time_sec = dist_m / walk_speed_mps
               
               # ML liefert asymmetrische Strafen pro gerichteter Kante
               penalty_sec = ml_time_penalties.get(f"{u}_{v}", 0.0)
               self.G[u][v]['active_time_weight'] = base_time_sec + penalty_sec

7. Das Einbahnstraßen-Paradoxon
-------------------------------
Wie in Kapitel 1 dargelegt, löst die Architektur das Problem von Kassenzonen und Schranken nativ auf Datenstrukturebene. Da das System einen gerichteten Graphen (``nx.DiGraph``) verwendet, wird das Einbahnstraßen-Paradoxon implizit gelöst. 

Während reguläre Gänge beim ETL-Import zwingend zwei Kanten (Hin- und Rückweg) erhalten, wird an Kassenzonen ausschließlich die Vorwärtskante injiziert (definiert durch ``bidirectional: false`` im JSON-Artefakt). Ein Zurücknavigieren ist topologisch unmöglich, da der Routing-Solver im RAM schlichtweg keine physische Rückwärtskante findet, die er traversieren könnte. Dies macht fehleranfällige Custom-Attribute oder Sonderregeln im Dijkstra-Algorithmus obsolet.

8. UI Path Smoothing: Vektorisierte Catmull-Rom Splines
-------------------------------------------------------
Der Dijkstra-Algorithmus operiert auf einem eckigen Wegenetz (90-Grad-Abbiegungen). Physische Einkaufswagen bewegen sich jedoch in interpolierten Kurven. Um die Route organisch zu rendern, wendet die Engine **Uniform Catmull-Rom Splines** an. 

Die naive Implementierung über skalare For-Schleifen ist in Python extrem ineffizient. Die Architektur nutzt stattdessen die kubische charakteristische Matrix und wendet **NumPy-Broadcasting** an, um tausende Koordinaten simultan auf C-Ebene im RAM zu multiplizieren. Dies garantiert höchste Latenzfreiheit.

Die Matrix-Gleichung für den Spline-Vektor $P(t)$ lautet:
$$ P(t) = \frac{1}{2} \begin{bmatrix} 1 & t & t^2 & t^3 \end{bmatrix} \begin{bmatrix} 0 & 2 & 0 & 0 \\ -1 & 0 & 1 & 0 \\ 2 & -5 & 4 & -1 \\ -1 & 3 & -3 & 1 \end{bmatrix} \begin{bmatrix} P_0 \\ P_1 \\ P_2 \\ P_3 \end{bmatrix} $$

.. code-block:: python

   import numpy as np

   def vectorized_catmull_rom(P0: np.ndarray, P1: np.ndarray, P2: np.ndarray, P3: np.ndarray, num_points: int = 10) -> np.ndarray:
       """ 
       Hochperformante, vektorisierte Spline-Interpolation über Numpy Broadcasting.
       Garantiert, dass die Kurve exakt durch P1 und P2 verläuft.
       """
       # Zeitvektor t aufspannen (Shape: num_points x 1)
       t = np.linspace(0, 1, num_points).reshape(-1, 1)
       
       # Potenz-Matrix [1, t, t^2, t^3] (Shape: num_points x 4)
       T = np.hstack([np.ones_like(t), t, t**2, t**3])
       
       # Uniform Catmull-Rom Charakteristische Matrix (Shape: 4 x 4)
       M = np.array([
           [ 0,  2,  0,  0],
           [-1,  0,  1,  0],
           [ 2, -5,  4, -1],
           [-1,  3, -3,  1]
       ]) * 0.5
       
       # Kontrollpunkt-Matrix
       P = np.vstack([P0, P1, P2, P3])
       
       # Matrix-Multiplikation: (T @ M) @ P 
       # Führt Berechnungen iterativfrei im C-Backend von Numpy aus.
       return (T @ M) @ P

9. Integrität & Graceful Degradation
------------------------------------
Ein fehlerhaftes ETL-Update darf nicht zum Crash führen. Die Validierung prüft den Graphen beim Booten und aktiviert bei strukturellen Defekten deterministisch das Hard-Coded-Fallback-Szenario.

.. code-block:: python

   from core.ontology import StoreOntology

   def validate_integrity(self) -> None:
       """ Fail-Fast Validierung beim Booten der Live-Engine. """
       with self._lock:
           nav_nodes = [n for n, d in self.G.nodes(data=True) if d['type'] == 'WAYPOINT']
           subgraph = self.G.subgraph(nav_nodes)
           
           # Prüft schwache Erreichbarkeit im DiGraph
           if not nx.is_weakly_connected(subgraph):
               # Graceful Degradation: Fallback auf deterministische Notfall-Ontologie
               self._hydrate_from_etl(StoreOntology.EMERGENCY_FALLBACKS)
           
           if any(d.get('active_time_weight', 0) < 0 for _, _, d in self.G.edges(data=True)):
               raise ValueError("Kritisch: Negative Zeitgewichte verletzen Dijkstra!")

10. Algorithmische Komplexität (Big-O Analyse)
----------------------------------------------
Um die Skalierbarkeit des Systems im Enterprise-Kontext (Filialen mit $>10.000$ Artikeln) zu validieren, wurde die Zeit- und Speicherkomplexität der Kernkomponenten analysiert. Die Metriken beweisen, dass die gewählten Datenstrukturen Latenzspitzen im Frontend mathematisch ausschließen.

==================================  ========================================  ================================
**Komponente**                      **Zeitkomplexität (Time)**                **Speicherkomplexität (Space)**
----------------------------------  ----------------------------------------  --------------------------------
**DiGraph Konstruktion (RAM)**      O(|V| + |E|)                              O(|V| + |E|)
**K-d Tree Aufbau**                 O(N log N)                                O(N)
**K-d Tree Touch-Query**            O(log N)                                  O(1)
**Orthogonale Projektion**          O(1)                                      O(1)
**Temporale Arbitrage Update**      O(|E|)                                    O(1)
**Catmull-Rom (Vektorisiert)**      O(1) *C-Ebene*                            O(num_points)
==================================  ========================================  ================================