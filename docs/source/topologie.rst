Supermarkt-Topologie & Graphen-Mapping
======================================

Nachdem im vorherigen Kapitel die bimodale Backend-Architektur und der Lebenszyklus eines Requests definiert wurden, taucht dieses Kapitel tief in die fundamentale In-Memory-Datenstruktur der Live-Engine ein: Die Graphen-Topologie.

Bevor algorithmische Optimierungen für Laufwege berechnet oder Produkte durch die ETL-Pipeline zugewiesen werden können, muss die physische Realität des JMU Smart Supermarkets in ein maschinenlesbares, mathematisches Format überführt werden (Spatial Computing).

Anstelle eines simplen zweidimensionalen Arrays (wie es in klassischen, naiven A*-Pfadfindungs-Algorithmen oder Pixel-Grids genutzt wird), modelliert das System den Supermarkt als **gerichteten, kanten-gewichteten Attribut-Graphen** $G = (V, E, W)$. 

Diese architektonische Entscheidung ist zwingend: Ein echter Supermarkt weist durch physische Barrieren (Regalreihen) und strikte Laufrichtungen (Einbahnstraßen an der Kasse) topologische Restriktionen auf. Ein Pixel-Grid würde in der Wegfindung ein "Clipping" riskieren (der Algorithmus könnte mathematisch diagonal durch ein Regal hindurch navigieren). Ein Graph hingegen definiert durch seine Kanten exakt den Raum, in dem Bewegung physikalisch legal ist. Wo keine Kante ist, existiert für das System kein Raum.

Dieses Kapitel dokumentiert die programmatische Konstruktion dieses Graphen, das räumliche Koordinatensystem, die Speicherarchitektur der Knoten ($V$), die mathematische Definition der Kanten ($E$) und die Implementierung hochperformanter Raum-Suchbäume (K-d Trees).

1. Graphentheoretisches Fundament: Speichereffizienz & NetworkX
---------------------------------------------------------------
Das System nutzt die stark optimierte, in C-Strukturen verwurzelte Python-Bibliothek ``NetworkX``. Eine kritische Design-Entscheidung bei der Modellierung von Graphen ist die Wahl der internen Datenstruktur im Arbeitsspeicher: **Adjazenzmatrix vs. Adjazenzliste**.

Ein Supermarkt ist von Natur aus ein *dünnbesetzter Graph* (Sparse Graph). Ein Regalplatz ist physisch immer nur mit der Kreuzung links und rechts verbunden, niemals mit allen 5.000 anderen Regalen im Gebäude. 

*Verständnis-Exkurs:* Würde das System eine **Adjazenzmatrix** nutzen, sähe das im RAM aus wie eine riesige Excel-Tabelle mit 5.000 Zeilen und 5.000 Spalten. Jede Zelle speichert die Distanz der Regale zueinander. Da 99 % der Regale nicht direkt miteinander verbunden sind, wäre diese Tabelle fast ausschließlich mit Nullen gefüllt. Der Arbeitsspeicher würde mit einer Komplexität von $\mathcal{O}(|V|^2)$ explodieren und die CPU-Caches verstopfen. 
Das System nutzt stattdessen eine **Adjazenzliste**. Das Prinzip ähnelt einem lokalen Adressbuch: Jeder Knoten speichert *nur* seine tatsächlichen, physischen Nachbarn. Die Speicherkomplexität schrumpft dadurch auf das theoretische absolute Minimum von $\mathcal{O}(|V| + |E|)$.

.. declaration:google:search{queries: ["Adjacency list vs adjacency matrix graph representation data structure"]}


.. code-block:: python

   import networkx as nx
   from typing import Dict, Any, Tuple

   class StoreTopology:
       """
       Verwaltet den In-Memory Graphen des Supermarktes.
       Operiert strikt auf Adjazenzlisten zur OOM-Vermeidung (Out of Memory).
       """
       def __init__(self):
           # nx.DiGraph erzwingt einen gerichteten Graphen. 
           # Dies ist die zwingende mathematische Voraussetzung für Einbahnstraßen.
           self.G: nx.DiGraph = nx.DiGraph()
           
           self._build_base_topology()
           self._validate_topology() # Fail-Fast Integritätsprüfung beim Boot

       def _build_base_topology(self) -> None:
           """Konstruiert den statischen Bauplan in O(|V| + |E|) Laufzeit."""
           self._inject_nodes()
           self._inject_edges()

2. Räumliches Referenzsystem & Knoten-Architektur (Vertices)
------------------------------------------------------------
Jeder Knoten $v \in V$ im Graphen repräsentiert eine diskrete, begehbare Raum-Zone. 

**Das Koordinaten-Referenzsystem (CRS):**
Anstatt fehleranfällige globale GPS-Koordinaten (Längen-/Breitengrade) zu nutzen, spannt das System ein lokales, kartesisches 2D-Grid auf. Der Nullpunkt $(0.0, 0.0)$ ist starr auf den Haupteingang kalibriert. Alle Koordinaten werden als Gleitkommazahlen in "Metern" abgebildet, was eine 1:1-Skalierung zwischen der Routing-Mathematik und der physischen Realität garantiert.

*Verständnis-Exkurs (Python Memory Overhead):* Warum werden Knoten als einfache Dictionaries via ``add_node`` eingefügt und nicht als schöne, objektorientierte Python-Klassen (z. B. ``class ShelfNode: ...``)? Python weist jedem nativen Objekt intern ein eigenes ``__dict__`` zur Attributverwaltung zu. Bei Tausenden von Graphen-Knoten würde dieser Metadaten-Overhead den RAM belasten. Die Nutzung der in C optimierten NetworkX-Attribute ist massiv speichereffizienter.

Die Ontologie erzwingt eine strikte Klassifizierung der Knoten:

2.1 Hardware-gebundene Knoten (Fixed Zones)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Bestimmte Knoten sind physisch an die Supermarkt-Hardware gebunden und dürfen vom nachgelagerten ETL-Prozess niemals überschrieben werden. Ein Kühlregal erfordert Starkstrom, eine Kasse erfordert Netzwerkkabel.

.. code-block:: python

   def _inject_fixed_zones(self) -> None:
       """Injiziert Zonen mit unveränderlicher Hardware-Limitierung."""
       self.G.add_node(
           "MEAT_COUNTER_1", 
           type="FIXED_ZONE",
           category="Fleischtheke",
           hardware_requirement="COOLING_UNIT_3 PHASE",
           # Physische Verortung: 12.5m rechts, 45.0m tief vom Eingang
           coordinates=(12.5, 45.0) 
       )

2.2 Das dynamische Container-Konzept (Flexible Zones)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Die Mehrheit der Regal-Knoten fungiert initial beim Boot-Vorgang als leere Allokations-Slots. Um die Physis eines echten Regals abzubilden, besitzt jeder dieser Knoten ein hartes Kapazitätslimit (``max_capacity``). Dies ist ein algorithmischer Selbstschutz: Der Graph verhindert dadurch aktiv, dass die Daten-Pipeline später 100 Milchprodukte in ein Regal zwängt, das physikalisch nur für 10 ausgelegt ist.

.. code-block:: python

   def _inject_flexible_zones(self) -> None:
       """Injiziert leere Regale, die später vom ETL-Prozess befüllt werden."""
       self.G.add_node(
           "R_D_6", # Regal Reihe D, Segment 6
           type="FLEXIBLE_ZONE",
           max_capacity=SystemConfig.MAX_SHELF_CAPACITY, # Physisches Limit (z. B. 15)
           capacity_used=0,
           stock=[] # Leeres Array, das via Sainte-Laguë gefüllt wird
       )

2.3 Spatial Indexing (Der K-d Baum)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Wenn ein Nutzer auf dem Tablet-Display auf einen Punkt der Karte tippt, sendet das Frontend eine nackte Fließkomma-Koordinate (z. B. $X=12.45, Y=44.91$) an das Backend. Das Backend muss nun das "Spatial Mapping" durchführen: *Welches Regal meint der Kunde?*
Eine naive lineare Suche (Ein For-Loop über alle 5.000 Graphenknoten zur Distanzmessung) würde eine Suchzeit von $\mathcal{O}(|V|)$ erfordern und das System bei parallelen Anfragen extrem ausbremsen.

**Die algorithmische Lösung:** Das System instanziiert beim Boot parallel zum Graphen einen **K-d Baum (k-dimensional tree)**.

*Verständnis-Exkurs:* Ein K-d Baum ist eine binäre Suchstruktur, die den Supermarkt algorithmisch in Zonen aufteilt. Er zieht eine Linie in der Mitte des Raumes und fragt: "Ist die gesuchte Koordinate links oder rechts?" Dann zieht er eine horizontale Linie und fragt: "Oben oder unten?". Dies iteriert er, bis jedes Regal in einer eigenen, winzigen Box liegt. Der Aufbau dieses Baums kostet initial $\mathcal{O}(|V| \log |V|)$ Zeit. Sucht das Tablet nun den nächsten Knoten, muss der Server nicht alle 5.000 Regale prüfen, sondern folgt einfach dem Entscheidungsbaum. Die Suchzeit kollabiert dadurch auf rasante logarithmische Zeit $\mathcal{O}(\log |V|)$.

.. declaration:google:search{queries: ["kd tree spatial partitioning explanation diagram", "k-dimensional tree nearest neighbor search"]}


.. code-block:: python

   from scipy.spatial import cKDTree
   import numpy as np

   def _build_spatial_index(self) -> None:
       """
       Baut den k-d Baum für ultraschnelle Nearest-Neighbor Suchen.
       Nutzt explizit cKDTree (C-Implementierung), um das Python GIL zu umgehen.
       """
       # 1. Extrahiere alle Koordinaten und IDs aus dem Graphen
       self.node_ids = list(self.G.nodes())
       coordinates = [self.G.nodes[n]["coordinates"] for n in self.node_ids]
       
       # 2. Kompiliere den Suchbaum im C-Backend von SciPy
       self.kd_tree = cKDTree(np.array(coordinates))

   def get_nearest_node(self, x: float, y: float) -> str:
       """Mappt einen Screen-Tap in O(log V) auf die echte Graphen-ID."""
       # query() liefert die euklidische Distanz und den Array-Index des Knotens
       distance, index = self.kd_tree.query([x, y])
       return self.node_ids[index]

*Architektonische Resilienz:* Da ein K-d Baum eine statische Datenstruktur ist, erzwingt die Architektur, dass dieser Baum bei jeder topologischen Änderung des Graphen atomar neu berechnet wird. Dies verhindert "Stale Data" (das fehlerhafte Verweisen auf Regale, die im Graphen nicht mehr existieren).

3. Kanten-Metriken und topologisches Mapping (Edges)
----------------------------------------------------
Nachdem die Knoten im Raum positioniert sind, müssen sie durch gerichtete Kanten $e \in E$ verbunden werden. Die Basis-Metrik (das initiale Kantengewicht $w$) entspricht der reinen euklidischen Distanz. Dies geschieht hochpräzise über den Satz des Pythagoras in Python (``math.hypot``):

.. code-block:: python

   import math

   def _inject_edges(self) -> None:
       """
       Verbindet Knoten und berechnet das physikalische Startgewicht.
       Erzwingt asymmetrische Graphen-Logik für Einbahnstraßen.
       """
       # 1. Definition eines normalen, in beide Richtungen begehbaren Ganges
       u, v = "R_D_6", "R_D_7"
       coord_u = self.G.nodes[u]["coordinates"]
       coord_v = self.G.nodes[v]["coordinates"]
       
       # Berechnung der physikalischen Distanz d_base(e) in Metern
       dist = math.hypot(coord_v[0] - coord_u[0], coord_v[1] - coord_u[1])
       
       # Bidirektionale Kanten-Injection (Laufwege in beide Richtungen erlaubt)
       self.G.add_edge(u, v, weight=dist, edge_type="aisle_transit")
       self.G.add_edge(v, u, weight=dist, edge_type="aisle_transit")
       
       # 2. Physische Einbahnstraße (Kassenzone)
       # Der Algorithmus addiert NUR die Kante HIN zur Kasse. 
       # Eine Rück-Kante (v nach u) existiert im Graphen physikalisch nicht!
       self.G.add_edge("CHECKOUT_WAIT", "CHECKOUT_PAY", weight=2.5) 

*Verständnis-Exkurs:* Diese rein geometrische Distanz fungiert später bei Algorithmen wie Dijkstra als unbestechliche Basis-Metrik. Der enorme architektonische Vorteil eines *gerichteten* Graphen (``nx.DiGraph``) zeigt sich an den Kassen: Durch das schlichte Weglassen der Kante von ``CHECKOUT_PAY`` zurück nach ``CHECKOUT_WAIT`` modelliert das System eine perfekte Einbahnstraße. Der Routing-Algorithmus kann den Kunden nach draußen navigieren, aber die Mathematik verbietet ihm, wieder zurück in den Laden zu routen.

4. System-Resilienz & Integritätsprüfung
----------------------------------------
Ein häufiger und schwerwiegender Fehler in räumlichen Graphen-Systemen ist die Entstehung von "Inseln" (z. B. ein Regal, das durch einen Tippfehler im Code mit keiner Kante verbunden ist). Der Routing-Algorithmus würde versuchen, den Weg dorthin zu finden, endlos suchen und abstürzen.

Um die Produktionsreife zu garantieren, führt die ``StoreTopology`` direkt nach dem Boot-Vorgang eine strenge Integritätsprüfung durch (das industrielle Fail-Fast-Prinzip). Das Backend verweigert den Start konsequent, wenn die Graphen-Physik verletzt ist:

.. code-block:: python

   class GraphTopologyError(Exception): pass

   def _validate_topology(self) -> None:
       """Graphentheoretischer Safety-Check beim Server-Boot."""
       # 1. Prüfe auf Isolierung: Sind alle Knoten topologisch erreichbar?
       if not nx.is_weakly_connected(self.G):
           raise GraphTopologyError("CRITICAL BOOT FAILURE: Graph enthält isolierte Inseln!")
           
       # 2. Dijkstra Precondition Check: Existieren physikalisch unmögliche Distanzen?
       # (Negative Kantengewichte würden Dijkstra in Endlosschleifen treiben)
       if any(weight < 0 for _, _, weight in self.G.edges(data='weight')):
           raise GraphTopologyError("CRITICAL BOOT FAILURE: Negative Kantengewichte entdeckt!")

5. Brücke zur Live-Engine: Die Gewichts-Mutation
------------------------------------------------
Dieses Kapitel hat die statische Topologie erschaffen. Das Alleinstellungsmerkmal dieses Backends ist jedoch, dass die Kanten im Live-Betrieb nicht statisch bleiben. 

Wie im Routing-Kapitel (Operations Research) ausführlich beschrieben, greift das KI-Modell im laufenden Betrieb exakt auf diese ``weight``-Attribute der Kanten zu und mutiert sie dynamisch. Die finale Gewichtsfunktion für eine Kante $e$ zum Zeitpunkt $t$ ist programmatisch definiert als:

.. math::

   W(e, t) = d_{base}(e) + \lambda \cdot P_{traffic}(e, t)

* **Die Baseline ($d_{base}$):** Ist exakt die hier im Topologie-Modul berechnete, physikalische Pythagoras-Distanz.
* **Der Penalty-Term ($P_{traffic}$):** Ist die vom ML-Modell prognostizierte Personenanzahl. Der Skalar $\lambda$ übersetzt im RAM Personen in künstliche "Straf-Meter", wodurch Gänge algorithmisch länger werden.

Damit das Frontend diese gigantische Raumstruktur konsumieren kann, wird der verifizierte In-Memory-Graph via ``nx.node_link_data()`` in das standardisierte JSON-Format serialisiert. 
Das einzige verbleibende Problem am Ende dieses Boot-Prozesses: **Die Knoten des Graphen (die Regale) sind noch komplett leer.**

Genau an dieser architektonischen Schnittstelle übergibt die Topologie den Staffelstab an die **Data-Engineering-Pipeline**. Im folgenden Kapitel wird erläutert, wie diese leeren Raum-Zonen aus rohen B2B-Datenbanken vollautomatisch extrahiert, linguistisch transformiert und algorithmisch via Sainte-Laguë-Verfahren befüllt werden.