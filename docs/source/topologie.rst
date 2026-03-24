Supermarkt-Topologie & Graphen-Mapping
======================================

Bevor intelligente Routing-Algorithmen Laufwege berechnen oder Produkte durch eine Daten-Pipeline in Regale sortiert werden können, muss die physische Realität des JMU Smart Supermarkets in ein maschinenlesbares, mathematisches Format überführt werden. In der Informatik nennt man diese Brücke zwischen physischem Raum und digitaler Logik **Spatial Computing** (Räumliches Rechnen), wodurch ein sogenannter **Digitaler Zwilling (Digital Twin)** des Gebäudes entsteht.

Ein klassischer Architekturfehler in rudimentären Pfadfindungs-Anwendungen ist die Modellierung des Raumes als simples zweidimensionales Pixel-Raster (Grid). Ein solches Raster birgt in der Wegfindung immer das Risiko des sogenannten "Clippings" – der Algorithmus könnte mathematisch verbotenerweise diagonal durch ein Regal hindurch navigieren, wenn die Suchheuristik es verlangt. 

Dieses System modelliert den Supermarkt stattdessen zwingend als **gerichteten, kanten-gewichteten Attribut-Graphen** G = (V, E). Ein gerichteter mathematischer Graph definiert durch seine Kanten (Edges, E) exakt den Raum, in dem Bewegung physikalisch legal ist. Wo keine Kante programmiert wurde, existiert für das System schlichtweg kein begehbarer Raum. Regale und Wände stellen somit natürliche, undurchdringbare Barrieren dar.

Dieses Kapitel dokumentiert den programmatischen Bauplan dieses Graphen. Es dekonstruiert die Speichereffizienz der zugrundeliegenden Datenstruktur, die fehlerresistente Deserialisierung der JSON-Topologie, die Lösung des Indoor-GPS-Problems, die physikalische Transformation von Metern in Laufzeit-Metriken sowie die Implementierung hochperformanter Raum-Suchbäume (K-d Trees).

1. Graphentheoretisches Fundament: Adjazenzliste vs. Matrix
-----------------------------------------------------------
Das System nutzt für das Graphen-Mapping die in C-Strukturen verwurzelte Python-Bibliothek ``NetworkX``. Bevor der Graph im Arbeitsspeicher (RAM) des Servers materialisiert wird, muss eine fundamentale Design-Entscheidung getroffen werden: Wie wird der Graph intern repräsentiert? Die theoretische Informatik unterscheidet hier primär zwischen der **Adjazenzmatrix** und der **Adjazenzliste**.

Ein Supermarkt ist graphentheoretisch ein extrem *dünnbesetzter Graph* (Sparse Graph). Ein Regalplatz ist physisch immer nur mit den unmittelbaren Kreuzungen davor und dahinter verbunden. Der durchschnittliche Knotengrad (Degree k, also die Anzahl der ausgehenden Kanten) liegt bei k ≈ 3.

1.1 Der mathematische Beweis der Speichereffizienz
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ein Prüfer im Kolloquium wird zwingend fragen, warum die Wahl der Datenstruktur direkte Auswirkungen auf die Stabilität eines Web-Servers hat. Der Beweis liegt in der Big-O-Speicherkomplexität:

Würde das System eine **Adjazenzmatrix** nutzen, entspräche dies im RAM einer Tabelle der Größe V x V. Nehmen wir einen realistischen Supermarkt mit V = 5000 Knoten (Regalmeter und Kreuzungen) an. Die Matrix bestünde aus 5000 x 5000 = 25.000.000 Speicherzellen. Da jedes Regal nur ca. 3 Nachbarn hat, wären über 99,9 % dieser Zellen mit Nullen gefüllt. Die Speicherkomplexität wächst quadratisch mit O(V^2). Bei der Instanziierung von parallelen Graphen-Kopien (z.B. für Mutationen durch die KI) würde der Server sofort mit einem **Out-of-Memory-Error (OOM)** abstürzen.

Die Klasse ``StoreTopology`` erzwingt daher durch ``NetworkX`` intern die Nutzung einer **Adjazenzliste**. Jeder Knoten speichert in einem kompakten Hash-Dictionary *ausschließlich* seine tatsächlichen, physischen Nachbarn. 

*Code-Exkurs (Die Python-Implementierung):* Unter der Haube speichert NetworkX den Graphen als verschachteltes Dictionary (``G[u][v] = weight``). Der Zugriff auf einen Nachbarn erfolgt durch Hashing in konstanter Zeit O(1). Die Speicherkomplexität schrumpft auf das theoretische absolute Minimum von O(V + E). Bei 5000 Knoten und ca. 15000 Kanten verwaltet das System nur noch 20000 Speicherverweise statt 25 Millionen – eine Reduktion des RAM-Bedarfs um den Faktor 1250.

.. code-block:: python

   import networkx as nx
   import logging

   class StoreTopology:
       """
       Verwaltet den In-Memory Graphen des Supermarktes als Digitalen Zwilling.
       Operiert strikt auf Hash-basierten Adjazenzlisten zur OOM-Vermeidung.
       """
       def __init__(self, config_filepath: str):
           # nx.DiGraph initialisiert einen gerichteten Graphen (Directed Graph). 
           # Dies ist die zwingende mathematische Voraussetzung, um später 
           # asymmetrische Laufwege (wie Kassen-Einbahnstraßen) legal abbilden zu können.
           self.G: nx.DiGraph = nx.DiGraph()
           
           # Der strukturierte Lebenszyklus der Topologie-Erstellung:
           self._parse_and_build(config_filepath)
           self._build_spatial_index()
           self._validate_topology() # Fail-Fast Integritätsprüfung beim Server-Boot

2. Indoor-Positioning & Dimensionalitätsreduktion (Z-Achse)
-----------------------------------------------------------
Jeder Knoten v in V repräsentiert eine diskrete Raum-Zone (z.B. ein Regal oder eine Wegkreuzung). Doch wie verortet die Architektur diese im Raum? Ein klassischer Anfängerfehler in der Systemarchitektur wäre es, hier globale GPS-Koordinaten (WGS84) zu verwenden. 

*Das physikalische Problem:* GPS-Signale können die dicken Stahlbetondecken eines modernen Supermarktes nicht zuverlässig durchdringen (Faradayscher Käfig). Das Signal wird durch Wände reflektiert, was zum sogenannten **Multi-Path-Effekt** führt. Die Genauigkeit eines Smartphones fällt im Gebäude oft auf einen Radius von 10 bis 15 Metern ab. Ein Supermarktregal ist jedoch nur 1 Meter breit. GPS ist für Spatial Computing in Innenräumen unbrauchbar.

*Die Lösung:* Das System abstrahiert die Welt und spannt ein lokales, unbestechliches **kartesisches 2D-Grid (Coordinate Reference System - CRS)** auf. Der Nullpunkt P(0.0, 0.0) wird starr auf den physischen Haupteingang des Marktes kalibriert. Alle Koordinaten (X, Y) werden als IEEE-754 Fließkommazahlen (Floats) in der Maßeinheit "Meter" abgebildet. Dies garantiert eine millimetergenaue, 1:1-Skalierung zwischen der Graphen-Mathematik im RAM und dem echten Fußboden.

*Die Abstraktion der Z-Achse (Dimensionalitätsreduktion):* Ein Supermarkt ist physisch dreidimensional; Produkte liegen auf verschiedenen Regalhöhen (Z-Achse). Die Architektur entscheidet sich bewusst gegen Vektoren im Raum (X, Y, Z) und nutzt nur (X, Y). Die Z-Achse ist für die *Laufwege-Mathematik* (das Routing) irrelevant. Die topologischen Kosten für das Bücken oder Strecken nach einem Produkt werden mathematisch elegant in die sogenannte Interaktionszeit (siehe Phase 4) ausgelagert. Dieser Kniff reduziert den Speicherbedarf der Vektoren um 33 % und halbiert die Baumtiefe des K-d Index, ohne dass das System an Präzision für die Navigation verliert.

3. Deserialisierung, Resilienz & Knoten-Ontologie
-------------------------------------------------
Das Layout des Marktes darf auf keinen Fall fest im Python-Code verankert sein (Hardcoding), da Supermärkte regelmäßig durch Aktionsware oder Umbauten ihr Layout ändern. Das System lädt den Graphen dynamisch aus einer standardisierten JSON-Datei. 

Da externe Dateien potenziell fehlerhaft oder unvollständig sind, darf der Parser nicht blind auf Schlüsselwörter (Keys) zugreifen. Ein fehlender Key würde einen KeyError werfen und den Webserver beim Booten komplett zerstören. Die Deserialisierung nutzt daher strikt sichere Dictionary-Zugriffsfunktionen mit sinnvollen Default-Werten.

*Der Python Memory-Overhead:* Ein Prüfer könnte fragen, warum das System für Knoten reine Dictionaries und keine klassischen objektorientierten Klassen nutzt. Die Antwort liegt im Speicher-Design von Python (CPython): Python weist nativen Klassen-Objekten intern ein eigenes, verstecktes Dictionary zur Attributverwaltung zu. Ein leeres Objekt belegt so bereits über 100 Bytes. Bei Tausenden von Graphen-Knoten würde dieser Metadaten-Overhead den Arbeitsspeicher massiv aufblähen (Bloat). Die direkte Injektion von simplen Key-Value-Paaren in das NetworkX-Objekt ist architektonisch zwingend geboten.

Die Ontologie klassifiziert die Topologie beim Einlesen streng in zwei Knotentypen:

.. code-block:: python

   import json

   class StoreTopology:
       # ... [__init__ Methode wie zuvor] ...

       def _parse_and_build(self, filepath: str) -> None:
           """ 
           Deserialisiert das JSON-Layout von der Festplatte und materialisiert 
           den Graphen im flüchtigen Arbeitsspeicher. 
           """
           with open(filepath, 'r', encoding='utf-8') as file:
               layout = json.load(file)
               
           # 1. Knoten (Vertices) sicher injizieren
           for node_data in layout.get("nodes", []):
               node_id = node_data.get("id")
               
               # Resilienz: Überspringt korrupte JSON-Einträge ohne Absturz
               if not node_id:
                   continue 
                   
               node_type = node_data.get("type", "FLEXIBLE_ZONE")
               coords = tuple(node_data.get("coordinates", (0.0, 0.0)))
               
               # Fall A: Fixed Zones (Hardware-gebundene Knoten)
               # Diese Knoten (z.B. Tiefkühltruhen) sind an physische Starkstromanschlüsse 
               # gebunden. Sie dürfen vom späteren KI-Algorithmus nicht verschoben werden.
               if node_type == "FIXED_ZONE":
                   self.G.add_node(
                       node_id, 
                       type=node_type,
                       category=node_data.get("category", "Unbekannt"),
                       coordinates=coords
                   )
                   
               # Fall B: Flexible Zones (Das Container-Konzept)
               # Diese Regale sind initial leere Allokations-Slots.
               # Das 'max_capacity' Attribut ist ein essenzieller algorithmischer Schutz, der 
               # verhindert, dass später 100 Produkte in ein Regal für 15 Artikel gepresst werden.
               elif node_type == "FLEXIBLE_ZONE":
                   self.G.add_node(
                       node_id, 
                       type=node_type,
                       max_capacity=node_data.get("max_capacity", 15),
                       capacity_used=0,
                       stock=[], # Leeres Array, bereit für die Data-Engineering Pipeline
                       coordinates=coords
                   )

4. Kanten-Metriken: Die physikalische Zeit-Transformation
---------------------------------------------------------
Nachdem die Regale als Knoten existieren, werden sie durch Laufwege (Kanten E) verbunden. Hier muss eine fundamentale physikalische Diskrepanz der Informatik gelöst werden: 

Die JSON-Konfigurationsdatei liefert logischerweise geometrische Abstände in Metern. Unsere späteren Routing-Algorithmen (wie Held-Karp oder Dijkstra) und Machine-Learning-Modelle optimieren jedoch **Laufzeit in Sekunden**. Ein Graph, der Meter als Kantengewicht nutzt, kann später nicht mit KI-Stau-Vorhersagen (die in Straf-Sekunden gemessen werden) verrechnet werden – das System würde physikalisch Äpfel mit Birnen addieren und mathematisch invalide Ergebnisse produzieren.

Die Architektur führt daher zwingend beim Boot-Vorgang eine physikalische Datentransformation durch. Die geometrische Distanz wird via Pythagoras berechnet und durch die durchschnittliche menschliche Schrittgeschwindigkeit (hier wissenschaftlich standardisiert auf 1.4 m/s) dividiert. 

**Das Interaktionszeit-Paradoxon:** Ein Weg im Supermarkt besteht nicht nur aus Laufen. Wenn ein Kunde vor einem Regal ankommt, benötigt er Zeit, um das Produkt zu suchen und in den Wagen zu legen. Ein naiver Ansatz würde diese Zeit (z.B. 8 Sekunden) direkt auf die Kante addieren. Das führt jedoch zu einem Fehler in der Graphentheorie: Läuft der Kunde an 10 Regalen nur *vorbei*, um zur Kasse zu kommen, würde der Algorithmus 10 Mal die Suchzeit berechnen, obwohl der Kunde dort nichts kauft. 
Die Kanten in dieser Topologie repräsentieren daher *ausschließlich* reine Transitzeiten. Die Interaktionszeit (die das Greifen auf der Z-Achse abstrahiert) wird erst im Operations-Research-Modul dynamisch als Knotengewicht addiert.

.. code-block:: python

   import math

   class StoreTopology:
       # ... [Vorherige Methoden] ...

       def _parse_and_build(self, filepath: str) -> None:
           # ... [Knoten-Injektion wie oben] ...
           
           WALKING_SPEED_MPS = 1.4 # Durchschnittliche Schrittgeschwindigkeit (Meter/Sekunde)
           
           # 2. Kanten injizieren und Geometrie in die Dimension "Zeit" transformieren
           for edge_data in layout.get("edges", []):
               u = edge_data.get("source")
               v = edge_data.get("target")
               
               # Topologischer Schutz vor verwaisten Kanten (Knoten existiert nicht)
               if not u or not v or u not in self.G or v not in self.G:
                   continue 
                   
               edge_type = edge_data.get("type", "bidirectional")
               
               coord_u = self.G.nodes[u]["coordinates"]
               coord_v = self.G.nodes[v]["coordinates"]
               
               # Physikalische Distanz s via Pythagoras: sqrt((x2-x1)^2 + (y2-y1)^2)
               dist_meters = math.hypot(coord_v[0] - coord_u[0], coord_v[1] - coord_u[1])
               
               # Transformation in reine Zeitmetrik: t_transit = s / v
               base_time_seconds = dist_meters / WALKING_SPEED_MPS
               
               # Fall A: Standard-Gänge (Laufwege in beide Richtungen erlaubt)
               if edge_type == "bidirectional":
                   self.G.add_edge(u, v, base_time=base_time_seconds, current_weight=base_time_seconds)
                   self.G.add_edge(v, u, base_time=base_time_seconds, current_weight=base_time_seconds)
                   
               # Fall B: Physische Einbahnstraßen (Die Kassenzone)
               # Es wird NUR die gerichtete Kante von u nach v programmiert. 
               elif edge_type == "one_way":
                   self.G.add_edge(u, v, base_time=base_time_seconds, current_weight=base_time_seconds)

**Der architektonische Geniestreich (Gerichtete Graphen):** Der massive Vorteil der initialen Wahl eines gerichteten Graphen (``nx.DiGraph``) entfaltet sich exakt im Fall B dieses Parsers. Durch das Weglassen der Rück-Kante von der Bezahlstation zurück in den Laden modelliert das System eine unüberwindbare Einbahnstraße. Der spätere Routenalgorithmus kann den Kunden problemlos nach draußen navigieren, aber die Mathematik des Graphen verbietet es ihm rigoros, den Kunden gegen die Laufrichtung wieder in den Laden zu routen, da in dieser Richtung topologisch absolutes Vakuum herrscht.

5. Spatial Indexing: Die Übersetzung von Bildschirm-Klicks
----------------------------------------------------------
Die Geometrie des Graphen ist nun intakt im RAM. Wenn ein Nutzer jedoch im Frontend auf dem Tablet-Display auf einen Punkt der digitalen Karte tippt, weiß das System nicht, welches Regal dort physisch steht. Es erhält lediglich eine nackte Fließkomma-Koordinate (z. B. X=12.45, Y=44.91). Das Backend muss das Spatial Mapping durchführen: *Welcher Graphen-Knoten liegt dieser XY-Koordinate mathematisch am nächsten?*

Eine lineare Suche (ein For-Loop über alle 5000 Knoten, der für jeden Knoten den Pythagoras ausrechnet) würde bei jedem Klick eine Suchzeit von O(V) erfordern. Das würde den Server bei parallelen Kunden im Markt extrem ausbremsen. 

5.1 Der K-d Baum (k-dimensional tree)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Um diese Suche in Sekundenbruchteilen zu ermöglichen, instanziiert das System parallel zum Graphen einen **K-d Baum**. Ein K-d Baum ist eine binäre Raum-Suchstruktur. Das "k" steht für die Anzahl der Dimensionen (hier k=2 für X und Y). 

Der Baum halbiert den Supermarkt algorithmisch durch Hyperebenen. Er zieht eine Linie in der Mitte des Raumes auf der X-Achse und fragt: "Liegt der Klick des Nutzers links oder rechts?". Liegt er links, wird die gesamte rechte Supermarkt-Hälfte verworfen. Im nächsten Schritt zieht er eine Linie auf der Y-Achse durch die verbliebene linke Hälfte und fragt: "Oben oder unten?". Dies iteriert der Baum durch abwechselnde Achsen, bis das exakte Regal isoliert ist. 

Der initiale Aufbau kostet O(V log V) Zeit beim Server-Start. Die eigentliche Suche nach dem Klick im laufenden Betrieb kollabiert dadurch jedoch auf rasante O(log V). Anstatt 5000 Regale zu prüfen, findet der Baum das nächstgelegene Regal in nur maximal 13 binären Rechenschritten.

5.2 Der Edge-Case der Boundary Protection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Was passiert, wenn der Kunde auf dem Tablet abrutscht und versehentlich auf den Parkplatz außerhalb des Supermarktes tippt? Der K-d Baum sucht rein euklidisch und würde stumpf das nächstgelegene Regal an der Außenwand (im Gebäude) zurückgeben, was zu massiver Verwirrung führt. Das System implementiert daher einen harten Suchradius (Threshold), um fehlerhafte Klicks außerhalb der Graphen-Grenzen abzufangen.

.. code-block:: python

   from scipy.spatial import cKDTree
   import numpy as np

   class StoreTopology:
       # ... [Vorherige Methoden] ...

       def _build_spatial_index(self) -> None:
           """
           Baut den k-d Baum zur euklidischen Raumsuche.
           Nutzt explizit cKDTree (die C-Implementierung von SciPy), um das Python 
           GIL (Global Interpreter Lock) zu umgehen und maximale CPU-Geschwindigkeit 
           in der zugrundeliegenden C-Bibliothek zu garantieren.
           """
           self.node_ids = list(self.G.nodes())
           coordinates = [self.G.nodes[n]["coordinates"] for n in self.node_ids]
           
           # Kompiliert den Suchbaum als hochperformante C-Struktur in den Arbeitsspeicher
           self.kd_tree = cKDTree(np.array(coordinates))

       def get_nearest_node(self, x: float, y: float, max_radius: float = 3.0) -> str:
           """ 
           Mappt einen Screen-Tap des Nutzers in O(log V) auf die echte Graphen-ID. 
           max_radius (3 Meter) schützt vor versehentlichen Out-of-Bounds Klicks.
           """
           # query() liefert die euklidische Distanz und den Index des nächsten Knotens
           distance, index = self.kd_tree.query([x, y])
           
           if distance > max_radius:
               # Graceful Degradation: Klick liegt im "Nirgendwo"
               raise ValueError(f"Koordinate ({x}, {y}) ist zu weit vom Inventar entfernt.")
               
           return self.node_ids[index]

6. Graphentheoretische Resilienz: Das Connectivity-Paradoxon
------------------------------------------------------------
Ein extrem gefährlicher Fehler bei der Graphen-Konstruktion ist die Entstehung von "Inseln" (z. B. ein Regal, das durch einen Tippfehler im JSON mit keiner Kante verbunden ist). Sucht ein Kunde ein Produkt in diesem Regal, würde der Dijkstra-Routing-Algorithmus endlos den Graphen absuchen, um einen Weg zu finden. Da keiner existiert, würde der Suchraum im RAM überlaufen und den Webserver zum Absturz bringen.

Um dies abzuwenden, führt die Klasse direkt nach dem Boot-Vorgang das **Fail-Fast-Prinzip** aus: Der Server verweigert den Port-Start sofort, falls die Physik des Graphen verletzt ist.

*Der Beweis im Kolloquium (Strong vs. Weak Connectivity):* In der reinen Mathematik prüft man Graphen oft auf starke Zusammenhänge (Strongly Connected), was bedeutet, dass von jedem Knoten jeder andere Knoten auf geradem oder indirektem Weg erreichbar sein muss. 
Warum nutzt unsere Architektur hier zwingend schwache Zusammenhänge (Weakly Connected)? 
Die Antwort ist das Herzstück des Modells: Weil wir in Phase 4 Einbahnstraßen für die Kassen definiert haben! Aus einem stark zusammenhängenden Graphen müsste man vom Kassen-Ausgang wieder zurück zum Eingang in den Markt laufen können. Da diese Kante physikalisch verboten wurde, ist unser Supermarkt-Graph naturgemäß niemals stark zusammenhängend. Die Prüfung auf "schwache Zusammenhänge" (welche die Kantengerichtungen bei der Prüfung temporär ignoriert) ist die einzige mathematisch korrekte Methode, um isolierte Insel-Regale zu identifizieren, ohne legale Einbahnstraßen fälschlicherweise als Error zu markieren.

.. code-block:: python

   class GraphTopologyError(Exception): 
       pass

   class StoreTopology:
       # ... [Vorherige Methoden] ...

       def _validate_topology(self) -> None:
           """ Graphentheoretischer Safety-Check vor der Server-Freigabe. """
           
           # 1. Isolierungs-Prüfung: Schwacher Zusammenhang wegen Kassen-Einbahnstraßen!
           if not nx.is_weakly_connected(self.G):
               raise GraphTopologyError("BOOT FAILURE: Graph enthält isolierte Regale (Inseln)!")
               
           # 2. Dijkstra Precondition Check: Existieren negative Distanzen?
           # Negative Kantengewichte würden den Dijkstra-Algorithmus später 
           # in eine Endlosschleife treiben, da er "negative" Kosten endlos weiter minimieren will.
           for u, v, weight in self.G.edges(data='current_weight'):
               if weight < 0:
                   raise GraphTopologyError(f"BOOT FAILURE: Negative Kante bei {u}->{v} entdeckt!")

**Fazit & Systemübergabe:** Dieses Modul hat den statischen Bauplan des Supermarktes in ein robustes, zeitbasiertes und mathematisch fehlerfreies Fundament verwandelt. Der Graph ist durch Adjazenzlisten speichereffizient aufgebaut, durch K-d Trees logarithmisch durchsuchbar, physikalisch von Metern in reale Laufzeit umgerechnet und durch asymmetrische Kanten logisch restriktiert. 

Das einzige verbleibende physische Problem am Ende dieses Prozesses: **Die Regalknoten (Flexible Zones) sind noch komplett leer.** Genau an dieser Schnittstelle übernimmt die **Data-Engineering-Pipeline** (Kapitel 4), um diese leeren Raum-Zonen vollautomatisch mit Produkten zu befüllen.