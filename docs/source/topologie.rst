Supermarkt-Topologie & Graphen-Mapping
======================================

Bevor algorithmische Optimierungen für Laufwege berechnet oder Produkte durch die ETL-Pipeline zugewiesen werden können, muss die physische Realität des JMU Smart Supermarkets in ein maschinenlesbares, mathematisches Format überführt werden. 

Anstelle eines simplen zweidimensionalen Koordinatensystems (wie es in klassischen A*-Pfadfindungs-Algorithmen genutzt wird), modelliert das System den Supermarkt als **gerichteten, kanten-gewichteten Attribut-Graphen**. Diese Entscheidung ist architektonisch zwingend: Ein Supermarkt weist durch physische Barrieren (Regalreihen) strikte topologische Restriktionen auf, die eine freie 2D-Bewegung (z. B. diagonales Laufen durch Regale hindurch) physisch unmöglich machen.

1. Das Topologische Mapping (Vom Grundriss zum Netzwerk)
--------------------------------------------------------

Der erste Schritt der Systeminitialisierung ist das **Topologische Mapping**. Hierbei wird der physische Bauplan des Supermarkts in eine Adjazenzstruktur übersetzt, die von der Bibliothek ``NetworkX`` im Backend verarbeitet werden kann.

* **Topologische Diskretisierung:** Die durchgehenden Gänge des Marktes werden in diskrete, logische Abschnitte (Knoten) unterteilt. Steht ein Kunde physisch vor dem Nudelregal, befindet er sich logisch auf dem Knoten ``vD6``.
* **Adjazenz-Definition:** Das System definiert explizit, welche Knoten benachbart sind. Wenn ein Kunde von ``vD6`` nach ``vC6`` laufen kann, existiert eine Kante (Edge) zwischen diesen Knoten. Wenn eine Wand oder ein Regal dazwischensteht, existiert keine Kante.
* **Wegnetz-Initialisierung:** Beim Start der ``model.py`` wird dieser Graph im Arbeitsspeicher instanziiert. Das System kennt nun die exakte Raumstruktur, weiß aber noch nicht, welche Produkte sich an welchen Koordinaten befinden.

2. Die Knoten-Architektur (Vertices)
------------------------------------

Jeder Knoten im Graphen repräsentiert eine begehbare Zone. Im Gegensatz zu klassischen Graphentheorie-Modellen sind die Knoten in unserem System keine leeren Koordinaten, sondern komplexe Daten-Container (Python-Dictionaries), die den aktuellen Systemzustand, das Inventar und Metadaten halten.

Die Klasse ``StoreOntology`` definiert hierbei eine strikte physikalische Trennung der Knoten in drei Kategorien:

2.1 Hardware-gebundene Knoten (Fixed Zones)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Exakt 12 Knoten des Graphen sind statisch an bestimmte Supermarkt-Kategorien gebunden. Diese architektonische Design-Entscheidung reflektiert reale, unveränderliche Limitierungen im stationären Einzelhandel:

* **Kühlketten-Restriktionen:** Produkte wie frisches Fleisch ("Fleischtheke", Knoten ``vA10``) benötigen spezielle Kühl-Hardware und Starkstromanschlüsse.
* **Sicherheitszonen:** Hochpreisiger Alkohol ("Spirituosenschrank", Knoten ``v5``) erfordert verschlossene Vitrinen oder eine erhöhte Kameraüberwachung.
* **Point of Sale (Kassen-Topologie):** Die Knoten ``vW1``, ``vW2`` und ``vW3`` sind dedizierte Hardware-Injections für Spontankäufe (Quengelware wie Batterien oder Kaugummis) und markieren gleichzeitig die stochastischen Endpunkte (M/M/1/K-Warteschlangen) des Laufwegs.

2.2 Das dynamische Container-Konzept (Flexible Zones)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Die verbleibenden 18 Regal-Knoten (z. B. ``vD6``, ``vC3``, ``vB10``) fungieren als leere Allokations-Slots. Zur Initialisierung des Systems weisen sie keine festen Warengruppen auf. Erst im nachgelagerten ETL-Prozess (Data Engineering) werden diese Knoten über das Sainte-Laguë-Verfahren dynamisch mit Produktkategorien befüllt.

**Knoten-Attribute und Kapazitätsgrenzen:**
Um die Physis eines echten Regals abzubilden, besitzt jeder Knoten ein Kapazitätslimit (``MAX_CAPACITY = 6``). Der Graph erlaubt es dem Algorithmus nicht, einen Knoten mit unendlich vielen Produkten zu überladen. Die interne Repräsentation eines Knotens im Speicher hält dabei ein Array an Produkt-Daten:

.. code-block:: python

   # Interne Speicherrepräsentation eines Graphen-Knotens (z.B. vD6)
   node_vD6_state = {
       "node_id": "vD6",
       "capacity_used": 4,
       "max_capacity": 6,
       "stock": [
           {"name": "Spaghetti", "brand": "Barilla", "price": "2.49 €"},
           {"name": "Penne", "brand": "Buitoni", "price": "1.99 €"}
       ]
   }

2.3 Routing-Knoten (Wege-Infrastruktur)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sogenannte Transitknoten (rote Gang-Knoten) enthalten kein Inventar. Sie dienen den Operations-Research-Algorithmen (wie Held-Karp) als zwingende Transitpunkte, um das Wechseln der Regalreihen logisch abzubilden.

3. Kanten-Metriken und Gewichtsfunktionen (Edges & Weights)
-----------------------------------------------------------

Die Kanten des Graphen definieren die zulässigen Laufwege. Im Smart Cart System sind diese Kanten jedoch nicht statisch, sondern hochdynamisch. Die Gewichtsfunktion, welche die "Kosten" (die benötigte Laufzeit) einer Kante zum Zeitpunkt *t* beschreibt, setzt sich mathematisch aus zwei Komponenten zusammen:

.. math::

   W(e, t) = d_{base}(e) + \lambda \cdot P_{traffic}(e, t)

* **Die Baseline:** Die Komponente :math:`d_{base}(e)` beschreibt die nominelle, physikalische Distanz zwischen zwei Regalen. Sie bildet die statische Grundlage für die initiale Lösung des Traveling Salesperson Problems (TSP).
* **Der dynamische Penalty-Term:** Die Komponente :math:`P_{traffic}(e, t)` ist der Punkt, an dem der Graph direkt mit der künstlichen Intelligenz interagiert. Der ``TrafficPredictor`` injiziert die vorhergesagte Stau-Wahrscheinlichkeit als "Strafzeit" auf die Kante. Der Skalar :math:`\lambda` gewichtet dabei, wie stark ein prognostizierter Stau das Routing beeinflussen soll.

4. Die Graph-Synchronisation (Shared State)
-------------------------------------------

Das architektonische Meisterstück dieses topologischen Aufbaus ist die Rolle des Graphen als **Single Source of Truth** für das gesamte Backend. Da der Graph zentral im Arbeitsspeicher (`In-Memory`) gehalten wird, können alle Subsysteme synchron darauf zugreifen:

1. Die **MLOpsEngine (Suche)** fragt den Graphen ab, um zu verifizieren, in welchem Knoten ein klassifiziertes Produkt aktuell lagert.
2. Das **Operations Research Modul** extrahiert in Echtzeit Sub-Graphen aus dieser Topologie, um die Kantengewichte auszulesen und die kürzesten Laufwege zu berechnen.
3. Das **Dashboard (Frontend)** iteriert asynchron über die Attribute der Knoten, um die Regale in der Benutzeroberfläche visuell zu füllen und Staus farblich zu markieren.

Dieser Entwurf garantiert, dass Datengenerierung, Routing und Suche in Millisekunden auf denselben validen Systemzustand zugreifen, ohne dass teure und latenzintensive Datenbankabfragen (SQL/NoSQL) notwendig sind.