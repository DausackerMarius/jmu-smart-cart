Agentenbasierte Simulation & Synthetische Datengenerierung
==========================================================

Moderne Machine-Learning-Modelle für die prädiktive Stau-Vorhersage erfordern eine hochdimensionale, historisierte und sauber gelabelte Datenbasis. Da im Rahmen dieser Arbeit keine realen, lückenlosen Sensordaten zur Verfügung stehen (z. B. aufgrund von DSGVO-Richtlinien im Einzelhandel), löst die Architektur das Kaltstart-Problem durch synthetische Datengenerierung.

Das System erzeugt seine Ground-Truth-Daten über eine vollautonome, agentenbasierte Simulation (Agent-Based Modeling, ABM). Dieses Modul fungiert als Digitaler Zwilling (Digital Twin). Es transformiert die statische Graphen-Topologie durch die Injektion autonomer Kunden-Agenten in ein hochdynamisches System, um das emergente Rauschen der Realität für das spätere XGBoost-Training zu simulieren.

1. Architektonisches Paradigma: Zeitdiskrete Simulation
-------------------------------------------------------
Um für prädiktive Zeitreihen-Modelle streng synchronisierte Snapshots in äquidistanten Intervallen zu generieren, implementiert die Engine ein zeitdiskretes Paradigma (Tick-based ABM). 

Die Systemzeit wird in diskreten Zeitschritten von :math:`\Delta t = 5` Sekunden quantisiert. In jedem Tick der Main-Loop wird die physikalische Position und der Zustand (State Machine) aller Agenten evaluiert. Um das Modell für das Training eines gesamten Jahres (z. B. 2025) speichereffizient zu gestalten, werden die Systemzustände (Edge Loads, Queue-Längen) aggregiert und periodisch in eine CSV-Datei exportiert.

2. Stochastische Ankunftsprozesse & Demografische Profile
---------------------------------------------------------
Die Instanziierung (Spawning) neuer Agenten am Eingang modelliert die komplexen zirkadianen Rhythmen eines echten Supermarkts.

2.1 Makro-Timing: Saisonalität und Inhomogener Poisson-Prozess
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Das System berechnet zunächst ein absolutes Tageslimit an Kunden, welches durch Wochentage, Feiertage und saisonale Faktoren (z. B. das Weihnachtsgeschäft mit einem Faktor von 1.5) moduliert wird. 

Dieses Tagesvolumen wird anschließend auf eine bimodale Stundenkurve (Peaks zur Mittagspause und zum Feierabend) heruntergebrochen. Die tatsächliche, minutengenaue Generierung der Agenten an der Eingangstür erfolgt mathematisch über eine Poisson-Verteilung, um die natürliche Varianz und das Rauschen von Kundenankünften abzubilden.

2.2 Mikro-Profilierung: Die Lognormalverteilung
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Jeder instanziierte Agent erhält ein demografisches Profil ("Express", "Normal", "Family") mit individuellen Parametern für die Laufgeschwindigkeit. 

Ein kritisches Detail der Simulation ist die Generierung der Warenkorbgröße (Anzahl der Produkte). Hierfür nutzt das System zwingend eine Lognormalverteilung (``np.random.lognormal``). Diese Verteilung ist für Einkaufsgrößen mathematisch ideal, da sie keine negativen Werte zulässt und die reale Asymmetrie abbildet: Die meisten Kunden kaufen wenige Artikel, während ein langer "Tail" (Schwanz) der Verteilung die seltenen, aber massiven Wocheneinkäufe repräsentiert.

3. Kinematik & Lokale Dichte-Strafen
------------------------------------
Die Agenten bewegen sich fraktional auf den Kanten des Graphen. Die Routing-Entscheidung der Agenten ist bewusst als "myopic" (kurzsichtig) modelliert: Sie nutzen zur Navigation von Regal zu Regal einen heuristischen Nearest-Neighbor-Ansatz (Greedy-TSP) auf Basis vorberechneter Dijkstra-Distanzen. Dies verhindert, dass die Agenten bereits die perfekte ML-Routenoptimierung nutzen, welche das Modell erst erlernen soll.

Die physikalische Interaktion der Agenten wird über eine lokale Dichte-Strafe (Local Density Penalty) abgebildet. In jedem Tick ermittelt das System die Anzahl der Agenten auf einer Kante. Die Basis-Laufgeschwindigkeit des Agenten wird durch einen linearen Faktor gedrosselt, je mehr Personen sich im selben Gangabschnitt befinden:

.. code-block:: python

   # Berechnung des lokalen Geschwindigkeitsfaktors
   local_speed_factor = max(0.05, 1.0 - (agents_on_edge * Config.LOCAL_DENSITY_PENALTY))
   walk_distance = Config.WALK_SPEED_BASE * local_speed_factor * dt

Dieser Mechanismus erzwingt physikalisch plausible Verzögerungen bei Menschenansammlungen, ohne dass sich Agenten gegenseitig blockieren (Deadlocks).

4. Das Herzstück: Physischer Kassen-Spillover
---------------------------------------------
Ein klassischer Fehler in naiven Simulationen ist die Annahme, dass Kassenknoten eine unendliche Kapazität besitzen (Kunden stapeln sich auf einem Punkt). In der Realität stauen sich Warteschlangen in die Regalgänge zurück und blockieren dort Kunden, die eigentlich nur einkaufen möchten.

Die Simulation löst dieses Problem durch eine dedizierte Spillover-Engine (``_generate_spillover_paths``). 

Jede Kasse besitzt einen definierten "Fluchtweg" rückwärts in das Innere des Marktes (z. B. von Kasse 1 zurück in den Hauptgang). Das System berechnet die Aufnahmekapazität dieser Kanten basierend auf einem Platzbedarf von ca. 1.5 Personen pro Laufmeter. 
Reiht sich ein Agent in die Warteschlange ein, berechnet der Checkout-Manager anhand des aktuellen Index in der Schlange (``queue_index``), auf welcher physischen Kante des Graphen der Agent de facto steht.

.. code-block:: python

   def get_spillover_edge(self, lane, queue_index):
       """
       Ermittelt, in welchem Gang man physikalisch steht, wenn man der 
       X-te Kunde in der Schlange von Kasse Y ist.
       """
       edges = self.spillover_edges[lane]
       current_cap = 0
       for edge_key, cap in edges:
           current_cap += cap
           if queue_index < current_cap:
               return edge_key
       return edges[-1][0]

Dieser Spillover-Effekt ist essenziell für die Generierung der Machine-Learning-Trainingsdaten, da er genau jene Kettenreaktionen und Staus erzeugt, die das prädiktive Routing später auflösen muss.

5. Memory Management: Chunked CSV-Export
----------------------------------------
Da die Simulation die Datenpunkte (Snapshots) für ein gesamtes Kalenderjahr (über 100.000 Ticks) generiert, würde das Halten aller Historien im Arbeitsspeicher unweigerlich zu einem Out-of-Memory-Error (OOM) führen.

Die Architektur implementiert daher ein strenges Buffer-Management. Die Systemzustände (Timestamp, Queue-Längen, Kassenöffnungen und die JSON-repräsentierte Auslastung aller Kanten) werden in einem Array gesammelt. Sobald das konfigurierte Limit (``CSV_BUFFER_SIZE = 1000``) erreicht ist, streamt die Engine den Block hochperformant über den ``csv.writer`` auf die Festplatte und leert den RAM-Speicher. Dies garantiert eine konstante Speicherkomplexität von :math:`\mathcal{O}(1)` während des gesamten Simulationsjahres.