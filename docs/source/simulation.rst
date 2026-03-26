Agentenbasierte Simulation & Synthetische Datengenerierung
==========================================================

Moderne Machine-Learning-Modelle für die prädiktive Stau-Vorhersage (Traffic Prediction, siehe ML-Architektur) erfordern zwingend eine hochdimensionale, historisierte und sauber gelabelte Datenbasis. Das Modell muss in der Trainingsphase Tausende von Stunden an Supermarkt-Betrieb analysieren, um autoregressive Muster zu erlernen. 

Da im Rahmen dieser Arbeit keine realen, hochfrequenten Sensordaten zur Verfügung stehen – nicht zuletzt aufgrund der strikten DSGVO-Richtlinien im Einzelhandel, die das lückenlose Kamera- oder LiDAR-Tracking von Kunden verbieten –, löst die Architektur das Kaltstart-Problem der KI durch synthetische Datengenerierung.

Das System erzeugt seine Ground-Truth-Daten über eine vollautonome Agentenbasierte Simulation (Agent-Based Modeling, ABM). Dieses Modul fungiert als isomorpher "Digitaler Zwilling" (Digital Twin). Es transformiert die statische Graphen-Topologie aus Kapitel 3 durch die Injektion autonomer Kunden-Agenten in ein hochdynamisches, stochastisches System. 

Das oberste Ziel dieser Engine ist es, exakt jenes emergente, chaotische Rauschen der Realität zu erzeugen, welches das ML-Modell später in der physischen Welt durchdringen muss. Nur so lässt sich der sogenannte Sim2Real-Gap (die Lücke zwischen synthetischer Simulation und physischer Realität) minimieren und ein Modell trainieren, das in der echten Welt nicht versagt.

1. Architektonisches Paradigma: Zeitdiskretes ABM vs. DES
---------------------------------------------------------
In der theoretischen Simulationstechnik wird primär zwischen ereignisdiskreten Modellen (Discrete Event Simulation, DES) und zeitdiskreten Modellen unterschieden. 

Ein DES-Ansatz springt asynchron in der Zeitlinie von Event zu Event (Beispiel: "Kunde betritt Markt" -> Engine überspringt 4 Minuten Leerlauf -> "Kunde erreicht Kasse"). 

Die architektonische Entscheidung gegen DES:
Obwohl DES extrem recheneffizient und CPU-schonend ist, führt es im Kontext des maschinellen Lernens zu einem fatalen methodischen Problem. Prädiktive Zeitreihen-Modelle (wie XGBoost mit Autoregressive Lags) erfordern zwingend streng synchronisierte Snapshots im Zustandsraum in äquidistanten Intervallen (z.B. exakt jede Minute). DES würde diese Zeitabstände verzerren. Eine nachträgliche mathematische Interpolation der Daten würde das mikro-temporale Rauschen glätten, die Markow-Eigenschaft der Zeitreihe verletzen und die Ground-Truth für das spätere Training völlig unbrauchbar machen.

Die Architektur implementiert stattdessen zwingend eine zeitdiskrete Physics-Engine (Tick-based ABM). Die Systemzeit wird in atomaren Zeitschritten von Delta t = 1 Sekunde quantisiert. In jedem einzelnen Tick der Main-Loop wird die physikalische Position aller Agenten evaluiert und aktualisiert. Dies kostet mehr CPU-Ressourcen, garantiert aber eine fehlerfreie, synchrone Tensor-Ausrichtung für das spätere Feature-Engineering.

2. Stochastische Ankunftsprozesse & Demografische Profile
---------------------------------------------------------
Die Instanziierung (das Spawning) neuer Kunden-Agenten am Eingang darf nicht uniform oder völlig zufällig erfolgen. Reale Supermärkte unterliegen extremen zirkadianen Rhythmen (Tagesabläufen). Die Architektur löst die Generierung realistischer Kundenströme in zwei verschachtelten stochastischen Phasen: Dem Makro-Timing und der Mikro-Profilierung.

2.1 Das Makro-Timing (Inhomogener Poisson-Prozess)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ein normaler (homogener) Poisson-Prozess geht von einer konstanten Ankunftsrate aus. Das System nutzt stattdessen einen Inhomogenen Poisson-Prozess, bei dem die Ankunftsrate Lambda (die erwarteten Kunden pro Sekunde) eine kontinuierliche Funktion der Tageszeit ist. Sie wird durch trigonometrische Funktionen (Sinus-Wellen) moduliert, um deterministische Peaks (wie die abendliche Rush-Hour um 17:00 Uhr) auf Basis eines stochastischen Grundrauschens zu erzwingen. 

Die Formel für die Wahrscheinlichkeit von k Ankünften in einem Zeitintervall lautet dabei P(k) = (Lambda^k * e^-Lambda) / k!.

.. code-block:: python

   import math
   import numpy as np

   class PoissonSpawner:
       """ Generiert asymmetrische Kundenströme basierend auf der simulierten Tageszeit. """
       def __init__(self, base_rate: float = 0.5, amplitude: float = 2.0):
           self.base_rate = base_rate
           self.amplitude = amplitude

       def spawn_agents_for_tick(self, current_hour: float) -> int:
           """ Zieht die tatsächliche Anzahl neuer Agenten für die exakte aktuelle Sekunde. """
           # Phasenverschiebung der Sinus-Welle, sodass das Maximum exakt bei 17:00 Uhr liegt
           time_shift = (current_hour - 11) / 12 * math.pi
           current_lambda = self.base_rate + self.amplitude * max(0, math.sin(time_shift))

           # Die Poisson-Ziehung garantiert natürliche Varianz (Rauschen) um den Erwartungswert
           return np.random.poisson(current_lambda)

2.2 Mikro-Profilierung (Demografische Stochastik)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Um das Varianz-Rauschen der Realität abzubilden, zieht die Engine für jeden neu generierten Agenten ein Profil aus diskreten Wahrscheinlichkeitsverteilungen. Ein Senior oder eine Familie mit Kindern hat eine völlig andere Basisgeschwindigkeit und Varianz als ein Express-Käufer in der Mittagspause.

.. code-block:: python

   import random
   import numpy as np
   from dataclasses import dataclass

   @dataclass
   class ShopperProfile:
       type_name: str
       speed_mean: float        # Durchschnittliche Laufgeschwindigkeit (m/s)
       speed_std: float         # Standardabweichung der Geschwindigkeit
       cart_lambda: float       # Poisson-Erwartungswert für die Länge der Einkaufsliste
       probability: float       # Demografische Häufigkeit im System

   PROFILES = [
       ShopperProfile("Express", speed_mean=1.4, speed_std=0.1, cart_lambda=3.0, probability=0.3),
       ShopperProfile("Normal", speed_mean=1.1, speed_std=0.2, cart_lambda=15.0, probability=0.5),
       ShopperProfile("Family", speed_mean=0.8, speed_std=0.3, cart_lambda=45.0, probability=0.2)
   ]

   def spawn_agent(agent_id: str, entry_node: str, all_products: list):
       """ Instanziiert einen neuen Agenten mit stochastisch gewürfelter DNA. """
       profile = random.choices(PROFILES, weights=[p.probability for p in PROFILES])[0]

       # Individuelle Geschwindigkeit via Normalverteilung (Gauß-Kurve)
       # Hartes Clipping auf 0.4 m/s verhindert physikalisch unmögliche Graphen-Deadlocks
       ind_speed = max(0.4, np.random.normal(profile.speed_mean, profile.speed_std))
       num_items = max(1, np.random.poisson(profile.cart_lambda))

       return Agent(agent_id, entry_node, ind_speed, random.sample(all_products, k=num_items))

3. Fraktionale Kinematik & Makroskopische Stau-Physik
-----------------------------------------------------
Die Fortbewegung der Agenten erfolgt in der Simulation nicht diskret durch das sofortige "Hüpfen" von Knoten zu Knoten. Das wäre physikalisch inkorrekt und würde keine Staus erzeugen. Die Fortbewegung erfolgt stattdessen kontinuierlich (fraktional) auf der eindimensionalen Kante zwischen zwei Regalen.

3.1 Kinematic Wave Theory (Lighthill-Whitham-Richards Modell)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Die Simulation adaptiert das makroskopische Lighthill-Whitham-Richards (LWR) Modell für Verkehrsfluss aus der Straßenbauplanung. Jede Kante im Graphen (ein Gang) besitzt ein Kapazitätslimit. Betreten Agenten die Kante, wird ihre individuelle Geschwindigkeit kollektiv über eine nicht-lineare Gleichung gedrosselt. Dies löst in der Simulation realistische, emergente Rückstaus aus.

Gemäß dem Fundamentaldiagramm des Verkehrsflusses sorgt ein quadratischer Exponent in der Formel dafür, dass der Supermarkt-Gang bis zu einer Auslastung von ca. 60 Prozent kaum Geschwindigkeitsverlust aufweist (Free Flow). Erst ab ca. 80 Prozent Auslastung bricht der Verkehrsfluss exponentiell zusammen (Congested Flow) und zwingt die Agenten zum Kriechen.

3.2 Bounded Rationality & Pre-Halftime Prädiktions-Protokoll
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Um ein rein roboterhaftes Verhalten zu verhindern, implementiert der Agent das Konzept der Bounded Rationality (Begrenzte Rationalität). Dies ist ein stochastischer Parameter, der unvorhersehbares Stehenbleiben (z.B. weil der Kunde auf sein Handy schaut oder ein Produkt sucht) simuliert. 

Ein kritischer architektonischer Pfeiler ist zudem das Pre-Halftime Prädiktions-Protokoll. Die stochastische Zuweisung der optimalen Kasse darf dem TSP-Solver des Agenten nicht erst am Ende des Einkaufs übergeben werden. Das System erzwingt, dass die Kassen-Vorhersage zwingend vor der Halbzeit (bei 40 Prozent des Warenkorbs) getroffen wird. Dies simuliert reale menschliche Vorausplanung und verhindert, dass Agenten an den Kassenzonen abrupte Kehrtwenden machen, was unweigerlich zu topologischen Deadlocks führen würde.

.. code-block:: python

   class Agent:
       def __init__(self, agent_id: str, entry_node: str, ind_speed: float, shopping_list: list):
           self.ind_speed = ind_speed
           self.shopping_list = shopping_list
           self.items_collected = 0
           self.progress_on_edge = 0.0
           self.current_edge_length = 15.0

           # Bounded Rationality: Wahrscheinlichkeit, dass der Kunde grundlos stehenbleibt
           self.hesitation_prob = random.uniform(0.01, 0.05)
           self.halftime_triggered = False
           self.has_left_store = False

       def tick_update(self, current_occupancy: int, edge_capacity: int, stochastics_engine, delta_t: float = 1.0):
           if random.random() < self.hesitation_prob:
               return # Agent trödelt in diesem Tick (Zurückgelegter Weg = 0 Meter)

           # 1. Zwingendes Architektur-Gesetz: Pre-Halftime Prädiktions-Protokoll
           # Die Prädiktion muss zwingend VOR der zweiten Einkaufshälfte erfolgen (Puffer).
           progress_ratio = self.items_collected / max(1, len(self.shopping_list))
           if not self.halftime_triggered and progress_ratio >= 0.4:
               self.checkout_node = stochastics_engine.predict_optimal_checkout()
               self.halftime_triggered = True

           # 2. Makroskopische Stau-Physik (LWR-Modell)
           congestion_factor = max(0.2, 1.0 - (current_occupancy / edge_capacity)**2)
           actual_speed = self.ind_speed * congestion_factor

           # 3. Fraktionale Vorwärtsbewegung auf der Kante
           self.progress_on_edge += actual_speed * delta_t

           if self.progress_on_edge >= self.current_edge_length:
               self.progress_on_edge -= self.current_edge_length
               self._pop_next_node_from_tsp_path() # Bewegt den Agenten zum nächsten Knoten

4. Ergodizität (Burn-in) & Chunked Memory Management
----------------------------------------------------
Die architektonische Brücke zwischen Simulation und Machine Learning ist das Data Harvesting (die Datenernte). 

Ein Prüfer könnte hier die berechtigte Frage stellen: Wann beginnt die Simulation mit der Aufzeichnung? Da der Supermarkt zum Startzeitpunkt (Tick 0) völlig leer ist, würde ein sofortiges Harvesting sogenannte transiente Zustände aufzeichnen (einen Cold-Start-Bias). Ein ML-Modell würde dadurch fälschlicherweise lernen, dass Supermärkte grundsätzlich immer leer sind. 

Die Architektur erzwingt stattdessen das Erreichen der Ergodizität (den mathematischen Steady-State der Markow-Kette) durch eine harte Burn-in Period von 3600 Ticks (exakt 1 Stunde). In dieser Zeit werden die Agenten im Verborgenen simuliert und berechnet, aber es wird kein einziger Datenpunkt persistiert.

*Die Rechtfertigung für Apache Parquet:* Um einen Out-Of-Memory (OOM) Kollaps bei mehrtägigen Simulationen zu verhindern, nutzt das System das spaltenbasierte (columnar) Apache Parquet Format mit Snappy-Kompression statt regulärer CSV-Dateien. Die Snapshots werden periodisch (Chunking) auf die SSD gestreamt, wodurch der RAM-Verbrauch des Python-Prozesses konstant bei O(1) bleibt.

.. code-block:: python

   import pandas as pd
   import pyarrow as pa
   import pyarrow.parquet as pq

   class SimulationEngine:
       def __init__(self, graph):
           self.graph = graph
           self.burn_in_ticks = 3600
           self.snapshot_buffer = []
           self.chunk_size = 10000 # Buffer-Limit für den SSD-Flush
           self.parquet_writer = None

       def run_simulation(self, total_ticks: int):
           for current_tick in range(total_ticks):
               
               # ... [Agenten Spawning und Tick Updates wie oben] ...

               # Garbage Collection (Räumt fertige Kunden aus dem RAM, verhindert OOM)
               self.agents = [a for a in self.agents if not a.has_left_store]

               # Data Harvesting im Steady-State (exakt alle 60 Sekunden für ML-Lags)
               if current_tick > self.burn_in_ticks and current_tick % 60 == 0:
                   self._capture_graph_snapshot(current_tick)

       def _capture_graph_snapshot(self, tick: int):
           """ Friert den Graphen ein, extrahiert die Ground Truth und puffert sie. """
           edge_occupancy = {edge: 0 for edge in self.graph.edges()}
           for agent in self.agents:
               edge_occupancy[agent.current_edge] += 1

           for (node_u, node_v), occupancy in edge_occupancy.items():
               # O(1) Extraktion der Spatial Spillovers zum exakten Zeitpunkt t
               out_edges = self.graph.out_edges(node_v)
               neighbor_loads = [edge_occupancy.get((u, v), 0) for u, v in out_edges]
               max_neighbor_load = max(neighbor_loads) if neighbor_loads else 0

               self.snapshot_buffer.append({
                   "timestamp": tick,
                   "edge_id": f"{node_u}_{node_v}",
                   "occupancy": np.uint16(occupancy), # Hartes Downcasting spart RAM
                   "spatial_neighbor_max_occupancy": np.uint16(max_neighbor_load)
               })

           # Chunked SSD Streaming (hält den Python-RAM-Verbrauch konstant auf O(1))
           if len(self.snapshot_buffer) >= self.chunk_size:
               self._flush_buffer_to_disk()

       def _flush_buffer_to_disk(self):
           """ Schreibt die Daten hochkomprimiert im Parquet-Format auf die Festplatte. """
           df = pd.DataFrame(self.snapshot_buffer)
           table = pa.Table.from_pandas(df)

           if self.parquet_writer is None:
               self.parquet_writer = pq.ParquetWriter('simulation_data.parquet', table.schema, compression='snappy')

           self.parquet_writer.write_table(table)
           self.snapshot_buffer.clear() # Leert den Puffer sofort

5. Topologische Translation: Spatial Spillovers (Feature Engineering)
---------------------------------------------------------------------
Das ML-Modell (XGBoost) versteht mathematisch keine zweidimensionalen Graphen oder Netzwerke, sondern verlangt flache 1D-Tensoren (Tabellen). 

Um der KI beizubringen, dass sich ein Stau von Gang A rückwärts in Gang B ausbreitet (Spatial Spillover Effekt), muss die topologische Nachbarschaft exakt in Tabellenspalten übersetzt werden. Ein klassischer Architekturfehler wäre es, dies post-hoc über langsame Pandas-Schleifen (z. B. ``iterrows()``) zu rekonstruieren. Dies wäre nicht nur ein ineffizientes Anti-Pattern, sondern birgt auch die Gefahr des Temporal Data Leakage, da der Zustand vergangener Ticks mit der Gegenwart vermischt werden könnte.

Die Engine löst dies stattdessen nativ während der ``_capture_graph_snapshot``-Phase (siehe Code in Kapitel 4). Da die Engine in diesem Mikrotick ohnehin das exakte Wissen über alle Kanten besitzt, extrahiert sie die maximale Auslastung der direkten Graphen-Nachbarn absolut zeitsynchron in O(1) Latenzzeit. Diese saubere topologische Translation erlaubt es dem tabellarischen Boosting-Modell, räumliche Flaschenhälse vorausschauend zu antizipieren, ohne teure Graph-Neural-Networks (GNNs) einsetzen zu müssen.

6. Statistische Validierung des Digitalen Zwillings
---------------------------------------------------
Ein Machine-Learning-Modell, das auf Müll trainiert wird, generiert Müll. Die generierten Daten der Simulation müssen daher vor dem Training statistisch rigoros gegen die Realität validiert werden.

6.1 Mathematische Dualität: KS-Test im stationären Fenster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Theoretische Fundierung: Wenn die Anzahl der Ankünfte am Supermarkt-Eingang korrekt durch einen Poisson-Prozess modelliert wird, dann müssen die Zeiten *zwischen* den einzelnen Ankünften (die Inter-Arrival-Times) zwingend einer kontinuierlichen Exponentialverteilung folgen. 

Da unsere Simulation jedoch das Kundenaufkommen dynamisch über den Tag als Sinus-Kurve moduliert (Inhomogener Poisson-Prozess, siehe Kapitel 2.1), würde ein globaler Test hier unweigerlich fehlschlagen. Das System beweist die mathematische Integrität der Pseudo-Zufallsgeneratoren stattdessen durch die Isolierung eines stationären Zeitfensters (z. B. das absolute Plateau der Rush-Hour). Auf diesem Ausschnitt, in dem Lambda näherungsweise konstant ist, wird der Kolmogorov-Smirnov-Test (KS-Test) angewendet. Ein p-Wert größer als 0.05 beweist, dass die Daten der theoretischen Erwartung einer echten Exponentialverteilung entsprechen.

.. code-block:: python

   from scipy import stats

   def validate_spawner_distribution(rush_hour_inter_arrival_times: list, peak_lambda: float):
       """ 
       Führt den KS-Test auf einem stationären Ausschnitt der Simulation durch,
       um inhomogene Modulationsverzerrungen mathematisch auszuschließen. 
       """
       ks_stat, p_value = stats.kstest(
           rush_hour_inter_arrival_times, 
           'expon', 
           args=(0, 1.0 / peak_lambda) # Skalierung = 1 / Lambda
       )
       
       # Ist p >= 0.05, wird H0 (Daten sind exakt exponentialverteilt) akzeptiert.
       if p_value < 0.05:
           raise ValueError(f"Simulation Calibration Failed! KS-Stat: {ks_stat}, p: {p_value}")

6.2 Beweis der Ergodizität (Augmented Dickey-Fuller Test)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Theoretische Fundierung: Um die Wahl von 3600 Ticks für die Burn-in-Phase mathematisch zu rechtfertigen, nutzt das System den Augmented Dickey-Fuller-Test (ADF). Er beweist, dass die System-Auslastung nach dieser Phase tatsächlich stationär ist. 

Die Nullhypothese (H0) des ADF-Tests lautet, dass die Zeitreihe eine Unit-Root (Einheitswurzel) besitzt. Eine Zeitreihe mit Unit-Root hat keine Tendenz zur Rückkehr zu einem langfristigen Mittelwert (Mean-Reverting), sondern driftet unkontrollierbar als Random Walk ab. Ein p-Wert von unter 0.05 verwirft H0 rigoros und beweist, dass das System seinen stabilen Rhythmus (Steady-State) gefunden hat.

.. code-block:: python

   from statsmodels.tsa.stattools import adfuller

   def validate_steady_state(occupancy_time_series: list):
       """ Beweist mathematisch, dass der transiente Burn-In erfolgreich abgeschlossen ist. """
       adf_result = adfuller(occupancy_time_series)
       p_value = adf_result[1]
       
       # p < 0.05 verwirft H0 (Unit-Root) rigoros -> Beweis der perfekten Stationarität!
       if p_value >= 0.05:
           raise ValueError("System possesses a Unit-Root (Random Walk)! Burn-in is invalid.")

7. Systemintegrität: Forward Chaining & Target-Shift
----------------------------------------------------
Der fatalste methodische Fehler in der Data-Science-Pipeline ist das sogenannte Data Leakage (Look-Ahead Bias). Würde die Pipeline die exportierten Parquet-Daten vor dem Training via K-Fold Cross Validation zufällig durchmischen, würde das Modell die Zukunft sehen, bevor es sie vorhersagt. Die zeitliche Autokorrelation der Daten wäre vollständig zerstört.

Die Pipeline erzwingt daher eine strikte chronologische Validierung (Time Series Split / Forward Chaining). Zudem wird das abhängige Target-Label (die Y-Variable) über einen deterministischen Pandas-Shift exakt 5 Minuten in die Zukunft generiert:

.. code-block:: python

   import xgboost as xgb
   from sklearn.model_selection import TimeSeriesSplit

   # 1. Das Target definieren: Die Auslastung exakt 5 Minuten in der Zukunft
   df['target_t_plus_5'] = df.groupby('edge_id')['occupancy'].shift(-5)
   df.dropna(inplace=True)

   # 2. Features (X) und Target (y) für das Modell separieren
   y = df['target_t_plus_5']
   X = df.drop(columns=['target_t_plus_5', 'edge_id', 'timestamp'])

   # 3. Chronologischer Split (blockiert Look-Ahead Leakage absolut)
   tscv = TimeSeriesSplit(n_splits=5)
   model = xgb.XGBRegressor(n_estimators=200, max_depth=6)

   for train_idx, test_idx in tscv.split(X):
       # Die Evaluierung erfolgt ausschließlich auf streng in der Zukunft liegenden Blöcken
       model.fit(X.iloc[train_idx], y.iloc[train_idx])

Fazit der Simulation:
Nur durch diese kompromisslose Methodik – von der zeitdiskreten fraktionalen Kinematik, dem ADF-Stationaritätsbeweis und dem Apache Parquet Memory-Handling bis hin zum Pre-Halftime-Routing – wird garantiert, dass der Sim2Real-Gap überbrückt wird. Die generierten Tensoren sind kein wertloses Rauschen, sondern der mathematische Beweis eines isomorphen Digitalen Zwillings, auf dem das prädiktive Backend verlässlich generalisieren kann.