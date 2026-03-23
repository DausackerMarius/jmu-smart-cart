Agentenbasierte Simulation & Synthetische Datengenerierung
==========================================================

Moderne Machine-Learning-Modelle für die prädiktive Stau-Vorhersage (Traffic Prediction, Kapitel 6) erfordern eine hochdimensionale, historisierte und gelabelte Datenbasis, die Tausende von Stunden an Supermarkt-Betrieb abbildet. Da im Rahmen dieses Prototyps keine realen, hochfrequenten Sensordaten (z. B. LiDAR oder Kamera-Tracking) zur Verfügung stehen, löst die Architektur das Kaltstart-Problem der KI durch synthetische Datengenerierung. 

Das System erzeugt seine Ground-Truth-Daten über eine vollautonome **Agentenbasierte Simulation (Agent-Based Modeling, ABM)**. Dieses Modul fungiert als "Digitaler Zwilling" (Digital Twin). Es transformiert die statische Graphen-Topologie durch die Injektion autonomer Entitäten in ein dynamisches, stochastisches System und erzeugt exakt das stochastische Rauschen, welches das ML-Modell später in der Realität prädizieren soll.

1. Architektonisches Paradigma: Zeitdiskretes ABM vs. DES
---------------------------------------------------------
In der theoretischen Simulationstechnik wird zwischen ereignisdiskreten (Discrete Event Simulation, DES) und zeitdiskreten Modellen unterschieden. Ein DES-Ansatz springt asynchron von Event zu Event ("Kunde betritt Markt", "Kunde erreicht Kasse"). Das ist recheneffizient, führt aber zu einem fatalen Problem: Das prädiktive Machine Learning erfordert streng synchronisierte Zeitreihendaten in äquidistanten Intervallen. DES würde hier durch nachträgliche Interpolation das Rauschen glätten und die Ground-Truth verfälschen.

Die Architektur implementiert stattdessen eine **zeitdiskrete Physics-Engine** (Tick-based ABM). Die Systemzeit wird in atomaren Zeitschritten von $\Delta t = 1\text{s}$ quantisiert. In jedem einzelnen Tick der Main-Loop wird die physikalische Position aller Agenten im RAM evaluiert. Dies garantiert eine mathematisch fehlerfreie Tensor-Ausrichtung für das spätere Feature-Engineering.

2. Stochastische Ankunftsprozesse & Demografische Profile
---------------------------------------------------------
Die Instanziierung neuer Agenten darf nicht uniform erfolgen. Die Architektur löst die Generierung realistischer Kundenströme in zwei Phasen: Dem Makro-Timing und der Mikro-Profilierung.

**2.1 Das Makro-Timing (Inhomogener Poisson-Prozess)**
Die zeitabhängige Ankunftsrate $\lambda(t)$ (Erwartungswert der Kunden pro Zeiteinheit) wird durch trigonometrische Funktionen moduliert. Dies erzwingt deterministische Peaks (die abendliche Rush-Hour) auf Basis eines stochastischen Grundrauschens.

$$P(k \text{ Ankünfte in } \Delta t) = \frac{(\lambda(t) \cdot \Delta t)^k e^{-\lambda(t) \cdot \Delta t}}{k!}$$

**2.2 Mikro-Profilierung (Demografische Stochastik)**
Ein Rentner bewegt sich physisch langsamer und kauft andere Mengen als ein Student. Um dieses Varianz-Rauschen abzubilden, zieht die Engine für jeden Agenten ein Profil aus diskreten Wahrscheinlichkeitsverteilungen.

.. code-block:: python

   import numpy as np
   import random
   from dataclasses import dataclass

   @dataclass
   class ShopperProfile:
       type_name: str
       speed_mean: float        # Ø Laufgeschwindigkeit (m/s)
       speed_std: float         # Standardabweichung der Geschwindigkeit
       cart_lambda: float       # Poisson-Erwartungswert für Einkaufslisten-Länge
       probability: float       # Demografische Häufigkeit im System

   # Definition der Agenten-Demografie
   PROFILES = [
       ShopperProfile("Express", speed_mean=1.4, speed_std=0.1, cart_lambda=3.0, probability=0.3),
       ShopperProfile("Normal", speed_mean=1.1, speed_std=0.2, cart_lambda=15.0, probability=0.5),
       ShopperProfile("Family", speed_mean=0.8, speed_std=0.3, cart_lambda=45.0, probability=0.2)
   ]

   def spawn_agent(agent_id: str, entry_node: str, all_products: list):
       """ Erzeugt einen Agenten mit individuellen, verrauschten Parametern. """
       profile = random.choices(PROFILES, weights=[p.probability for p in PROFILES])[0]
       
       # Individuelle Geschwindigkeit via Normalverteilung (Gauß)
       # Limitierung auf 0.4 m/s verhindert physikalisch unmögliche Deadlocks
       ind_speed = max(0.4, np.random.normal(profile.speed_mean, profile.speed_std))
       
       # Einkaufslistengröße via Poisson-Verteilung
       num_items = max(1, np.random.poisson(profile.cart_lambda))
       shopping_list = random.sample(all_products, k=num_items)
       
       return Agent(agent_id, entry_node, ind_speed, shopping_list)

3. Fraktionale Kinematik & Makroskopische Stau-Physik
-----------------------------------------------------
Sobald der Agent existiert, berechnet er seinen Basis-Pfad via TSP-Solver. Die Fortbewegung auf diesem Pfad erfolgt nicht diskret von Knoten zu Knoten, sondern kontinuierlich auf der eindimensionalen Kante (Fraktionale Kinematik).

**3.1 Kinematic Wave Theory & Drosselung:**
Jede Kante $e$ (ein Supermarktgang) besitzt ein physisches Kapazitätslimit $c_{max}(e)$. Betreten Agenten die Kante, wird ihre individuelle Geschwindigkeit $v_{ind}$ kollektiv über eine nicht-lineare Penalty-Gleichung gedrosselt, was emergente Rückstaus auslöst:

$$v_{actual} = v_{ind} \cdot \max\left(v_{min}, 1 - \left(\frac{\text{occupancy}(e)}{c_{max}(e)}\right)^\gamma \right)$$

**3.2 Code-Implementierung der fraktionalen Bewegung:**
Die Simulation berechnet in jedem Tick den exakten Fortschritt in Metern. Überschreitet der akkumulierte Weg die Gesamtlänge der Kante, springt der Agent auf die nächste Kante seines TSP-Pfades.

.. code-block:: python

   class Agent:
       def __init__(self, ...):
           self.progress_on_edge = 0.0 # Zurückgelegte Meter auf der aktuellen Kante
           self.current_edge_length = 15.0 # Beispiel: 15 Meter langer Gang

       def tick_update(self, current_occupancy: int, edge_capacity: int, delta_t: float = 1.0):
           """ Wird 1x pro Sekunde aufgerufen. """
           # 1. Berechne die durch Stau gedrosselte Ist-Geschwindigkeit
           congestion_factor = max(0.2, 1.0 - (current_occupancy / edge_capacity)**2)
           actual_speed = self.ind_speed * congestion_factor
           
           # 2. Fraktionale Vorwärtsbewegung (Weg = Geschwindigkeit * Zeit)
           self.progress_on_edge += actual_speed * delta_t
           
           # 3. Kanten-Übergang prüfen
           if self.progress_on_edge >= self.current_edge_length:
               self.progress_on_edge -= self.current_edge_length
               self._pop_next_node_from_tsp_path()

4. Die Tick-Loop & Eliminierung des Cold-Start-Bias (Burn-in)
-------------------------------------------------------------
Die zentrale architektonische Brücke zwischen Simulation und Machine Learning ist das **Data Harvesting** (die Speicherung der RAM-Zustände auf die Festplatte). 

Ein naiver Ansatz würde Daten ab Sekunde 0 aufzeichnen. Da der Supermarkt zu Beginn jedoch komplett leer ist, würde das ML-Modell transiente Zustände (fehlerhafte Dynamiken) erlernen. Die Architektur erzwingt daher eine **Burn-in Period** (Warm-up-Phase). Die Simulation läuft für 3600 Ticks (1 Stunde) im Verborgenen, bis das System sein stochastisches Gleichgewicht (Steady-State) erreicht hat. Erst danach beginnt das Harvesting exakt alle 60 Ticks.

.. code-block:: python

   import pandas as pd
   import networkx as nx

   class SimulationEngine:
       def __init__(self, graph: nx.DiGraph, total_ticks: int, burn_in_ticks: int = 3600):
           self.graph = graph
           self.total_ticks = total_ticks
           self.burn_in_ticks = burn_in_ticks
           self.agents = []
           self.snapshot_dataset = []

       def run_simulation(self):
           """ Die zeitdiskrete Physik-Schleife. """
           for current_tick in range(self.total_ticks):
               self._spawn_new_agents(current_tick)
               
               for agent in self.agents:
                   occupancy = self.graph.edges[agent.current_edge]['occupancy']
                   capacity = self.graph.edges[agent.current_edge]['capacity']
                   agent.tick_update(occupancy, capacity)
                   
               # Data Harvesting: Nur im Steady-State und exakt jede Minute
               if current_tick > self.burn_in_ticks and current_tick % 60 == 0:
                   self._capture_graph_snapshot(current_tick)

       def _capture_graph_snapshot(self, tick: int):
           """ Friert den Graphen ein und extrahiert die Ground Truth. """
           edge_occupancy = {edge: 0 for edge in self.graph.edges()}
           for agent in self.agents:
               edge_occupancy[agent.current_edge] += 1
                   
           for (node_u, node_v), occupancy in edge_occupancy.items():
               self.snapshot_dataset.append({
                   "timestamp": tick,
                   "edge_id": f"{node_u}_{node_v}",
                   "occupancy": occupancy # Das Target für die Regression
               })

       def export_to_parquet(self, filepath: str):
           """ Sichert Millionen Snapshots hochkomprimiert. """
           pd.DataFrame(self.snapshot_dataset).to_parquet(filepath, engine='pyarrow')

5. Feature Engineering: Topologische Translation (Spatial Spillovers)
---------------------------------------------------------------------
Das ML-Modell (XGBoost) versteht keine 2D-Graphen, sondern nur flache 1D-Tensoren. Um zu lernen, dass sich ein Stau von Gang A in Gang B ausbreitet (Spatial Spillover), muss das Feature-Engineering die topologische Nachbarschaft des Graphen in Tabellenspalten zwingen.

.. code-block:: python

   def extract_spatial_features(df: pd.DataFrame, G: nx.DiGraph) -> pd.DataFrame:
       """ 
       Transformiert die Graphen-Topologie in flache ML-Features.
       Zieht die Auslastung der direkt angrenzenden Graphen-Kanten.
       """
       neighbor_loads = []
       for _, row in df.iterrows():
           target_node = row['edge_id'].split('_')[1] # Der Endknoten der aktuellen Kante
           
           # Finde alle ausgehenden Kanten (die physischen Nachbar-Gänge)
           out_edges = G.out_edges(target_node, data=True)
           loads = [data.get('occupancy', 0) for _, _, data in out_edges]
           
           # Das Feature ist die maximale Auslastung in der unmittelbaren Umgebung
           neighbor_loads.append(max(loads) if loads else 0)
           
       df['spatial_neighbor_max_occupancy'] = neighbor_loads
       return df

Diese architektonische Transformation erlaubt es dem tabellarischen Boosting-Modell, räumliche Flaschenhälse zu antizipieren, ohne dass extrem rechenintensive Graph Neural Networks (GNNs) eingesetzt werden müssen.

6. Systemintegrität: Forward Chaining & Target-Shift
----------------------------------------------------
Der fatalste methodische Fehler bei der Validierung synthetischer Daten ist **Data Leakage**. Würde man die exportierten Parquet-Daten vor dem Training via K-Fold Cross Validation zufällig mischen, könnte die KI "in die Zukunft sehen" (z. B. aus den Daten von 17:06 Uhr die Werte für 17:05 Uhr ableiten). 

Die Pipeline erzwingt daher eine strikte chronologische Validierung (**Time Series Split / Forward Chaining**). Zudem wird das abhängige Target-Label ($Y$) über einen negativen Pandas-Shift generiert:

.. code-block:: python

   import xgboost as xgb
   from sklearn.model_selection import TimeSeriesSplit

   # 1. Das Target definieren: Die Auslastung exakt 5 Minuten (5 Snapshots) in der Zukunft
   df['target_t_plus_5'] = df.groupby('edge_id')['occupancy'].shift(-5)
   df.dropna(inplace=True)

   # 2. Chronologischer Split (blockiert Leakage)
   tscv = TimeSeriesSplit(n_splits=5)
   model = xgb.XGBRegressor(n_estimators=200, max_depth=6)

   for train_idx, test_idx in tscv.split(X):
       model.fit(X.iloc[train_idx], y.iloc[train_idx])
       # Evaluierung erfolgt ausschließlich auf streng in der Zukunft liegenden Daten

Nur diese kompromisslose Methodik – von der fraktionalen Kinematik über den Burn-in-Schutz bis zum Forward-Chaining – garantiert, dass die generierten Daten kein wertloses Zufallsrauschen sind, sondern ein robuster, isomorpher Digitaler Zwilling der Supermarkt-Realität, auf dem die KI sicher generalisieren kann.