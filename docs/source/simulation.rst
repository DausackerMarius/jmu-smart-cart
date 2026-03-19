Agentenbasierte Simulation & Synthetische Datengenerierung
==========================================================

Moderne Machine-Learning-Modelle für die prädiktive Stau-Vorhersage (Traffic Prediction) erfordern eine hochdimensionale, historisierte und gelabelte Datenbasis. Da im Rahmen dieses Prototyps keine realen Sensordaten (z. B. LiDAR oder Kamera-Tracking) aus einem physischen Supermarkt zur Verfügung stehen, generiert das System seine Ground-Truth-Daten über eine **Agentenbasierte Simulation (Agent-Based Modeling, ABM)**.

Dieses Modul fungiert als "Digitaler Zwilling" (Digital Twin) des Supermarkts. Es transformiert die statische Graphen-Topologie durch die Injektion autonomer Entitäten (Agenten) in ein dynamisches, stochastisches System.

Teil I: Zeitdiskrete Simulation und Agenten-Kinematik
-----------------------------------------------------

Die Simulation läuft nicht als kontinuierlicher Fluss, sondern ist als zeitdiskrete *Physics Engine* implementiert. Die Systemzeit wird in atomaren Schritten (Ticks) von :math:`\Delta t = 1\text{s}` quantisiert. 

1. Stochastischer Spawn-Prozess (Inhomogener Poisson)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Die Instanziierung neuer Agenten an den Eingangsknoten erfolgt über einen inhomogenen Poisson-Prozess. Die zeitabhängige Ankunftsrate :math:`\lambda(t)` wird durch eine Sinus-Funktion über den simulierten Tagesverlauf moduliert, um Stoßzeiten (z. B. den Feierabend-Rush um 17:00 Uhr) mathematisch zu erzwingen.

.. math::

   P(k \text{ Ankünfte in } \Delta t) = \frac{(\lambda(t) \cdot \Delta t)^k e^{-\lambda(t) \cdot \Delta t}}{k!}

2. Der Lifecycle eines Agenten (State Machine)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Jeder Agent ist als Python-Objekt implementiert, das einen internen Zustandsautomaten (Finite State Machine) durchläuft:

* **Target Allocation:** Bei der Initialisierung zieht der Agent aus der Menge aller befüllten Regalknoten (Flexible & Fixed Zones) ein zufälliges Subset :math:`S \subset V`. Dies repräsentiert den individuellen Einkaufszettel.
* **Pathfinding (Dijkstra):** Das System berechnet über die ``NetworkX``-Bibliothek den initial kürzesten Pfad über die Transitknoten des Graphen.
* **Kinematik & Service Time:** Der Agent bewegt sich mit einer stochastisch normalverteilten Basisgeschwindigkeit :math:`v_{base} \sim \mathcal{N}(\mu, \sigma^2)` entlang der Kanten. Erreicht er einen Zielknoten aus :math:`S`, wechselt sein Zustand von ``MOVING`` auf ``SHOPPING``. Er verweilt dort für eine definierte *Service Time* (Such- und Greifzeit), während der er den Knoten für nachfolgende Agenten physisch blockiert.

3. Kanten-Auslastung und dynamische Penalty-Funktionen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Um das Entstehen von Staus (Congestion) zu simulieren, darf das System keine Kollisionen (Clipping) von Agenten zulassen. Die zentrale Metrik hierfür ist die Auslastung einer Kante (``edge_occupancy``). 

Betreten mehrere Agenten dieselbe Kante :math:`e`, greift eine dynamische Penalty-Schleife, die die tatsächliche Laufgeschwindigkeit :math:`v_{actual}` aller Agenten auf dieser Kante drosselt:

.. math::

   v_{actual} = v_{base} \cdot \max\left(0, 1 - \frac{\text{occupancy}(e)}{c_{max}(e)}\right)

Nähert sich die Auslastung der physischen Kapazitätsgrenze :math:`c_{max}`, konvergiert die Geschwindigkeit gegen Null. Erst durch diesen Rückkopplungsmechanismus staut sich das System messbar auf – das stochastische Chaos eines echten Supermarkts entsteht durch *Emergenz*.

Teil II: Feature Engineering und Data Harvesting
------------------------------------------------

Während der Tick-Loop läuft, führt das System in definierten Intervallen (z. B. alle 60 Sekunden) einen State-Dump durch. Die Transformation dieser reinen Graphen-Zustände in einen Machine-Learning-kompatiblen Vektorraum ist der kritischste Schritt der Pipeline.

1. Zirkadiane Rhythmik (Trigonometrisches Time-Encoding)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ein tiefgreifendes Problem in der Zeitreihenanalyse ist die numerische Repräsentation der Uhrzeit. Ein Modell, dem die Stunde als linearer Integer (0 bis 23) übergeben wird, kann den Tagesübergang nicht verstehen (23:00 Uhr und 01:00 Uhr sind numerisch maximal weit entfernt, obwohl chronologisch nah).
Das System umgeht diese Singularität durch die Transformation in einen zweidimensionalen trigonometrischen Raum:

.. math::

   h_{sin} = \sin\left(\frac{2\pi \cdot t_{hour}}{24}\right), \quad h_{cos} = \cos\left(\frac{2\pi \cdot t_{hour}}{24}\right)

2. Autoregressive Lag-Features (Sliding Window)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Die Vorhersage eines Staus erfordert die Kenntnis des Momentums (Trendrichtung). Um die Markov-Eigenschaft (Zukunft hängt nur vom direkten Jetzt ab) zu überwinden, extrahiert die Pipeline historische Kontext-Vektoren.
Für jede Kante werden die Auslastungen der vorherigen Zeitpunkte :math:`t-1, t-2` und :math:`t-3` als separate Features angehängt. Das Modell lernt somit nicht nur den absoluten Zustand, sondern den Gradienten (steigt der Stau gerade an oder löst er sich auf?).

3. Topologisches Target-Encoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Kanten-IDs (z. B. ``edge_vD1_vD2``) sind kategoriale Strings. Da High-Cardinality-Features klassisches One-Hot-Encoding "aufblähen" würden (Curse of Dimensionality), nutzt die Pipeline *Target Encoding*. Hierbei wird der String durch den historischen Mittelwert der Zielvariable (die durchschnittliche Staudichte dieser spezifischen Kante im gesamten Datensatz) ersetzt.

Teil III: ML-Architektur und Hyperparameter-Optimierung
-------------------------------------------------------

Das Ziel des Modells ist die Prädiktion eines kontinuierlichen Wertes (Personenanzahl auf Kante :math:`e` zum Zeitpunkt :math:`t+x`). Dies klassifiziert das Vorhaben als multivariates Regressionsproblem.

1. Wahl des Algorithmus (Gradient Boosting Ensembles)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Anstatt klassischer linearer Regressionsmodelle (OLS) nutzt das System baumbasierte Ensemble-Verfahren aus der Familie des **Gradient Boosting** (z. B. Random Forest / XGBoost). Die architektonische Begründung ist zweifach:
* **Nicht-Linearität:** Das Zusammenspiel aus topologischer Platzierung, Uhrzeit und Agenten-Dichte ist hochgradig nicht-linear. Entscheidungsbäume können verschachtelte Interaktionseffekte ohne manuelle Polynom-Transformationen nativ abbilden.
* **Fehler-Metrik (MSE):** Das Modell wird auf die Minimierung des mittleren quadratischen Fehlers (Mean Squared Error) trainiert. Durch die Quadrierung des Residuals werden extreme Stau-Ausreißer härter bestraft. Das zwingt den Algorithmus, sich auf die kritischen Engpässe zu fokussieren, anstatt nur den leeren Durchschnitts-Gang gut vorherzusagen.

2. Vermeidung von Data Leakage (Time Series Split)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Der fatalste methodische Fehler bei der Evaluierung von Zeitreihen ist der Einsatz von regulärem Random-Shuffle Cross-Validation (k-Fold CV). Würde man die Simulationsdaten zufällig mischen, würde das Modell die Stausituation von 17:05 Uhr vorhersagen, während es die Auslastung von 17:06 Uhr bereits in der Trainingsphase gesehen hat (klassischer Informationsfluss aus der Zukunft, *Data Leakage*).

Um die absolute Integrität der Evaluation zu garantieren, implementiert die Pipeline einen strikten **Time Series Split (Expanding Window)** aus der ``scikit-learn``-Bibliothek. 
Das Modell trainiert iterativ nur auf Vergangenheitsdaten (z. B. Tag 1 bis 3) und validiert auf ungesehenen Zukunftsdaten (Tag 4). Nur dieses strenge Paradigma stellt sicher, dass die in der Evaluation ausgewiesenen Metriken (wie ein :math:`R^2 > 0.9`) die echte Generalisierungsfähigkeit für den späteren Live-Betrieb des Supermarkts widerspiegeln.