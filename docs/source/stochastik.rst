Stochastik, Warteschlangentheorie & System-Synthese
===================================================

Dieses Kapitel markiert den architektonischen Schlussstein des Backend-Systems. Im vorherigen Verlauf der Architektur wurde das Routing-Problem durch das Supermarkt-Labyrinth (TSP-Graphentheorie) deterministisch gelöst und durch prädiktive Machine-Learning-Modelle (KI-Stauvorhersagen für die Gänge) dynamisiert. 

An diesem Punkt entsteht jedoch eine kritische Systemlücke: Die physische "Letzte Meile". *An welcher der verfügbaren Kassen soll der Kunde seinen Einkauf beenden?*
Ein rein räumliches Routing würde hier katastrophal versagen. Ein geometrisch kurzer Weg zu Kasse 1 ist wertlos, wenn dort ein Kunde mit einem übervollen Einkaufswagen den Durchfluss blockiert, während die etwas weiter entfernte Kasse 2 komplett leer ist. 

**Die Abgrenzung zum Machine Learning:**
Warum nutzen wir für die Kassen kein Machine Learning (KI)? ML ist ein Schätzverfahren basierend auf Mustern der Vergangenheit. Eine Kassen-Schlange unterliegt jedoch harten, kausalen physikalischen Gesetzen. Da das Backend durch die IoT-Sensoren der Smart Carts in Echtzeit exakt weiß, wie viele Artikel die wartenden Kunden physisch im Wagen haben, muss das System nicht "raten". Es kann die Wartezeit präzise berechnen. Die Architektur übergibt die finale Entscheidung daher an ein dediziertes Modul für **Stochastik und Warteschlangentheorie (Queueing Theory)**.

1. Das Timing-Paradoxon: Die zwingende Halftime-Prädiktion
----------------------------------------------------------
Eine mathematische Vorhersage der besten Kasse ist nutzlos, wenn sie zur falschen Zeit berechnet wird. 

*Das Problem:* Wenn der Kunde bereits 90% seiner Einkaufsroute absolviert hat und kurz vor den Kassen steht, fehlt dem Graphen-Algorithmus der topologische Spielraum, um den Pfad noch elegant zu einer völlig anderen Kasse zu biegen. Der Kunde müsste abrupt umdrehen. Berechnet das System die Wartezeit jedoch schon am Eingang (bei 0%), ist die Vorhersage wertlos, da sich die Schlangen an den Kassen in der darauffolgenden halben Stunde massiv verändern werden.

*Die architektonische Lösung:* Das System erzwingt ein striktes, asynchrones Event-Timing. Die Vorhersage der optimalen Kasse darf niemals am Ende erfolgen. Die Architektur diktiert zwingend, dass Prädiktionen **immer vor oder exakt zur Halbzeit (Halftime)** des individuellen Einkaufs berechnet und in den TSP-Graphen injiziert werden müssen. Hat der Kunde 50% seiner Einkaufsliste abgearbeitet, wird das Stochastik-Modell getriggert. Dies garantiert die perfekte Balance aus Vorhersagegenauigkeit und topologischem Manövrierraum für den Routing-Algorithmus auf der zweiten Hälfte der Route.

2. Kybernetik des Checkouts: Der Bruch der Gedächtnislosigkeit
--------------------------------------------------------------
In der Mathematik werden Wartesysteme standardmäßig mit der **Kendall-Notation** ($A/S/c/K/D$) klassifiziert. Ein naiver Architekturansatz würde eine Supermarktkasse als simples **M/M/1-System** modellieren:
* **M (Arrival):** Markov-Ankunftsprozess (Zufälliges Eintreffen der Kunden).
* **M (Service):** Markov-Bedienzeit (Zufällige Dauer des Kassiervorgangs).
* **1:** Ein Kassierer pro Schlange.

**Der fatale Fehler der klassischen Theorie:**
Das zweite "M" (Markov) impliziert in der Stochastik zwingend die Eigenschaft der *Gedächtnislosigkeit* (Memoryless Property). 
*Verständnis-Exkurs:* Gedächtnislosigkeit bedeutet, dass die Wahrscheinlichkeit, dass ein Prozess endet, immer gleich bleibt, egal wie lange er schon läuft (wie beim Werfen eines Würfels). Für einen Kassiervorgang ist das eklatant falsch! Wenn ein Kassierer bereits 3 Minuten lang einen 50-Artikel-Warenkorb scannt, ist die Wahrscheinlichkeit, dass er in der nächsten Sekunde fertig wird, extrem hoch. Der Prozess hat ein "Gedächtnis".

Das JMU Smart Cart System verwirft diese Standardmodelle und implementiert stattdessen **IoT-gespeiste M/G/1-Systeme**:
Die Bedienzeit ist nicht zufällig (Markov), sondern folgt einer *Generellen Verteilung (G)*. Hier schließt sich die Brücke zur Hardware: Das Backend fragt den Zustand der Smart Carts ab und berechnet die Service-Zeit präzise aus der echten, physischen Anzahl der gescannten Artikel in der Schlange.

.. code-block:: python

   import numpy as np
   from typing import List, Dict

   class ServiceTimeDistribution:
       """
       Berechnet die General Distribution (G) der Service-Zeiten 
       basierend auf echten IoT-Warenkorbgrößen.
       """
       def __init__(self, scan_rate_sec: float = 1.8, base_payment_sec: float = 35.0):
           self.scan_rate = scan_rate_sec
           self.base_payment = base_payment_sec # Bezahlvorgang dauert im Schnitt 35s
           
       def get_moments(self, cart_sizes: List[int]) -> Dict[str, float]:
           """ Berechnet den Erwartungswert und die Varianz. """
           if not cart_sizes:
               return {"mean": 0.0, "variance": 0.0, "second_moment": 0.0}
               
           # Service-Zeit = (Anzahl Artikel * Scan-Geschwindigkeit) + Bezahlzeit
           service_times = [(size * self.scan_rate) + self.base_payment for size in cart_sizes]
           mean_s = np.mean(service_times)
           var_s = np.var(service_times)
           
           # Das zweite Moment (E[S^2]) ist das mathematische Herzstück zur Stau-Berechnung
           second_moment = var_s + (mean_s ** 2)
           
           return {"mean": mean_s, "variance": var_s, "second_moment": second_moment}

3. Prädiktive Wartezeit & Der Pooling-Effekt
--------------------------------------------
Um die exakte Wartezeit für den herannahenden Kunden zu berechnen, unterscheidet das System algorithmisch strikt zwischen klassischen Kassen und Self-Checkouts.

**Klassische Kassen (Pollaczek-Chintschin):**
Für normale Kassen (M/G/1) nutzt das System die fundamentale Pollaczek-Chintschin-Gleichung. Sie beweist, dass nicht nur die schiere Anzahl der Artikel, sondern vor allem die *Varianz* (die Unterschiedlichkeit der Warenkörbe) Staus verursacht:

.. math::
   W_q = \frac{\lambda \cdot \mathbb{E}[S^2]}{2(1-\rho)}

*Verständnis-Exkurs:* Wenn 10 Kunden exakt je 5 Artikel haben (Varianz = 0), fließt die Schlange perfekt. Hat aber ein Kunde 1 Artikel und der nächste 150 Artikel (hohe Varianz), stockt das System massiv. Der Term $\mathbb{E}[S^2]$ zwingt den Python-Code, Kassen mit hochvariablen Warenkörben mathematisch abzustrafen.

**Self-Checkouts & Der Pooling-Effekt (Kingman-Approximation):**
Moderne SB-Kassen teilen sich *eine* einzige Warteschlange für *mehrere* Terminals ($c$). In der Operations-Research nennt man dies den **Pooling-Effekt**. 
Ein System aus $4$ gepoolten Terminals (M/G/4) ist mathematisch drastisch effizienter als $4$ einzelne Kassen (4 $\times$ M/G/1). Warum? Weil ein Problemkunde (z.B. falscher Barcode) an Terminal 1 nicht die gesamte Schlange blockiert (Head-of-Line Blocking) – die restlichen Kunden fließen einfach an Terminal 2, 3 und 4 ab. 

Da für M/G/c Systeme keine exakte Formel existiert, implementiert das System die **Kingman-Approximation**, um diesen Pooling-Vorteil präzise zu belohnen:

.. code-block:: python

   def calculate_mgc_wait_time(lam: float, mu: float, c: int, var_a: float, var_s: float) -> float:
       """
       Implementierung der Kingman-Approximation für gepoolte Self-Checkouts.
       c = Anzahl der parallelen Terminals (Pooling-Faktor).
       """
       rho = lam / (c * mu) # Systemauslastung (0.0 bis 1.0)
       
       # Systemstabilitäts-Check: Wenn Auslastung > 100%, wächst die Schlange ins Unendliche
       if rho >= 1.0: return float('inf') 
           
       ca2 = var_a * (lam ** 2) # Variationskoeffizient Ankünfte
       cs2 = var_s * (mu ** 2)  # Variationskoeffizient Bedienung
       
       # Kingman-Formel: c im Nenner belohnt gepoolte Schlangen massiv
       term1 = (rho ** (np.sqrt(2 * (c + 1)) - 1)) / (c * (1 - rho))
       term2 = (ca2 + cs2) / 2.0
       
       return term1 * term2 * (1.0 / mu)

4. Modellierung irrationaler Akteure: Balking & Little's Law
------------------------------------------------------------
Ein robustes System muss menschliche Irrationalität abfangen. Wenn eine Kasse komplett überlastet ist, greifen in der Realität psychologische Effekte: **Balking** (Kunden stellen sich gar nicht erst an).

Das Backend nutzt das fundamentale **Gesetz von Little (Little’s Law:** $L = \lambda \cdot W$), um den strukturellen Kollaps eines Kassen-Nodes zu detektieren. Überschreitet die prognostizierte Wartezeit ($W$) einen Schmerz-Schwellenwert (z.B. 10 Minuten), implementiert das System eine exponentielle Penalty-Funktion. Die Kosten der Kasse explodieren ins Unermessliche, wodurch der TSP-Solver sie als Ziel verwirft.

.. code-block:: python

   import math

   def apply_balking_penalty(wait_time_seconds: float, patience_threshold: float = 600.0) -> float:
       """
       Architektonischer Schutzmechanismus: Verhindert das Routing in kollabierende Kassen.
       """
       # Berechnung der Abbruchwahrscheinlichkeit P(Balk)
       p_balk = 1.0 - math.exp(-max(0, wait_time_seconds - patience_threshold) / 200.0)
       
       if wait_time_seconds > patience_threshold:
           # Penalty-Multiplikator schließt Kassen mit >10 Min Wartezeit de facto aus
           penalty_multiplier = 1.0 + (p_balk * math.exp((wait_time_seconds - patience_threshold) / 100.0))
           return wait_time_seconds * penalty_multiplier
           
       return wait_time_seconds

5. Multi-Kriterielle Optimierung (Die Zielfunktion)
---------------------------------------------------
An diesem Punkt fließen Graphentheorie (Laufwege) und Stochastik (Wartezeit) zusammen. Die komplexe Logik wird durch das **Facade Design Pattern** gekapselt. 

Die ``CheckoutOptimizationFacade`` nutzt eine psychologische Zielfunktion ($Z$). 
*Verständnis-Exkurs:* Menschen empfinden aktives Laufen als viel weniger quälend als passives Stehen in einer Schlange. Daher wird die Laufzeit ($\alpha$) normal gewichtet, während die Stehzeit ($\beta$) mit dem Faktor 1.8 überproportional bestraft wird. Das System schickt den Kunden im Zweifel lieber 20 Meter weiter, um ihm 1 Minute Stehen zu ersparen.

.. math::
   \min Z = \alpha \cdot \text{ActiveWalkTime} + \beta \cdot \text{PassiveWaitTime}

**Code-Implementierung der Zielfunktion:**

.. code-block:: python

   class CheckoutOptimizationFacade:
       def __init__(self, alpha: float = 1.0, beta: float = 1.8):
           self.alpha = alpha # Gewichtung: Physisches Laufen
           self.beta = beta   # Gewichtung: Passives Stehen (Schmerzfaktor)
           
       def evaluate_global_optimum(self, distances: dict, checkouts: list) -> str:
           best_node = None
           min_cost = float('inf')
           
           for checkout in checkouts:
               walk_time = distances.get(checkout.node_id, float('inf'))
               
               # 1. Stochastische Inferenz (IoT-Datenabgleich)
               raw_wait_time = checkout.calculate_expected_wait_time()
               
               # 2. Balking-Schutz (Abschreckung)
               penalized_wait_time = apply_balking_penalty(raw_wait_time)
               
               # 3. Die Zielfunktion Z anwenden
               total_cost = (self.alpha * walk_time) + (self.beta * penalized_wait_time)
               
               if total_cost < min_cost:
                   min_cost = total_cost
                   best_node = checkout.node_id
                   
           return best_node

6. Die globale System-Synthese (app.py)
---------------------------------------
Dieses Kapitel vereint nun alle bisherigen Module (Graphen, Machine Learning, Stochastik, IoT) in einem einzigen FastAPI-Endpunkt. Der Lebenszyklus eines Routen-Requests durchläuft das Backend in exakt vier Phasen, bevor das JSON-Paket an das Tablet geht:

1. **Topologische Ingestion:** Laden des Netzwerkgraphen.
2. **Prädiktive Inferenz (KI):** Das XGBoost-Modell modifiziert Gänge durch Stau-Strafen.
3. **Stochastische Terminierung (Halftime-Trigger):** Die Facade errechnet die beste Kasse auf Basis der IoT-Cart-Daten und setzt sie als hartes Ziel.
4. **Metaheuristisches Routing (TSP):** Der Solver (z. B. Held-Karp oder Simulated Annealing) berechnet den perfekten Weg durch das mutierte Labyrinth zum gesetzten Kassenziel.

.. code-block:: python

   from fastapi import FastAPI
   import uvicorn

   app = FastAPI(title="JMU Smart Cart OR-Engine")

   @app.post("/api/v1/routing/synthesize")
   async def synthesize_route(payload: RoutingRequest):
       """
       Der Endpunkt der globalen System-Synthese.
       Verschmilzt Graphentheorie, Machine Learning, IoT und Stochastik.
       """
       # 1. KI-Stauvorhersagen anwenden (Echtzeit-Stauumfahrung)
       penalized_graph = traffic_predictor.apply_edge_weights(base_graph)
       
       # 2. Stochastisches Checkout-Routing (Der Halftime-Trigger)
       # Liest die aktuellen Warenkorbgrößen aller Kassen aus dem IoT-Netzwerk
       facade = CheckoutOptimizationFacade()
       optimal_exit_node = facade.evaluate_global_optimum(
           distances=walk_distances_to_exits, 
           checkouts=iot_checkout_states # Die Live-Daten der anderen Smart Carts
       )
       
       # 3. TSP Route berechnen (Operations Research)
       tsp_path, _ = tsp_solver.calculate_tour(
           graph=penalized_graph, 
           shopping_list=payload.items,
           end_node=optimal_exit_node # Ziel ist bereits stochastisch fixiert!
       )
       
       return {
           "status": "OPTIMIZED",
           "route_nodes": tsp_path,
           "stochastic_exit": optimal_exit_node
       }

Der resultierende Payload ist das mathematisch perfekt abgewogene, interdisziplinäre Resultat aus Raum (Graphen), Vorhersage (ML) und Zeit (Stochastik) – berechnet und ausgeliefert im Bruchteil einer Sekunde.