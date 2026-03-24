Stochastik, Warteschlangentheorie & System-Synthese
===================================================

Dieses Kapitel markiert den architektonischen Schlussstein des gesamten serverseitigen Backends. Im bisherigen Verlauf der Daten-Pipeline wurde das komplexe Routing-Problem durch das Supermarkt-Labyrinth via Operations Research deterministisch gelöst (Kapitel 5) und durch prädiktive Machine-Learning-Modelle dynamisch an Verkehrsstaus in den Gängen angepasst (Kapitel 7).

An diesem Punkt der Pipeline entsteht jedoch eine kritische Systemlücke: Die physische "Letzte Meile". An welcher der verfügbaren Kassen soll der Kunde seinen Einkauf beenden?
Ein rein räumliches (topologisches) Routing würde hier katastrophal versagen. Ein geometrisch kurzer Weg zu Kasse 1 ist absolut wertlos, wenn dort ein einzelner Kunde mit einem übervollen 150-Artikel-Einkaufswagen den Durchfluss blockiert, während die etwas weiter entfernte Kasse 2 komplett leer ist.

Die Abgrenzung zum Machine Learning: Warum nutzt die Architektur für die Kassen-Vorhersage kein Machine Learning (wie zuvor bei den Gängen)? Machine Learning ist ein induktives Schätzverfahren basierend auf Vergangenheitsmustern. Eine Kassenschlange unterliegt jedoch harten, kausalen physikalischen Gesetzen. Da das Backend durch die IoT-Sensoren der Smart Carts in Echtzeit exakt weiß, wie viele Artikel die wartenden Kunden physisch im Wagen haben, muss das System nicht "raten". Es kann die zu erwartende Wartezeit mathematisch präzise berechnen. 

Die Architektur übergibt diese finale Entscheidung daher an ein dediziertes Modul für Stochastik und Warteschlangentheorie (Queueing Theory). Am Ende dieses Moduls werden alle Systemkomponenten in der Haupt-Applikation synthetisiert und für das Frontend-Tablet verpackt.

1. Das Timing-Paradoxon & Pre-Halftime Prädiktion
-------------------------------------------------
Eine mathematisch perfekte Vorhersage der besten Kasse ist für das System völlig nutzlos, wenn sie zur falschen Zeit berechnet wird. Die Architektur muss hier ein fundamentales Timing-Paradoxon lösen.

Das topologische Deadlock-Problem: Wenn der Kunde bereits 90 % seiner Einkaufsroute absolviert hat und direkt vor den Kassenzonen steht, fehlt dem Graphen-Algorithmus der räumliche Spielraum, um den Pfad noch elegant zu einer völlig anderen Kasse am anderen Ende des Marktes zu biegen. Der Kunde müsste abrupt umdrehen und gegen den Strom anderer Kunden navigieren (Geisterfahrer-Gefahr). 
Berechnet das System die Wartezeit jedoch schon am Eingang (bei 0 % Fortschritt), ist die Vorhersage ebenfalls wertlos, da sich die Schlangen an den Kassen im Verlauf des 40-minütigen Einkaufs massiv verändern werden.

Die architektonische Lösung: Das System erzwingt ein striktes, asynchrones Event-Timing – den "Commitment Point". Die Vorhersage der Kassen-Wartezeiten darf niemals am Ende des Einkaufs angefordert werden. Die Architektur diktiert zwingend, dass die Prädiktionen stets vor Erreichen der Halbzeit des individuellen Einkaufs (bei exakt 40 % bis 45 % abgehakter Artikel) berechnet und als harter Zielknoten in den TSP-Graphen injiziert werden. Dies garantiert dem Routing-Solver auf der zweiten Hälfte der Route den maximalen topologischen Manövrierraum, um den Kunden fließend und ohne kognitive Brüche in Richtung des besten Ausgangs zu leiten.

2. Kybernetik des Checkouts: Der Bruch der Gedächtnislosigkeit
--------------------------------------------------------------
In der theoretischen Informatik werden Wartesysteme standardmäßig mit der Kendall-Notation (A/S/c) klassifiziert. Ein naiver Architekturansatz würde eine Supermarktkasse als simples M/M/1-System modellieren:

* A (Arrivals) = M: Markov-Ankunftsprozess (Zufälliges Eintreffen der Kunden).
* S (Service) = M: Markov-Bedienzeit (Zufällige Dauer des Kassiervorgangs).
* c (Servers) = 1: Ein Kassierer pro Schlange.

Der fatale Fehler der klassischen Theorie: Das zweite "M" (Markov-Bedienzeit) impliziert in der Stochastik zwingend die Eigenschaft der Gedächtnislosigkeit (Memoryless Property). 
Gedächtnislosigkeit bedeutet, dass die Wahrscheinlichkeit, dass ein Prozess in der nächsten Sekunde endet, immer exakt gleich bleibt, egal wie lange er schon läuft (wie beim Werfen eines fairen Würfels). Für einen Kassiervorgang ist das physikalisch eklatant falsch! Wenn ein Kassierer bereits 3 Minuten lang einen großen Warenkorb scannt, ist die Wahrscheinlichkeit, dass er in der nächsten Sekunde fertig wird, extrem hoch. Der Abfertigungsprozess hat ein physikalisches "Gedächtnis".

Das JMU Smart Cart System verwirft diese fehlerhaften Standardmodelle und implementiert stattdessen IoT-gespeiste M/G/1-Systeme:
Die Bedienzeit ist nicht stochastisch zufällig, sondern folgt einer Generellen Verteilung (G). Hier schließt sich die Brücke zur Hardware: Das Backend fragt den Status der Smart Carts ab und berechnet die Service-Zeit präzise aus der echten Anzahl der gescannten Artikel.

.. code-block:: python

   import numpy as np
   from typing import List, Dict

   class ServiceTimeDistribution:
       """
       Berechnet die General Distribution (G) der Service-Zeiten
       basierend auf echten, ausgelesenen IoT-Warenkorbgrößen der aktiven Smart Carts.
       """
       def __init__(self, scan_rate_sec: float = 1.8, base_payment_sec: float = 35.0):
           self.scan_rate = scan_rate_sec
           self.base_payment = base_payment_sec # Bezahlvorgang dauert im Schnitt 35s

       def get_moments(self, cart_sizes: List[int]) -> Dict[str, float]:
           """ 
           Berechnet den Erwartungswert (Mean), die Varianz und das zweite Moment. 
           Diese Werte sind zwingend für die spätere Wartezeit-Formel erforderlich.
           """
           if not cart_sizes:
               return {"mean": 0.0, "variance": 0.0, "second_moment": 0.0}

           # Service-Zeit = (Anzahl Artikel * Scan-Geschwindigkeit) + Bezahlzeit
           service_times = [(size * self.scan_rate) + self.base_payment for size in cart_sizes]
           
           mean_s = float(np.mean(service_times))
           var_s = float(np.var(service_times))

           # Das zweite Moment (E[S^2]) ist das mathematische Herzstück zur Stau-Berechnung.
           # Es beweist, dass hohe Varianzen (sehr kleine und sehr große Einkäufe gemischt)
           # exponentiell zu Staus führen.
           second_moment = var_s + (mean_s ** 2)

           return {"mean": mean_s, "variance": var_s, "second_moment": second_moment}

3. Prädiktive Wartezeit & Der Pooling-Effekt
--------------------------------------------
Um die exakte Wartezeit für den herannahenden Kunden zu berechnen, unterscheidet das System algorithmisch strikt zwischen klassischen personalbesetzten Kassen und Self-Checkouts.

3.1 Klassische Kassen (Pollaczek-Chintschin für M/G/1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Für normale Kassen nutzt das System die fundamentale Pollaczek-Chintschin-Gleichung. Die zu erwartende Wartezeit W_q in der Schlange (ohne die eigene Bedienzeit) berechnet sich aus der Ankunftsrate Lambda und den zuvor berechneten Momenten der Service-Zeit.

Die Architektur erzwingt diese Formel, da sie mathematisch beweist, dass nicht nur die schiere Anzahl der Artikel, sondern vor allem die Varianz (die Unterschiedlichkeit der Warenkörbe) Staus verursacht. Wenn 10 Kunden exakt je 5 Artikel haben (Varianz = 0), fließt die Schlange perfekt. Hat aber ein Kunde 1 Artikel und der nächste 150 Artikel (hohe Varianz), stockt das System massiv. Der Term E[S^2] zwingt den Algorithmus, hochvariable Schlangen abzustrafen.

.. code-block:: python

   def calculate_mg1_wait_time(lam: float, mean_s: float, second_moment_s: float) -> float:
       """
       Implementierung der Pollaczek-Chintschin-Formel für eine klassische M/G/1 Kasse.
       Gibt die erwartete Wartezeit in Sekunden zurück.
       """
       rho = lam * mean_s # Auslastungsgrad (Traffic Intensity)

       # Systemstabilitäts-Check: Ist die Kasse überlastet?
       if rho >= 1.0 or lam == 0:
           return float('inf')

       # Pollaczek-Chintschin: W_q = (Lambda * E[S^2]) / (2 * (1 - Rho))
       numerator = lam * second_moment_s
       denominator = 2.0 * (1.0 - rho)
       
       return numerator / denominator

3.2 Self-Checkouts & Die Kingman-Approximation (M/G/c)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Moderne SB-Kassen teilen sich eine einzige Warteschlange für mehrere Terminals (c). In der Operations-Research nennt man dies den Pooling-Effekt.

Ein System aus 4 gepoolten Terminals (M/G/4) ist mathematisch drastisch effizienter als 4 einzelne, isolierte Kassen (4x M/G/1). Der Grund: Ein "Problemkunde" (z.B. falscher Barcode oder Kleingeld-Suche) an Terminal 1 blockiert nicht die gesamte Schlange (kein Head-of-Line Blocking). Die restlichen Kunden fließen einfach an Terminal 2, 3 und 4 ab. Da für M/G/c Systeme keine exakte geschlossene Formel existiert, implementiert das System die Kingman-Approximation.

.. code-block:: python

   def calculate_mgc_wait_time(lam: float, mu: float, c: int, var_a: float, var_s: float) -> float:
       """
       Implementierung der Kingman-Approximation für gepoolte Self-Checkouts.
       c = Anzahl der parallelen Terminals (Pooling-Faktor).
       """
       rho = lam / (c * mu) 

       if rho >= 1.0 or lam == 0: 
           return float('inf')

       ca2 = var_a * (lam ** 2) # Variationskoeffizient der Ankünfte
       cs2 = var_s * (mu ** 2)  # Variationskoeffizient der Bedienung

       # Kingman-Formel: 'c' im Nenner belohnt gepoolte Schlangen massiv
       term1 = (rho ** (np.sqrt(2 * (c + 1)) - 1)) / (c * (1 - rho))
       term2 = (ca2 + cs2) / 2.0

       return term1 * term2 * (1.0 / mu)

4. Modellierung irrationaler Akteure: Balking & Reneging
--------------------------------------------------------
Ein robustes Backend muss menschliche Irrationalität abfangen. Wenn eine Kasse komplett überlastet ist, greifen in der Realität psychologische Effekte:
* Balking: Kunden sehen die extrem lange Schlange und stellen sich gar nicht erst an.
* Reneging: Kunden stehen in der Schlange, verlieren die Geduld und brechen den Einkauf ab.

Das System nutzt das fundamentale Gesetz von Little (Little's Law: L = Lambda * W), um den strukturellen Kollaps eines Kassen-Knotens zu detektieren. Überschreitet die prognostizierte Wartezeit (W) einen psychologischen Schmerz-Schwellenwert (z.B. 10 Minuten), feuert das System eine exponentielle Penalty-Funktion. Die Kosten der Kasse explodieren algorithmisch ins Unermessliche.

.. code-block:: python

   import math

   def apply_balking_penalty(wait_time_seconds: float, patience_threshold: float = 600.0) -> float:
       """
       Architektonischer Schutzmechanismus: Verhindert das Routing in kollabierende Kassen.
       Nutzt eine exponentielle Straffunktion ab dem psychologischen Schwellenwert.
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
An diesem Punkt fließen Graphentheorie (die physischen Laufwege aus Kapitel 3) und Stochastik (die Wartezeit) zusammen. Die Logik wird durch das Facade Design Pattern gekapselt.

Die Klasse ``CheckoutOptimizationFacade`` nutzt eine psychologische Zielfunktion (Z). Menschen empfinden aktives Laufen als viel weniger quälend als passives Stehen in einer Schlange. Daher wird die Laufzeit (Alpha) normal gewichtet, während die Stehzeit (Beta) mit dem Faktor 1.8 überproportional bestraft wird. Das System schickt den Kunden im Zweifel lieber 20 Meter weiter, um ihm 1 Minute Stehen zu ersparen.

.. code-block:: python

   class CheckoutOptimizationFacade:
       """ Kapselt die Evaluierung aller Kassen und wählt das globale Optimum. """
       def __init__(self, alpha: float = 1.0, beta: float = 1.8):
           self.alpha = alpha # Gewichtung: Physisches, aktives Laufen
           self.beta = beta   # Gewichtung: Passives Stehen (Schmerzfaktor)

       def evaluate_global_optimum(self, distances_to_checkouts: dict, checkout_nodes: list) -> str:
           best_node = None
           min_cost = float('inf')

           for checkout in checkout_nodes:
               walk_time = distances_to_checkouts.get(checkout.node_id, float('inf'))

               # 1. Stochastische Inferenz (IoT-Datenabgleich)
               if checkout.is_pooled:
                   raw_wait = calculate_mgc_wait_time(...)
               else:
                   raw_wait = calculate_mg1_wait_time(...)

               # 2. Balking-Schutz (Abschreckung bei Überlastung)
               penalized_wait_time = apply_balking_penalty(raw_wait)

               # 3. Die psychologische Zielfunktion Z anwenden
               total_cost = (self.alpha * walk_time) + (self.beta * penalized_wait_time)

               if total_cost < min_cost:
                   min_cost = total_cost
                   best_node = checkout.node_id

           return best_node

6. Die globale System-Synthese (app.py)
---------------------------------------
Dieses Modul vereint nun alle bisherigen Architektur-Säulen – Graphen, Operations Research, Machine Learning, IoT und Stochastik – in einem einzigen, hochperformanten API-Endpunkt. 

Der Lebenszyklus eines Routen-Requests durchläuft das Backend in exakt vier Phasen, bevor das fertige JSON-Paket über das TCP-Netzwerk an das Tablet des Kunden gesendet wird:

1. Topologische Ingestion: Das System lädt den In-Memory Netzwerkgraphen.
2. Prädiktive Inferenz (KI): Das XGBoost-Modell modifiziert Gänge durch präventive Stau-Strafen.
3. Stochastische Terminierung: Die Facade errechnet die beste Kasse auf Basis der IoT-Daten und fixiert sie als Endknoten.
4. Metaheuristisches Routing: Der TSP-Solver berechnet den perfekten Weg durch das mutierte Labyrinth.

Um Typensicherheit und eine saubere API-Dokumentation (Swagger UI) zu garantieren, werden die Antworten in strikte Pydantic-Modelle gegossen.

.. code-block:: python

   from fastapi import FastAPI, HTTPException
   from pydantic import BaseModel
   import logging

   app = FastAPI(title="JMU Smart Cart System-Controller")

   class RouteResponse(BaseModel):
       """ Der strikte Data Contract für das Frontend-Tablet. """
       status: str
       computation_time_ms: float
       route_nodes: list[str]
       stochastic_exit: str
       estimated_wait_time_sec: float

   @app.post("/api/v1/routing/synthesize", response_model=RouteResponse)
   def synthesize_route(payload: RoutingRequest):
       """
       Der Endpunkt der globalen System-Synthese.
       Verschmilzt Graphentheorie, Machine Learning, IoT und Stochastik.
       """
       try:
           # 1. KI-Stauvorhersagen anwenden (Echtzeit-Stauumfahrung)
           penalized_graph = traffic_predictor.apply_edge_weights(base_graph)

           # 2. Stochastisches Checkout-Routing (Mit Pre-Halftime-Protokoll)
           facade = CheckoutOptimizationFacade()
           optimal_exit_node = facade.evaluate_global_optimum(
               distances_to_checkouts=walk_distances_to_exits,
               checkout_nodes=iot_checkout_states 
           )

           # 3. TSP Route berechnen (Operations Research Modul)
           tsp_path, computation_time = tsp_solver.calculate_tour(
               graph=penalized_graph,
               shopping_list=payload.items,
               end_node=optimal_exit_node # Ziel ist bereits stochastisch fixiert!
           )

           return RouteResponse(
               status="OPTIMIZED",
               computation_time_ms=computation_time,
               route_nodes=tsp_path,
               stochastic_exit=optimal_exit_node,
               estimated_wait_time_sec=facade.last_calculated_wait
           )
           
       except Exception as e:
           logging.error(f"Kritischer Synthese-Fehler: {e}")
           raise HTTPException(status_code=500, detail="Interne Routing-Synthese fehlgeschlagen.")

Der hier generierte Payload ist das mathematisch perfekt abgewogene, interdisziplinäre Resultat der gesamten Backend-Architektur – berechnet und ausgeliefert im Bruchteil einer Sekunde. Im nächsten Kapitel übernimmt das Frontend-Tablet dieses Datenpaket und transformiert die nackten Knoten-IDs in eine visuelle, interaktive Benutzeroberfläche für den Kunden.