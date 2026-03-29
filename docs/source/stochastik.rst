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

2. Kybernetik des Checkouts: Das M/M/1/K Verlustsystem
------------------------------------------------------
In der theoretischen Informatik werden Wartesysteme standardmäßig mit der Kendall-Notation (A/S/c) klassifiziert. Ein naiver Architekturansatz würde eine Supermarktkasse als simples M/M/1-System (unendliche Schlangenlänge) modellieren.

Der fatale Fehler der klassischen Theorie: Ein reguläres M/M/1-Modell geht von einem unendlichen Warteraum aus. In der physischen Realität eines Supermarkts ist der Platz vor der Kasse jedoch durch die umgebenden Regale streng limitiert. Staut sich die Schlange zu weit zurück, blockiert sie die Hauptgänge und neue Kunden weichen physikalisch bedingt auf andere Kassen aus.

Das JMU Smart Cart System implementiert daher das wesentlich realitätsnähere **M/M/1/K Warteschlangenmodell** (Verlustsystem). 
* **M (Arrivals):** Markov-Ankunftsprozess (Zufälliges Eintreffen der Kunden, abgeleitet aus den IoT-Daten).
* **M (Service):** Markov-Bedienzeit (Durchschnittliche Abfertigungsgeschwindigkeit).
* **1 (Servers):** Ein Kassierer pro Terminal.
* **K (Capacity):** Die strikt begrenzte Systemkapazität (z. B. $K=10$ Kunden). 

Ist die Schlange an einer Kasse maximal gefüllt, greift die Stochastik des Loss-Systems: Der ankommende Kunde wird abgewiesen (Loss) und der Routing-Algorithmus schwenkt automatisch auf die nächstbeste Kasse um.

3. Prädiktive Wartezeit & Systemstabilität
------------------------------------------
Die Berechnungen der Wartezeit basieren auf den stationären Gleichungen für Markow-Ketten. Der Auslastungsgrad (Traffic Intensity) ist definiert als $\rho = \frac{\lambda}{\mu}$. 

Das System berechnet die Wahrscheinlichkeit für ein leeres System ($P_0$) und ein komplett volles System ($P_K$). Aus der daraus resultierenden durchschnittlichen Warteschlangenlänge ($L_q$) und der effektiven Ankunftsrate leitet das Backend über das Gesetz von Little die präzise zu erwartende Wartezeit ab.

.. code-block:: python

    import random
    from typing import Dict

    class CheckoutStochasticEngine:
        """
        Berechnet die prädiktive Wartezeit für Checkout-Zonen 
        basierend auf dem M/M/1/K Verlustsystem.
        """
        @staticmethod
        def calculate_wait_metrics(lam: float, mu: float = 1.5, K: int = 10) -> Dict[str, float]:
            rho = lam / mu
            
            # --- Markow-Ketten Formeln für M/M/1/K ---
            if rho == 1.0:
                p0 = 1.0 / (K + 1)
                pk = 1.0 / (K + 1)
                lq = (K * (K - 1)) / (2 * (K + 1))
            else:
                p0 = (1 - rho) / (1 - rho**(K + 1))
                pk = (rho**K) * p0
                lq = (rho / (1 - rho)) - ((K + 1) * rho**(K + 1) / (1 - rho**(K + 1)))
                
            # Gesetz von Little für die erwartete Wartezeit
            lambda_eff = lam * (1 - pk)
            wait_minutes = lq / lambda_eff if lambda_eff > 0 else 0.0
            
            # Umrechnung in Sekunden + Basis-Bezahlzeit
            wait_sec = (wait_minutes * 60.0) + random.uniform(15.0, 35.0)
            
            return { 
                "wait_sec": wait_sec, 
                "p_wait": 1.0 - p0, 
                "lq": lq, 
                "p_loss": pk 
            }

4. Modellierung irrationaler Akteure: Balking & Reneging
--------------------------------------------------------
Ein robustes Backend muss menschliche Irrationalität abfangen. Wenn eine Kasse komplett überlastet ist, greifen in der Realität psychologische Effekte:
* Balking: Kunden sehen die extrem lange Schlange und stellen sich gar nicht erst an.
* Reneging: Kunden stehen in der Schlange, verlieren die Geduld und brechen den Einkauf ab.

Das System approximiert diesen strukturellen Kollaps eines Kassen-Knotens über die Kumulierte Verteilungsfunktion (CDF) einer psychologischen Schmerzgrenze. Überschreitet die prognostizierte Wartezeit einen definierten Schwellenwert (z.B. 10 Minuten), steigt die Abbruchwahrscheinlichkeit (Balking-Probability) drastisch an. Die Architektur feuert in diesem Fall eine exponentielle Penalty-Funktion, wodurch die Kosten der Kasse algorithmisch ins Unermessliche wachsen.

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

Die Klasse ``CheckoutOptimizationFacade`` nutzt die folgende psychologische Zielfunktion $Z$:

$$Z = (\alpha \cdot t_{walk}) + (\beta \cdot t_{wait\_penalized})$$

Menschen empfinden aktives Laufen als viel weniger quälend als passives Stehen in einer Schlange. Daher wird die reine physische Laufzeit ($t_{walk}$) mit dem Faktor $\alpha = 1.0$ normal gewichtet, während die erwartete Stehzeit an der Kasse ($t_{wait\_penalized}$) mit dem Faktor $\beta = 1.8$ überproportional bestraft wird. Das System schickt den Kunden im Zweifel lieber 20 Meter weiter, um ihm 1 Minute statisches Warten zu ersparen.

.. code-block:: python

    class CheckoutOptimizationFacade:
        """ Kapselt die Evaluierung aller Kassen und wählt das globale Optimum. """
        def __init__(self, alpha: float = 1.0, beta: float = 1.8):
            self.alpha = alpha # Gewichtung: Physisches, aktives Laufen
            self.beta = beta   # Gewichtung: Passives Stehen (Schmerzfaktor)
            self.last_calculated_wait = 0.0

        def evaluate_global_optimum(self, distances_to_checkouts: dict, checkout_nodes: list) -> str:
            best_node = None
            min_cost = float('inf')

            for checkout in checkout_nodes:
                walk_time = distances_to_checkouts.get(checkout.node_id, float('inf'))

                # 1. Stochastische Inferenz (IoT-Datenabgleich)
                metrics = CheckoutStochasticEngine.calculate_wait_metrics(lam=checkout.current_lambda)
                raw_wait = metrics["wait_sec"]

                # 2. Balking-Schutz (Abschreckung bei Überlastung)
                penalized_wait_time = apply_balking_penalty(raw_wait)

                # 3. Die psychologische Zielfunktion Z anwenden
                total_cost = (self.alpha * walk_time) + (self.beta * penalized_wait_time)

                if total_cost < min_cost:
                    min_cost = total_cost
                    best_node = checkout.node_id
                    self.last_calculated_wait = raw_wait

            return best_node

6. Die globale System-Synthese (Controller)
-------------------------------------------
Dieses Modul vereint nun alle bisherigen Architektur-Säulen – Graphen, Operations Research, Machine Learning, IoT und Stochastik – im zentralen Controller der Dash/Flask-Applikation. 

Der Lebenszyklus eines Routen-Requests durchläuft das Backend in exakt vier Phasen, bevor das fertige Daten-Paket an das Frontend-Tablet des Kunden gesendet wird:

1. Topologische Ingestion: Das System lädt den In-Memory Netzwerkgraphen.
2. Prädiktive Inferenz (KI): Das XGBoost-Modell modifiziert Gänge durch präventive Stau-Strafen.
3. Stochastische Terminierung: Die Facade errechnet die beste Kasse auf Basis der IoT-Daten und fixiert sie als Endknoten (unter strikter Einhaltung des Pre-Halftime-Protokolls).
4. Metaheuristisches Routing: Der TSP-Solver berechnet den perfekten Weg durch das mutierte Labyrinth.

Um Typensicherheit und Robustheit zu garantieren, werden die Antworten und Anfragen in strikte Pydantic-Modelle gegossen.

.. code-block:: python

    from pydantic import BaseModel
    import logging

    class RoutingRequest(BaseModel):
        """ Das strikte Schema für eingehende Tablet-Requests. """
        items: list[str]
        cart_progress: float   # Fortschritt des Einkaufs (0.0 bis 1.0)
        current_exit_node: str # Bisher berechneter oder Standard-Ausgangsknoten

    class RouteResponse(BaseModel):
        """ Der strikte Data Contract für das Frontend-Tablet. """
        status: str
        computation_time_ms: float
        route_nodes: list[str]
        stochastic_exit: str
        estimated_wait_time_sec: float

    class RoutingOrchestrator:
        """ Zentraler Controller der MVC-Architektur. """
        
        def synthesize_route(self, payload: RoutingRequest) -> RouteResponse:
            try:
                # 1. KI-Stauvorhersagen anwenden (Echtzeit-Stauumfahrung)
                penalized_graph = traffic_predictor.apply_edge_weights(base_graph)

                # 2. Stochastisches Checkout-Routing (Zwingendes Pre-Halftime-Protokoll)
                optimal_exit_node = payload.current_exit_node
                
                # Das Architektur-Gesetz greift: Die teure Neuberechnung der Kasse 
                # darf zwingend nur in der ersten Einkaufs-Hälfte erfolgen.
                facade = CheckoutOptimizationFacade()
                if payload.cart_progress <= 0.5:
                    optimal_exit_node = facade.evaluate_global_optimum(
                        distances_to_checkouts=walk_distances_to_exits,
                        checkout_nodes=iot_checkout_states 
                    )

                # 3. TSP Route berechnen (Operations Research Modul)
                tsp_path, computation_time = tsp_solver.calculate_tour(
                    graph=penalized_graph,
                    shopping_list=payload.items,
                    end_node=optimal_exit_node # Ziel ist durch Stochastik fixiert
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
                raise ValueError("Interne Routing-Synthese fehlgeschlagen.")

Der hier generierte Payload ist das mathematisch perfekt abgewogene, interdisziplinäre Resultat der gesamten Backend-Architektur – berechnet und ausgeliefert im Bruchteil einer Sekunde. Im nächsten Kapitel übernimmt das Frontend-Tablet dieses Datenpaket und transformiert die nackten Knoten-IDs in eine visuelle, interaktive Benutzeroberfläche für den Kunden.