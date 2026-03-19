Operations Research & Routing-Strategien
========================================

Das Herzstück der Navigation im Smart Supermarket ist die Lösung des **Traveling Salesperson Problems (TSP)**. Da das Finden einer global optimalen Route in einem Graphen mit :math:`n` Zielen (Produkten) in der Komplexitätstheorie als NP-schwer eingestuft wird, setzt das System auf eine hybride Architektur, die zwischen mathematischer Exaktheit und effizienten Metaheuristiken skaliert.

Die Baseline: Warum kein Christofides-Algorithmus?
--------------------------------------------------
In der theoretischen Informatik gilt der *Christofides-Algorithmus* als Standard für das metrische TSP, da er im Worst-Case eine Approximationsgüte von 1,5 garantiert. Für die Supermarkt-Topologie wurde sich jedoch bewusst gegen diesen Ansatz entschieden:

1. **Verletzung der Dreiecksungleichung:** Christofides setzt zwingend voraus, dass der direkte Weg zwischen zwei Knoten immer der kürzeste ist. In realen Supermarkt-Graphen (bedingt durch blockierte Gänge und Regale) ist die Dreiecksungleichung jedoch oft nicht gegeben.
2. **Latenz-Overhead:** Die Berechnung eines minimalen Spannbaums (Minimum Spanning Tree, MST) und eines Perfect Matchings ist für sehr kleine Punktmengen (Standardeinkauf) rechentechnisch ineffizient.

Stattdessen nutzt das System ein dynamisches **Strategy-Pattern**, das den Solver an die Warenkorbgröße anpasst.

Die Hybride Routing-Strategie
-----------------------------
Das System evaluiert bei jedem Berechnungsschritt die Anzahl der Zielknoten (:math:`n`) und wählt autonom den effizientesten Algorithmus:

* **Held-Karp (Exakte Dynamische Programmierung):** Für kleine Warenkörbe (:math:`n \le 11`) wird die globale Optimallösung berechnet. Durch Speicherung von Teilproblemen erreicht Held-Karp eine Zeitkomplexität von :math:`\mathcal{O}(n^2 2^n)`, was ihn für kleine :math:`n` zur perfekten "Ground Truth" macht.
* **Simulated Annealing (Physikalische Metaheuristik):** Für mittlere Mengen (:math:`12 \le n \le 15`) simuliert das System den thermodynamischen Abkühlungsprozess von Metallen. Hohe "Starttemperaturen" erlauben temporär auch schlechtere Routen, um lokale Minima im Graphen erfolgreich zu verlassen.
* **Evolutionäre Algorithmen (GA & ACO):** Bei Großlisten (:math:`n > 15`) greifen Ameisenkolonie-Optimierung und Genetische Algorithmen, um den Suchraum durch natürliche Selektion und Pheromon-Updates in Sekundenbruchteilen einzugrenzen.

Graphen-Manipulation: Integration des Traffic-Modells
-----------------------------------------------------
Ein klassisches TSP berechnet den kürzesten Weg auf Basis statischer Distanzen (in Metern). Im Smart Cart System wird dieser Graph jedoch durch den ``TrafficPredictor`` dynamisch transformiert. 

*(Hinweis: Die architektonische Evaluation und Güteprüfung des Traffic-Modells ist detailliert im Kapitel "Machine Learning Architektur" dokumentiert).*

Anstatt Kanten unverändert zu belassen, führt das OR-Modul eine **Edge Weight Penalty** (Kantenbestrafung) durch. Das Gewicht :math:`W` einer Kante :math:`e` zum Zeitpunkt :math:`t` berechnet sich aus der Basisdistanz :math:`d` und der vorhergesagten Personenanzahl :math:`p`:

.. math::

   W(e, t) = d(e) + \lambda \cdot P_{traffic}(e, t)

**Die algorithmische Konsequenz:**
Das Routing-System betrachtet Stausituationen als "künstliche Verlängerung" eines Ganges. Wenn der Regressor für einen Gang einen massiven Stau prognostiziert, erhöht sich die Kantengewichtung drastisch. Der TSP-Solver wertet diese Route nun als ineffizient aus und wird den Kunden automatisch durch einen physisch längeren, aber zeitlich schnelleren Neben-Gang navigieren. So verschmelzen Prädiktive KI und Operations Research zu einer adaptiven Echtzeit-Navigation.