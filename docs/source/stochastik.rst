Stochastik & Warteschlangentheorie (Checkout-Routing)
=====================================================

Die Optimierung der reinen physischen Laufwege (via TSP-Solver) löst nur einen Teil des Supermarkt-Routing-Problems. Das ultimative Nadelöhr (Bottleneck) in der Customer Journey ist der Kassenbereich. Ein deterministisch optimaler Laufweg verliert seinen gesamten Business-Wert, wenn der Kunde anschließend 15 Minuten in einer ineffizienten Warteschlange verbringt.

Um dieses Problem zu lösen, abstrahiert das JMU Smart Cart System den Kassenbereich nicht als statische Kanten im Graphen, sondern modelliert ihn dynamisch über die **Warteschlangentheorie (Queueing Theory)**.

1. Das mathematische Modell: M/M/1/K nach Kendall
-------------------------------------------------

Jede Kasse im Supermarkt (z. B. Knoten ``vK1``, ``vK2``) wird als ein unabhängiges, stochastisches System betrachtet. Die Architektur nutzt hierfür das **M/M/1/K-Warteschlangenmodell**. 

Die Kendall-Notation definiert die Systemgrenzen wie folgt:

* **M (Markov-Eigenschaft der Ankünfte):** Die Ankunft neuer Kunden an der Kasse (Geburtenprozess) ist poisson-verteilt. Die Zwischenankunftszeiten sind gedächtnislos (exponentiell verteilt) mit der Rate :math:`\lambda`.
* **M (Markov-Eigenschaft der Bedienung):** Die Servicezeit (Todesprozess), also der eigentliche Scan- und Bezahlvorgang durch den Kassierer, ist ebenfalls exponentiell verteilt mit der Rate :math:`\mu`.
* **1 (Anzahl der Server):** Es operiert genau ein Kassierer pro Schlange.
* **K (Systemkapazität):** Der Warteraum vor einer Kasse ist physisch begrenzt. Es können sich maximal :math:`K` Personen in der Schlange (inklusive der bedienten Person) aufhalten. 

Die Limitierung durch :math:`K` ist eine essenzielle architektonische Entscheidung: In einem echten Supermarkt blockiert eine endlose Schlange irgendwann den Hauptgang. Ist das System voll (:math:`n = K`), werden neu ankommende Kunden abgewiesen (Abweiswahrscheinlichkeit) und müssen eine andere Kasse wählen.

2. Systemdynamik und stationäre Zustände
----------------------------------------

Um die Wartezeit für das Routing vorherzusagen, berechnet die Klasse ``EnterpriseQueuingModel`` den stationären Zustand (Steady State) der Warteschlange. 

Die **Verkehrsintensität (Traffic Intensity)** des Systems ist definiert als:

.. math::

   \rho = \frac{\lambda}{\mu}

Auf Basis der Markow-Kette berechnet das System die Wahrscheinlichkeit :math:`P_n`, dass sich zu einem gegebenen Zeitpunkt exakt :math:`n` Kunden an der Kasse befinden:

.. math::

   P_n = 
   \begin{cases} 
   \frac{1 - \rho}{1 - \rho^{K+1}} \cdot \rho^n & \text{für } \rho \neq 1 \\
   \frac{1}{K + 1} & \text{für } \rho = 1 
   \end{cases}

3. Erwartungswerte und das Gesetz von Little (Little's Law)
-----------------------------------------------------------

Für die Wegeoptimierung benötigt das Frontend keinen abstrakten Wahrscheinlichkeitswert, sondern eine harte Metrik: **"Wie viele Sekunden dauert es?"**
Hierfür berechnet das Backend zunächst die erwartete Anzahl an Kunden im System (:math:`L`):

.. math::

   L = \sum_{n=0}^{K} n \cdot P_n

Anschließend wendet die Architektur eines der fundamentalsten Theoreme der Operations-Research-Forschung an: **Little's Law**. Es stellt den mathematischen Zusammenhang zwischen der Kundenanzahl (:math:`L`), der effektiven Ankunftsrate (:math:`\lambda_{eff}`) und der erwarteten Verweildauer (:math:`W`) her:

.. math::

   W = \frac{L}{\lambda \cdot (1 - P_K)}

Der Wert :math:`W` (in Sekunden) ist das finale Endprodukt der stochastischen Pipeline. Er repräsentiert die prognostizierte Gesamtzeit, die ein Kunde vom Anstellen bis zum Verlassen des Supermarktes benötigt.

4. Operationelle Synthese: Integration in den Graphen
-----------------------------------------------------

An diesem Punkt verschmilzt die Stochastik mit der Graphentheorie. Die berechnete Verweildauer :math:`W` wird von der ``QueuingModelFacade`` an den TSP-Solver übergeben. 

Das System addiert :math:`W` als **stochastische Zeitstrafe (Penalty)** auf das Kantengewicht, das zur jeweiligen Kasse führt. 
*Beispiel:* Befindet sich der Kunde am Ende seines Einkaufs, vergleicht der Held-Karp-Algorithmus die Pfade zu den Kassen ``vK1`` und ``vK2``. Obwohl ``vK1`` physisch 20 Meter näher liegt, prognostiziert das M/M/1/K-Modell hier aufgrund eines Feierabend-Rushes (hohes :math:`\lambda`) eine Wartezeit von :math:`W = 400\text{s}`. Für ``vK2`` beträgt :math:`W = 45\text{s}`. 
Der Algorithmus routet den Kunden autonom zur weiter entfernten, aber zeitlich effizienteren Kasse ``vK2``. 

5. Evaluierung der Warteschlangen-Dynamik
-----------------------------------------

Um die Sensitivität des Modells zu validieren, wurde die erwartete Wartezeit :math:`W` im Tagesverlauf unter variierenden Ankunftsraten simuliert.

.. image:: ../../eval_plots/stochastik_plot.png
   :width: 600px
   :align: center
   :alt: Simulation der Kassen-Wartezeiten

*Diskussion:* Der Plot visualisiert den massiven, nicht-linearen Anstieg der Wartezeit, sobald sich die Verkehrsintensität :math:`\rho` dem Wert 1 annähert (z. B. während der Rush-Hour um 17:00 Uhr). Es ist mathematisch bewiesen, dass Warteschlangen bei hoher Auslastung exponentiell eskalieren. Das System erkennt diese kritischen Schwellenwerte proaktiv und öffnet logisch alternative Routen zu Neben-Kassen, bevor der Hauptgang physisch blockiert wird (:math:`P_K \to 1`).