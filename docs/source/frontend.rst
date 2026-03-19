Frontend-Architektur & Human-Computer Interaction (UX)
======================================================

Die mathematische Exzellenz der Backend-Algorithmen (TSP-Solver, ML-Prädiktoren) entfaltet ihren praktischen Business-Wert erst durch eine performante und ergonomische Benutzeroberfläche. Das Frontend des JMU Smart Supermarkets fungiert als Übersetzungs-Schicht: Es abstrahiert die hochkomplexe Graphentheorie und stochastische Modellierung in intuitive, visuelle Heuristiken, um die kognitive Belastung (Cognitive Load) des Nutzers während des Einkaufs zu minimieren.

Das System ist als reaktive **Single-Page Application (SPA)** konzipiert und mittels des analytischen Frameworks *Plotly Dash* implementiert. Anstatt statische HTML-Dokumente auszuliefern, agiert das Frontend als interaktiver, webbasierter Vektor-Renderer für die zugrundeliegenden Graphen-Topologien.

1. Das architektonische Paradigma: Reaktives MVC-Pattern
--------------------------------------------------------

Das Frontend (``app.py``) trennt die Datenlogik strikt von der visuellen Repräsentation. Da Dash-Applikationen über zustandslose HTTP-Protokolle (Stateless) operieren, erfordert die Synchronisation mit dem Backend ein präzises Architekturmuster:

* **Model (Shared State):** Das Frontend speichert keinen eigenen topologischen Zustand. Es kommuniziert als reiner Client über In-Memory-Referenzen mit der ``model.py``. Es konsumiert den aktuellen Graphen (:math:`G`), die berechneten TSP-Routen und die ML-Vorhersagen als "Single Source of Truth".
* **View (Dynamic DOM):** Die Funktion ``serve_layout()`` generiert den Document Object Model (DOM) Baum. Architektonisch entscheidend ist hierbei die dynamische Evaluierung: Das Layout wird bei jedem Seitenaufruf als Funktion evaluiert (Runtime-Binding), um temporale Abhängigkeiten – wie die tagesaktuelle Initialisierung des Poisson-Prozesses – bei einem Seiten-Reload exakt zu synchronisieren.
* **Controller (DAG-Callbacks):** Die Nutzerinteraktionen werden über asynchrone Event-Listener (``@app.callback``) abgefangen. Dash orchestriert diese Callbacks intern als *Directed Acyclic Graph (DAG)*. Dies stellt sicher, dass verknüpfte UI-Updates in der korrekten mathematischen Reihenfolge ausgeführt werden, ohne blockierende Full-Page-Reloads zu erzwingen.

2. Visuelles Encoding & Geometrisches Rendering
-----------------------------------------------

Die Abbildung des Supermarkts im Browser ist kein simples, statisches Rasterbild (PNG/JPG), sondern eine dynamisch berechnete Vektor-Projektion.

2.1 Topologische Projektion und Heatmaps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Die abstrakten Knoten (:math:`V`) und Kanten (:math:`E`) des NetworkX-Graphen werden in Plotly-Traces (Scatter-Plots) übersetzt. Die X/Y-Koordinaten der Knoten verankern die Regale im zweidimensionalen Canvas. 
Ein besonderes Augenmerk liegt auf dem kognitiven Mapping der ML-Vorhersagen: Im *Heatmap-Modus* konvertiert das Frontend die vom Regressor vorhergesagten Stau-Wahrscheinlichkeiten in einen kontinuierlichen Farbgradienten (Color Scale). Der Nutzer erfasst so komplexe stochastische Vorhersagen innerhalb von Millisekunden visuell.

2.2 Geometrische Pfad-Interpolation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Der Held-Karp-Algorithmus (TSP) aus dem Operations-Research-Modul liefert aus Effizienzgründen lediglich eine diskrete Permutation von Zielknoten (z. B. Start -> Milch -> Kasse). Ein naives Rendering würde eine gerade Linie durch Wände und Regale ziehen. 
Die Funktion ``build_true_plot_path()`` implementiert daher eine Pfad-Interpolation: Sie nutzt den Dijkstra-Algorithmus, um die fehlenden Transit-Strecken (die physischen Gänge) zwischen den Wegpunkten topologisch korrekt aufzufüllen.

2.3 Vektormathematik für Navigationsanweisungen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Um den Nutzer nicht nur visuell, sondern auch textuell zu leiten ("Bitte links abbiegen"), muss das System die Geometrie des Graphen "verstehen". Die Funktion ``calculate_turn_direction()`` verzichtet auf rechenintensive Winkel-Funktionen (Trigonometrie) und nutzt stattdessen das elegante **2D-Kreuzprodukt** zweier Richtungsvektoren.

Gegeben seien drei aufeinanderfolgende Knotenkoordinaten :math:`p_1, p_2, p_3` auf der Route. Es werden zwei Vektoren gebildet:
:math:`\vec{u} = p_2 - p_1` (Eingangsrichtung) und :math:`\vec{v} = p_3 - p_2` (Ausgangsrichtung).

Das 2D-Kreuzprodukt (Determinante) berechnet sich als:

.. math::

   c = u_x \cdot v_y - u_y \cdot v_x

* Ist :math:`c > 0`, rotiert der Pfad gegen den Uhrzeigersinn (**Links abbiegen**).
* Ist :math:`c < 0`, rotiert der Pfad im Uhrzeigersinn (**Rechts abbiegen**).
* Ist :math:`c = 0`, verläuft der Weg kollinear (**Geradeaus**).

Diese mathematische Herangehensweise ist numerisch hochgradig stabil, resistent gegen Division-by-Zero-Fehler und garantiert in der Applikation extrem schnelle Laufzeiten bei der Erstellung der Wegbeschreibung.

3. Nebenläufigkeit (Concurrency) & Expert-in-the-Loop
-----------------------------------------------------

Ein webbasiertes Dashboard muss Multi-Threading und parallele Nutzerzugriffe sicher handhaben, insbesondere wenn das Backend hochsensible Speicheroperationen durchführt.

3.1 Thread-Safety und Race Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Die Funktion ``admin_manage()`` gewährt Administratoren das Recht, die Graphen-Topologie (Regale, Inventar) zur Laufzeit zu verändern. Würde parallel ein Nutzer eine Pfad-Berechnung triggern, käme es zu einer *Race Condition*, was zu einem fatalen Server-Absturz (Segmentation Fault / KeyError) führt.
Die Applikation nutzt daher strenge **Threading-Locks** (``with inv_manager._lock:``). Dies erzwingt Thread-Sicherheit und garantiert die Konsistenz und Isolation der In-Memory-Datenbank (angelehnt an die ACID-Prinzipien relationaler Datenbanken) während des Lese-/Schreibzugriffs.

3.2 Der Expert-in-the-Loop Ansatz
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Maschinelles Lernen ist probabilistisch und macht Fehler. Das UI-Design des Admin-Panels trägt dem Rechnung, indem es einen *Expert-in-the-Loop* Paradigma implementiert. Die Funktion ``ai_suggest_slot()`` fragt zwar die Logistic-Regression ab, um neue Produkte in Regale zu sortieren. Sinkt die Konfidenz (Probability) der Vorhersage jedoch unter einen definierten Schwellenwert (z. B. 80 %), erzwingt das UI ein manuelles Review durch den Menschen. Dies verhindert die schleichende Korrumpierung (Data Drift) der Supermarkt-Ontologie.

API-Referenz: UI Components & Callbacks
---------------------------------------

Der folgende Abschnitt dokumentiert die technischen Schnittstellen, Helper-Funktionen und asynchronen Controller-Routinen (Callbacks) der Frontend-Applikation.

.. automodule:: app
   :members:
   :undoc-members:
   :show-inheritance: