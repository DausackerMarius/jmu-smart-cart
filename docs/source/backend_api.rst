Backend-Architektur & System-Design
===================================

Die theoretische Konzeption von komplexen Routing-Algorithmen und Machine-Learning-Modellen ist in der modernen Informatik nur die halbe Miete. Die eigentliche ingenieurtechnische Herausforderung besteht darin, diese rechenintensiven mathematischen Modelle in eine serverseitige Architektur zu gießen, die performant, fehlerresistent und für zukünftige Entwickler wartbar bleibt. 

Dieses Kapitel widmet sich dem "Maschinenraum" des JMU Smart Cart Projekts. Es dokumentiert nicht nur die reinen API-Schnittstellen, sondern vor allem die menschlichen und architektonischen Designentscheidungen, die getroffen werden mussten, um eine hochkomplexe Supermarkt-Simulation in Echtzeit im Browser lauffähig zu machen.

1. Das architektonische Fundament: Die bimodale Trennung
--------------------------------------------------------

Als das System konzipiert wurde, standen wir vor einem massiven Performance-Problem: Der Download der Bundeslebensmittelschlüssel-Datenbank (BLS), das Bereinigen von Tausenden Text-Strings über reguläre Ausdrücke und die anschließende Neu-Allokation des Graphen dauern im Schnitt mehrere Sekunden bis Minuten. Würde dieser Prozess bei jeder Nutzeranfrage (oder auch nur bei jedem Server-Neustart im Live-Betrieb) getriggert, wäre das System für Endanwender unbrauchbar.

Die Lösung ist eine strikte **bimodale Systemarchitektur** (Separation of Concerns). Das Backend ist in zwei hermetisch voneinander abgeriegelte Lebenszyklen unterteilt:

1.1 Phase 1: Offline Data Engineering (Der "Pre-Flight")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Gesteuert durch die Datei ``generate_data_driven_store.py``, fungiert diese Säule als die "Fabrik" des Systems. Sie wird ausschließlich offline durch einen Administrator ausgeführt, bevor das System überhaupt ans Netz geht. 
Diese Pipeline arbeitet sich durch den rechenintensiven "Schmutz" der realen Daten: Sie lädt Excel-Dateien herunter, jagt Tausende Produkte durch den ``TextSanitizer``, um störende B2B-Begriffe zu entfernen, und nutzt das Sainte-Laguë-Verfahren, um die Produkte mathematisch fair auf die verfügbaren Regal-Knoten zu verteilen. 

Das Endprodukt dieser Fabrik ist hochgradig optimiert: Der ``MasterOrchestrator`` friert den fertigen, validen Zustand des Supermarkts in zwei leichtgewichtige, statische JSON-Dateien ein (``products.json`` für das Inventar und ``routing_config.json`` für die topologische Verortung).

1.2 Phase 2: Die Live-Engine (In-Memory Store Core)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Gesteuert durch die ``model.py``, ist dies der eigentliche Live-Server, mit dem der Nutzer interagiert. Das architektonische Gebot für diese Phase lautet: **Keine Festplatten-I/O-Operationen zur Laufzeit.** Festplattenzugriffe (selbst auf SSDs) sind im Vergleich zu Arbeitsspeicher-Operationen extrem langsam. Daher liest die ``SystemConfig`` beim Boot-Vorgang des Webservers die zuvor generierten JSON-Dateien sowie die kompilierten Machine-Learning-Modelle (Pickle-Dateien, ``.pkl``) exakt einmalig in den RAM ein. 

Wenn der Nutzer nun im Frontend einen Weg sucht, operiert das gesamte Backend (Suche, ML-Inferenz, TSP-Routing) ausschließlich im Arbeitsspeicher. Dadurch werden Latenzen im Bereich von wenigen Millisekunden realisiert.

2. Bewältigung der Komplexität: Software Design Patterns
--------------------------------------------------------

In einem Projekt, das Graphentheorie, Stochastik und KI vereint, neigt der Quellcode schnell dazu, in unwartbaren "Spaghetti-Code" zu degenerieren. Um eine akademisch exzellente Code-Qualität (Clean Code) zu gewährleisten, haben wir das System um vier klassische Entwurfsmuster (Design Patterns) der Softwaretechnik herum gebaut. Jedes Muster löst ein spezifisches Problem unserer Architektur:

2.1 Das Strategy Pattern (Dynamisches Routing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*Das Problem:* Wir wussten, dass wir für kleine Warenkörbe den perfekten Held-Karp-Algorithmus nutzen wollen, für große Einkaufslisten aber auf Metaheuristiken (Genetische Algorithmen, Simulated Annealing) ausweichen müssen, da sonst der Server abstürzt. Hätten wir dies mit verschachtelten ``if/else``-Blöcken im Hauptcode gelöst, wäre das System bei der Integration neuer Algorithmen kollabiert.
*Die Lösung:* Wir haben eine abstrakte Basisklasse ``RoutingStrategy`` geschaffen, die lediglich verspricht, dass es eine Funktion ``solve()`` gibt. Jeder spezifische Algorithmus (z. B. der ``AntColonySolver``) erbt von dieser Basis und implementiert seine eigene Mathematik. Die Hauptfunktion ``calculate_hybrid_route()`` muss nun nicht mehr wissen, *wie* der Algorithmus rechnet; sie tauscht die Rechenstrategie einfach dynamisch zur Laufzeit aus. Das macht den Code extrem modular.

2.2 Das Singleton Pattern (Ressourcen-Schutz)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*Das Problem:* Machine-Learning-Modelle wie XGBoost oder große TF-IDF-Matrizen belegen hunderte Megabyte im Arbeitsspeicher. Würde bei jedem Klick eines Nutzers im Dashboard eine neue Instanz der ``MLOpsEngine`` oder des ``SearchKernel`` erzeugt, käme es binnen Sekunden zu einem fatalen "Out of Memory" (OOM) Server-Crash.
*Die Lösung:* Wir nutzen das Singleton-Muster über ein statisches ``_instance``-Attribut in den Klassen. Wenn Nutzer A eine Suchanfrage stellt, wird die Engine in den RAM geladen. Wenn Nutzer B zeitgleich sucht, erkennt das System, dass die Instanz bereits existiert, und leitet die Anfrage ressourcenschonend an das geteilte Modell im Arbeitsspeicher weiter.

2.3 Das Facade Pattern (Stochastische Abstraktion)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*Das Problem:* Das Frontend (die Benutzeroberfläche) muss dem Nutzer lediglich eine Wartezeit in Minuten anzeigen. Es sollte aber absolut nichts davon wissen müssen, wie Kendalls-Notation, Poisson-Verteilungen oder die Differentialgleichungen eines M/M/1/K-Warteschlangenmodells funktionieren.
*Die Lösung:* Wir haben die ``QueuingModelFacade`` als architektonischen "Türsteher" programmiert. Diese Klasse bietet dem Frontend eine extrem simple Schnittstelle: ``calculate_wait_time(stunde, kassen_id)``. Intern ruft die Fassade das hochkomplexe ``EnterpriseQueuingModel`` auf, führt die Stochastik durch und gibt einen sauberen Float-Wert zurück. Das Frontend bleibt dadurch dumm, leichtgewichtig und fokussiert auf das Rendering.

2.4 Das Decorator Pattern (Elegantes Profiling)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*Das Problem:* Um die Latenzen der verschiedenen Routen-Algorithmen für unsere wissenschaftliche Auswertung zu messen, mussten wir die Rechenzeit stoppen. Hätten wir in jeden Algorithmus ``start = time.time()`` und ``print(ende - start)`` geschrieben, hätten wir unsere mathematische Geschäftslogik mit profanen Logging-Befehlen verunreinigt.
*Die Lösung:* Wir nutzen Python-Decorators (``@execution_profiler``). Dieser Wrapper legt sich wie eine unsichtbare Hülle um jede gewünschte Funktion. Er fängt den Funktionsaufruf ab, startet eine Stoppuhr im Hintergrund, führt die Kernlogik aus, stoppt die Zeit und schreibt die Metriken ins Log. Der mathematische Quellcode bleibt dadurch zu 100 % sauber und fokussiert.

3. Fehlerbehandlung und Resilienz
---------------------------------
Ein System für den Live-Betrieb muss Fehler abfangen, bevor sie den Server zum Absturz bringen. Die Backend-Architektur verzichtet auf generische ``Exception``-Fänge (welche die tatsächliche Fehlerursache maskieren würden) und implementiert stattdessen eine eigene Ausnahme-Hierarchie, angeführt von der ``StoreBackendException``. 

Spezifische Fehler wie ein ``GraphTopologyError`` (z. B. wenn versucht wird, eine Kante zu routen, die durch ein neu platziertes Regal physisch blockiert wurde) oder ein ``ConfigurationError`` (fehlende ML-Modell-Dateien) werden auf unterster Ebene geworfen, sicher geloggt und in für das Frontend verdauliche Fehlermeldungen übersetzt, sodass die UI nicht einfriert.

---

API-Referenz: Säule I (Data Engineering Pipeline)
-------------------------------------------------
Der nachfolgende Bereich wird automatisiert aus den Quellcode-Docstrings generiert und dokumentiert die Methodensignaturen der ETL-Architektur.

.. automodule:: generate_data_driven_store
   :members:
   :undoc-members:
   :show-inheritance:

API-Referenz: Säule II (Store Core & Operations Research)
---------------------------------------------------------
Der nachfolgende Bereich dokumentiert die Klassenstrukturen der In-Memory-Engine, der ML-Integration und der TSP-Wegfindung.

.. automodule:: model
   :members:
   :undoc-members:
   :show-inheritance: