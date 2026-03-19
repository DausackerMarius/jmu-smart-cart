Machine Learning Architektur: NLP & Prädiktive Modellierung
===========================================================

Das JMU Smart Cart System operiert in einem hochdynamischen Umfeld, in dem sowohl Nutzereingaben als auch Umgebungszustände (Kundenaufkommen) stochastischen Schwankungen unterliegen. Um diese Komplexität mathematisch zu beherrschen, implementiert die Architektur zwei strikt voneinander getrennte Machine-Learning-Säulen, die unterschiedliche Problemklassen bedienen:

1. **Natural Language Processing (Klassifikation):** Die Zuordnung diskreter Kategorien zur Auflösung semantischer Ambiguität bei der Produktsuche.
2. **Traffic Prediction (Regression):** Die Vorhersage kontinuierlicher Werte zur dynamischen Abbildung von Stausituationen im Supermarkt-Graphen.

Teil I: NLP-Kaskade und Such-Architektur
----------------------------------------

Die Produktsuche stellt die kritische Schnittstelle zum Nutzer dar. Die Herausforderung besteht darin, dass Eingaben auf mobilen Endgeräten stark fehlerbehaftet sind (Tippfehler, Synonyme, Dialekte). Um maximale Performanz bei minimaler Rechenlast zu garantieren, folgt die Suche einer hybriden "Fail-Fast"-Logik in drei Eskalationsstufen:

1. Ebene 1: Deterministisches Exact-Matching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Die Eingabe wird zunächst linguistisch normalisiert. Dies umfasst **Unicode-Normalisierung** (Auflösung von Sonderzeichen) und **Case-Folding** (konsequente Kleinschreibung). Anschließend erfolgt ein Hash-basiertes Exact-Matching. Dies garantiert eine Zeitkomplexität von $\mathcal{O}(1)$ – das System findet fehlerfreie Eingaben ("Apfel") sofort, ohne Rechenleistung für komplexe Modelle zu verschwenden.

2. Ebene 2: Heuristisches Fuzzy-Matching (Phonetische Diskriminierung)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Schlägt die exakte Suche fehl, greifen zwei komplementäre Heuristiken, um Rechtschreibfehler abzufangen:

* **Kölner Phonetik:** Dieser Algorithmus wandelt Buchstabenfolgen in einen numerischen Laut-Code um und ist speziell für den deutschen Sprachraum optimiert. Er bildet gleich klingende Konsonanten (z. B. "ph" und "f") auf denselben Wert ab. So wird aus dem Tippfehler "Mayonäse" und dem Original "Mayonnaise" derselbe phonetische Hash.
* **Damerau-Levenshtein-Distanz:** Diese Metrik berechnet die minimale Anzahl an atomaren Editieroperationen (Einfügen, Löschen, Ersetzen), die nötig sind, um zwei Strings anzugleichen. Der entscheidende Vorteil gegenüber der klassischen Levenshtein-Distanz ist die explizite Berücksichtigung von **Transpositionen** (Vertauschung zweier benachbarter Zeichen, z. B. "Brot" -> "Bort"), was den häufigsten Fehler auf Smartphone-Tastaturen darstellt.

3. Ebene 3: Probabilistisches Feature-Engineering (ML-Fallback)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Wenn heuristische Regeln versagen (z. B. bei stark fragmentierten Wörtern oder unbekannten Markennamen), wird ein Machine-Learning-Modell aktiviert.

* **Charakter-N-Gramme & TF-IDF:** Anstatt das Wort als Ganzes zu betrachten, wird es in überlappende Silbenfragmente ($n=2,3$) zerlegt. Das Wort "Schokolade" wird zu ["sch", "cho", "oko", ...]. Die **TF-IDF-Gewichtung** (Term Frequency-Inverse Document Frequency) sorgt dafür, dass seltene, hochinformative Fragmente stärker gewichtet werden als triviale Endungen (wie "-en" oder "-er"). 
* **Logistic Regression:** Diese N-Gramm-Vektoren werden in einen hochdimensionalen Raum projiziert. Ein logistischer Regressor zieht mathematische Entscheidungsgrenzen (Hyperplanes) zwischen den Kategorien und berechnet über eine Softmax-Funktion die Wahrscheinlichkeit, in welches Regal das Produkt gehört.

Evaluation des NLP-Modells
--------------------------

Um die Praxistauglichkeit zu beweisen, wurde die NLP-Pipeline in drei Dimensionen evaluiert: Robustheit, Präzision und Latenz.

**1. Robustheit gegenüber Störrauschen (Deep Typos)**

.. image:: ../../eval_plots/nlp_robustness.png
   :width: 600px
   :align: center
   :alt: Evaluation der Machine Learning Robustheit

*Diskussion:* Ein zentraler Kritikpunkt an wortbasierten Suchmodellen ist ihr kompletter Ausfall bei unbekannten Wörtern (Out-of-Vocabulary Problem). Das Diagramm widerlegt dies für unsere Architektur: Während die Accuracy (Klassifikationsgüte) bei sauberen Daten bei 84,2 % liegt, sinkt sie bei künstlich induzierten, schweren Tippfehlern ("Deep Typos") lediglich auf 73,9 %. Das Modell überlebt dieses Störrauschen, weil die Charakter-N-Gramm-Vektorisierung die semantische Bedeutung aus den intakten Silben-Fragmenten rekonstruiert, selbst wenn das Wort als Ganzes zerstört wurde.

**2. Klassifikationspräzision (Confusion Matrix)**

.. image:: ../../eval_plots/nlp_confusion_matrix.png
   :width: 700px
   :align: center
   :alt: End-to-End Confusion Matrix

*Diskussion:* Die Confusion Matrix visualisiert das Verhältnis von tatsächlichen Werten (Ground Truth) zu den Vorhersagen der KI. Die stark ausgeprägte Diagonale (True Positives) bestätigt eine exzellente Trennschärfe über alle Kategorien hinweg. Vernachlässigbare Fehlklassifikationen (False Positives/Negatives) treten primär bei semantisch stark überlappenden Clustern auf (z. B. "Vegan" vs. "Molkerei"). Dies ist systemimmanent, da Ersatzprodukte identische linguistische Wortstämme (z. B. "Hafer-Milch") nutzen. Für das operative Routing ist dies jedoch unkritisch, da sich diese Warengruppen physisch meist in unmittelbarer Proximität (im selben Gang) befinden.

**3. Systemlatenz und Echtzeit-Garantie (Production Latency)**

.. image:: ../../eval_plots/nlp_latency_profile.png
   :width: 700px
   :align: center
   :alt: System Latenz Histogramm

*Diskussion:* Da die Suche synchron zur Nutzereingabe im Frontend agiert, ist die Latenz (Verzögerungszeit) kritisch. Das Latenzprofil zeigt eine bimodale Verteilung (zwei Maxima). Das erste Maximum (nahe 0 ms) repräsentiert die O(1)-Lookups aus Ebene 1 und 2. Das zweite, flachere Maximum zeigt die rechenintensivere ML-Inferenz aus Ebene 3. Durch den Einsatz eines LRU-Caches (Least Recently Used), der häufige Suchen im Arbeitsspeicher hält, liegt die durchschnittliche Latenz bei nur 1,43 ms. Selbst das 95. Perzentil (P95) überschreitet die 3,0-ms-Grenze nicht. Damit ist das System uneingeschränkt echtzeitfähig.


Teil II: Prädiktive Stau-Modellierung (Traffic Prediction)
----------------------------------------------------------

Während das NLP-Modell auf Nutzer-Inputs reagiert, agiert der ``TrafficPredictor`` proaktiv. Um das TSP-Routing dynamisch zu optimieren, muss das System prädiktiv (vorausschauend) ermitteln, wie viele Personen sich auf einer bestimmten Kante (Gang) des Graphen befinden. Hierbei handelt es sich um ein **multivariates Regressionsproblem**.

Das Modell lernt auf Basis von historischen Trainingsdaten Korrelationen zwischen räumlichen Eigenschaften (Befindet sich an dieser Kante ein Regal?) und zeitlichen Dimensionen (Sinus/Kosinus-Transformation der Uhrzeit zur Abbildung zirkadianer Rhythmen).

Evaluation des Traffic-Modells
------------------------------

**1. Fehlerverteilung der Vorhersage (Residuals)**

.. image:: ../../eval_plots/ml_residuals.png
   :width: 600px
   :align: center
   :alt: Fehlerverteilung Residuals

*Diskussion:* Die Residualanalyse betrachtet die Differenz zwischen dem realen Kundenaufkommen und der KI-Vorhersage (Fehlerterm). Der Plot zeigt eine stark **leptokurtische (steilgipflige) Verteilung** exakt um den Nullpunkt. Dies bedeutet, dass die absolute Mehrheit der Vorhersagen fehlerfrei oder extrem nah am wahren Wert liegt. Das Bestimmtheitsmaß $R^2 = 0.920$ bestätigt, dass die ausgewählten Features 92 % der Varianz des Personenaufkommens erklären. Zudem fehlt eine asymmetrische Schiefe, was beweist, dass das Modell keinen systematischen Bias (permanente Über- oder Unterschätzung) aufweist.

**2. Feature Importance (Einflussfaktoren)**

.. image:: ../../eval_plots/ml_feature_importance.png
   :width: 600px
   :align: center
   :alt: Feature Importance

*Diskussion:* Die "Gain"-Metrik misst, wie stark eine bestimmte Variable den Fehler (Entropie) des Modells reduziert. Die Analyse zeigt, dass ``total_agents`` (die absolute Anzahl an Kunden im Markt) und ``is_shelf_aisle`` (Befindet sich hier ein Regal?) die mit Abstand wichtigsten Prädiktoren sind. Dies ist ein hervorragendes Indiz für die kausale Logik des Modells: Die KI hat autonom gelernt, dass Staus primär durch den Scan- und Steh-Prozess an Regalen entstehen und nicht durch das reine Vorbeilaufen auf den breiten Hauptverkehrsachsen (Main Aisles).

**3. Zuverlässigkeit der Stau-Erkennung (Business Matrix)**

.. image:: ../../eval_plots/ml_confusion_matrix.png
   :width: 600px
   :align: center
   :alt: Business Matrix

*Diskussion:* Da kontinuierliche Regressionswerte für das Routing in Zeitstrafen (Penalties) umgerechnet werden, evaluiert diese "Business Matrix" die binäre Verwertbarkeit. Die Matrix belegt, dass "freie Gänge" (0-2 Personen) zu 99,4 % und "kritische Staus" (>5 Personen) zu 92,1 % korrekt in ihre jeweiligen Klassen (Bins) eingeteilt werden. Die leichte Unschärfe im Mittelsegment (3-5 Personen) ist algorithmisch gewollt: Sie fungiert als Dämpfungsglied, um ein oszillierendes "Flackern" der berechneten TSP-Routen bei kleinsten Bewegungen einzelner Kunden zu verhindern.

**4. Modellgüte im Tagesverlauf (Temporale Stabilität)**

.. image:: ../../eval_plots/ml_error_by_hour.png
   :width: 600px
   :align: center
   :alt: Modellgüte Tagesverlauf

*Diskussion:* Der Plot visualisiert den mittleren absoluten Fehler (Mean Absolute Error, MAE) als Funktion der Tageszeit. Es ist deutlich erkennbar, dass der Fehlerwert ab 16:00 Uhr und insbesondere zum Peak um 17:00 Uhr ansteigt. Dies korreliert mit dem klassischen "Feierabend-Rush". In dieser Phase wird das System von einer massiven, extrem komprimierten Kundenanzahl geflutet. Das Laufverhalten wird dadurch hochgradig stochastisch und chaotisch (Kunden weichen unvorhersehbar aus). Der steigende MAE belegt somit keine Schwäche des Modells, sondern markiert die informationstheoretische Grenze der Vorhersagbarkeit von Massenpsychologie im Einzelhandel.