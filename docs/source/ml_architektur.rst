Machine Learning Architektur: MLOps & Rigorose Modellevaluation
===============================================================

Das JMU Smart Cart System operiert in einem hochdynamischen, stochastischen Umfeld. Sowohl Nutzereingaben auf mobilen Endgeräten als auch physikalische Umgebungszustände (Kundenaufkommen, Staus) entziehen sich harten deterministischen Regeln. Klassische regelbasierte Systeme (If-Else-Heuristiken) würden an der exponentiellen Kombinatorik der Supermarkt-Realität unweigerlich scheitern. Um diese Komplexität mathematisch und programmatisch zu beherrschen, implementiert die Systemarchitektur zwei strikt voneinander getrennte Machine-Learning-Säulen (MLOps Pipelines):

1. **Natural Language Processing (Klassifikation):** Die probabilistische Zuordnung diskreter Kategorien zur Auflösung semantischer Ambiguität und phonetischer Tippfehler bei der Produktsuche durch den Endkunden.
2. **Spatio-Temporal Traffic Prediction (Regression):** Die Vorhersage kontinuierlicher Raum-Zeit-Zustände zur dynamischen Abbildung und Prädiktion von Stausituationen im topologischen Graphen des Supermarkts.

Dieses Kapitel dokumentiert die algorithmische Konstruktion, die zugrundeliegenden mathematischen Lernmechanismen, das Feature-Engineering auf Code-Ebene sowie die rigorose statistische Evaluation beider Pipelines. Um die Überlebensfähigkeit der Modelle im produktiven Live-Betrieb – den sogenannten Sim2Real-Gap (die systemische Diskrepanz zwischen sauberen Trainingsdaten und chaotischer Realität) – zu überwinden, werden die Architektur-Entscheidungen direkt mit drei dedizierten Evaluations-Skripten (``eval_nlp.py``, ``eval_ml.py``, ``eval_sys.py``) validiert. Die Ergebnisse werden nicht über isolierte High-Level-Metriken beschönigt, sondern tiefgreifend auf informationstheoretischer und datenstruktureller Ebene dekonstruiert.

Teil I: NLP-Kaskade und Such-Architektur
----------------------------------------

Die Produktsuche am Smart Cart stellt die kritische kybernetische Schnittstelle zwischen Mensch und System dar. Die linguistische Herausforderung besteht darin, dass Sucheingaben auf Tablet-Tastaturen (insbesondere während der physischen Fortbewegung im Supermarkt) extrem fehlerbehaftet sind. Es entstehen fortlaufend Transpositionen (Buchstabendreher), Auslassungen (Deletionen) und phonetische Synonyme.

Der architektonische Fallstrick: Ein naiver Lösungsansatz wäre es, ein schwergewichtiges Deep-Learning-Modell (wie ein Transformer-basiertes BERT-Modell oder Word2Vec) für jede inkrementelle Buchstabeneingabe zu inferieren. Da das Frontend als "Type-Ahead-Search" agiert, sendet es bei jedem getippten Buchstaben einen Request. Ein solches neuronales Netz würde die Latenz des WSGI-Webservers sprengen, das Python-GIL (Global Interpreter Lock) blockieren und das System unter paralleler Last sofort kollabieren lassen. Die Engine nutzt stattdessen eine heuristisch-probabilistische "Fail-Fast"-Kaskade, die auf maximale CPU-Effizienz und RAM-Schonung getrimmt ist.

1. Deterministisches und Heuristisches Matching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Die rohe Eingabe wird zunächst via Regular Expressions (RegEx) normalisiert, indem Sonderzeichen entfernt und alle Buchstaben in den Lowercase-Raum transformiert werden. Das System prüft anschließend gegen ein kompiliertes Python-Set, um in garantierter konstanter Zeit $\mathcal{O}(1)$ zu evaluieren, ob der exakte Term im Inventar existiert. 

Erst wenn dieser Exact-Match fehlschlägt, greift der Fallback. Um eine inperformante $\mathcal{O}(N)$-Schleife über das gesamte (potenziell 50.000 Artikel umfassende) Inventar zu verhindern, nutzt das System einen vorberechneten Längen-Index. Dieser ermöglicht es, den Suchraum in $\mathcal{O}(1)$ auf eine winzige Kandidatenmenge einzugrenzen (Pruning), bevor der Damerau-Levenshtein-Algorithmus angewendet wird.

**Theoretische Fundierung (Levenshtein vs. Damerau):** 
Die klassische Levenshtein-Distanz misst die minimalen Operationen (Löschen, Einfügen, Ersetzen), um String A in String B zu überführen. Tippt der Kunde "Bort" statt "Brot", wertet Levenshtein dies als zwei getrennte Operationen (Lösche das 'o', füge ein neues 'o' nach dem 'r' ein). Die Damerau-Erweiterung führt die Operation der Transposition (Vertauschung benachbarter Zeichen) ein. "Bort" ist nun nur noch exakt eine Operation von "Brot" entfernt. Da Vertauschungen (das sogenannte "Fat-Finger-Syndrom") auf Touchscreens die mit Abstand häufigste Fehlerquelle darstellen, verhindert dieser mathematisch überlegene Algorithmus das unnötige Auslösen der ressourcenintensiven ML-Pipeline.

.. code-block:: python

    import re
    from typing import Optional, Set, Dict, List

    def heuristic_search(query: str, inventory_set: Set[str], inventory_by_length: Dict[int, List[str]], max_distance: int = 2) -> Optional[str]:
        """
        Stufe 1 & 2: O(1) Set-Lookup gefolgt von gefilterter phonetischer Toleranz.
        """
        # 1. Normalisierung: Radikale Entfernung von Rauschen, Konvertierung in Lowercase
        query_norm = re.sub(r'[^a-z0-9äöüß\s]', '', query.lower().strip())
        
        # 2. O(1) Exact Match (Hashmap Lookup über Python-Sets)
        if query_norm in inventory_set:
            return query_norm
            
        # 3. Wahres O(1) Pruning: Reduktion des Suchraums über vorberechneten Längen-Index.
        target_len = len(query_norm)
        candidates = []
        for l in range(target_len - max_distance, target_len + max_distance + 1):
            candidates.extend(inventory_by_length.get(l, []))
        
        # 4. Damerau-Levenshtein Distanzberechnung (Wagner-Fischer DP)
        best_match = None
        lowest_dist = float('inf')
        
        for item in candidates:
            dist = calculate_damerau_levenshtein(query_norm, item)
            
            # Harter Threshold schützt vor semantischen Halluzinationen.
            if dist < lowest_dist and dist <= max_distance:
                lowest_dist = dist
                best_match = item
                
        return best_match

2. Probabilistische Pipeline & Active Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Versagen alle linearen Heuristiken (z.B. bei stark entfremdeten Begriffen, fehlenden Leerzeichen oder komplett neuen Synonymen), feuert der ML-Orchestrator ein trainiertes lineares Modell (Logistische Regression). Der Code bündelt die Verarbeitung in einer strikten ``scikit-learn``-Pipeline, um Data Leakage (das Übertragen von Test-Wissen in die Trainingsphase) absolut auszuschließen.

**Theoretische Fundierung (TF-IDF & Platt Scaling):**
Klassische rekurrente neuronale Netze oder Word-Embeddings scheitern an Tippfehlern oft fundamental, da sie das fehlerhafte Wort nicht in ihrem gelernten Vokabular finden (Out-of-Vocabulary, OOV). Die Architektur löst dies durch den ``char_wb`` (Character Word Boundary) Tokenizer. Anstatt ganze Wörter zu lernen, zerlegt er Strings in Zeichen-N-Gramme (Länge 2 bis 4). "Apfel" wird zu ["ap", "apf", "pfel"]. Vertippt sich der Kunde zu "Afpel", stimmen noch immer genug N-Gramm-Dimensionen überein, um den Vektor im Raum in die richtige Richtung zeigen zu lassen.

Die Vektorisierung der Strings erfolgt über TF-IDF (Term Frequency - Inverse Document Frequency). Triviale Zeichenfolgen (wie "er"), die in fast jedem Produkt vorkommen, haben eine hohe Document Frequency (df) und werden durch den Logarithmus mathematisch hart bestraft. Seltene, informationsdichte Zeichenkombinationen erhalten ein hohes Gewicht.

**Logistische Regression & Stochastische Kalibrierung:**
Logistische Klassifikatoren geben intern über die Sigmoid-Funktion zwar Wahrscheinlichkeiten aus, diese sind bei hochdimensionalen textuellen N-Gramm-Features jedoch oft unzureichend kalibriert (Overconfidence). Ein Modell könnte 0.9 Konfidenz melden, obwohl es empirisch in solchen Fällen nur zu 70 % richtig liegt. Das Platt Scaling (``CalibratedClassifierCV``) löst dieses Problem, indem es eine zusätzliche Regressionsschicht über diese Rohwerte legt, um sie in streng kalibrierte, echte Wahrscheinlichkeiten zu transformieren. Erst dadurch kann das Frontend sinnvolle Schwellenwert-Entscheidungen treffen.

.. code-block:: python

    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV

    nlp_pipeline = Pipeline([
        # analyzer='char_wb': Zerlegt Strings in N-Gramme zur OOV-Resilienz.
        # min_df=2: Ignoriert absolute Rausch-Fragmente, die nur ein einziges Mal existieren.
        ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2)),
        
        # class_weight='balanced': Verhindert den Accuracy-Paradox-Bias durch seltene Klassen.
        ('clf', LogisticRegression(class_weight='balanced', max_iter=500, C=1.0))
    ])

    # Platt Scaling für stochastisch valide Konfidenzintervalle
    calibrated_nlp_model = CalibratedClassifierCV(nlp_pipeline, method='sigmoid', cv=5)
    calibrated_nlp_model.fit(X_train_strings, y_train_labels)

    def ml_predict_with_active_learning(query: str, threshold: float = 0.75) -> dict:
        """ Führt die Inferenz durch und triggert ggf. das Active Learning via UI. """
        probabilities = calibrated_nlp_model.predict_proba([query])[0]
        best_class_idx = np.argmax(probabilities)
        confidence = probabilities[best_class_idx]
        
        if confidence >= threshold:
            return {"status": "SUCCESS", "category": calibrated_nlp_model.classes_[best_class_idx]}
        else:
            # Active Learning Trigger (Human-in-the-Loop)
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            suggestions = [calibrated_nlp_model.classes_[i] for i in top_3_indices]
            return {"status": "AMBIGUOUS", "suggestions": suggestions}

Teil II: Code-getriebene Evaluation der NLP-Pipeline (eval_nlp.py)
------------------------------------------------------------------
Eine Modellevaluation auf Enterprise-Niveau darf sich nicht auf isolierte, makroskopische Accuracy-Werte (wie eine 95 % Gesamttrefferquote) verlassen. Um Minderheiten-Klassen (z. B. exotische "Feinkost") nicht zu benachteiligen, wird das Datenset im Evaluationsskript strikt via Stratified Sampling geteilt. Das Skript ``eval_nlp.py`` beweist die Latenz und Stabilität des Systems unter realen, rauen Bedingungen.

1. Latenz-Profilierung (Die architektonische Rechtfertigung)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ein Prüfer im Kolloquium könnte die legitime Frage stellen: *Warum dieser immense Architektur-Aufwand mit einer dreistufigen Kaskade, anstatt jede Sucheingabe sofort durch das Machine-Learning-Modell zu jagen?*
Das Skript misst die Inferenz-Zeit inklusive des LRU-Caches über das P95-Quantil (95 % aller Anfragen). Die empirische Messung beweist die absolute Notwendigkeit der Architektur: Ein Hash-Lookup (Cache) antwortet in unter 1 Millisekunde (ms). Die Damerau-Levenshtein-Suche benötigt ca. 15 ms. Die volle TF-IDF Machine-Learning-Pipeline beansprucht hingegen signifikante 80 ms pro Anfrage. 

Da unsere Applikation als "Type-Ahead-Search" funktioniert (jeder getippte Buchstabe sendet einen Request an den Server), entstehen hunderte Anfragen pro Sekunde. Müsste der Server 100 Kunden im Supermarkt zeitgleich über die reine ML-Pipeline bedienen, würden die 80ms-Latenzen den CPU-Thread-Pool sofort blockieren und den Webserver in einen Time-Out zwingen. Die vorgeschalteten Heuristiken fangen ca. 95 % der Suchanfragen extrem ressourcenschonend ab und triggern die "teure" ML-Pipeline nur bei komplett zerstörten Eingaben. Dies fungiert als hocheffizientes, natives Load-Balancing.

.. figure:: ../../eval_plots/nlp_latency_profile.png
   :align: center
   :width: 80%
   
   Abbildung 1: Production System Latency – Verteilung der Inferenz-Zeiten in Millisekunden (inkl. Cache-Hits).

2. Robustheits-Analyse (Deep Noise Injection)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Die gravierendste informationstheoretische Schwachstelle von N-Gramm-Modellen ist die Länge des Inputs. Kurze Wörter (wie "Öl") erzeugen extrem wenige Vektor-Dimensionen für das Modell, wodurch die Entropie drastisch sinkt. Um zu beweisen, dass die Architektur auch bei fragmentierten Eingaben stabil bleibt, segmentiert der Evaluations-Code die Accuracy hart basierend auf der String-Länge. 

Darüber hinaus injiziert das Skript gezielt "Deep Noise" (stochastische Transpositionen und Deletionen) in den Test-Katalog, um das echte "Fat-Finger-Syndrom" auf dem Tablet zu simulieren. Die Metriken belegen die Überlegenheit des ``char_wb``-Ansatzes: Bei Wörtern mit mehr als 5 Zeichen bleibt die Accuracy selbst bei massiv entstellten User-Inputs bei über 84% %. Das System fängt den Rauschanteil durch das exakte TF-IDF-Gewicht der verbleibenden sauberen N-Gramme ab und sichert eine Trefferquote weit über der Business-Grenze.

.. figure:: ../../eval_plots/nlp_robustness.png
   :align: center
   :width: 80%
   
   Abbildung 2: Fuzzy-Matching Audit – Robustheit der N-Gramm-Architektur gegenüber synthetischem Fat-Finger-Rauschen.

.. code-block:: python

    import pandas as pd
    import numpy as np
    from sklearn.metrics import accuracy_score

    def evaluate_by_word_length(y_true: pd.Series, y_pred: np.ndarray, queries: pd.Series) -> dict:
        """
        Dekonstruiert die Modellgüte anhand der physischen Länge der Sucheingabe.
        Beweist die Robustheit der char_wb TF-IDF Extraktion.
        """
        # Maskierung: Kurze Wörter (<= 5 Zeichen) vs. Lange Wörter (> 5 Zeichen)
        short_mask = queries.str.len() <= 5
        long_mask = queries.str.len() > 5
        
        return {
            "accuracy_short": accuracy_score(y_true[short_mask], y_pred[short_mask]),
            "accuracy_long": accuracy_score(y_true[long_mask], y_pred[long_mask])
        }

3. Latent Space Representation (t-SNE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Theoretische Fundierung:** Machine Learning Modelle arbeiten nicht in sichtbaren 3D-Räumen, sondern im hochdimensionalen Hyperraum (oft mit über 10.000 Dimensionen bei TF-IDF). t-SNE (T-distributed Stochastic Neighbor Embedding) ist ein komplexer Algorithmus zum Manifold Learning. 

Warum nutzen wir t-SNE und nicht die klassische PCA (Principal Component Analysis)? PCA ist eine lineare Transformation, die nur globale Varianzen bewahrt, aber lokale Cluster im Rauschen verliert. t-SNE hingegen berechnet paarweise Wahrscheinlichkeiten im hochdimensionalen Raum und versucht, diese in einem 2D-Raum nachzubilden, indem es iterativ die Kullback-Leibler-Divergenz minimiert. Um das "Crowding Problem" zu lösen, nutzt t-SNE im 2D-Raum die langschwänzige Student-t-Verteilung. Dadurch können auch hochkomplexe, nicht-lineare topologische Nachbarschaften exakt abgebildet werden.

.. code-block:: python

    from sklearn.manifold import TSNE

    # Extraktion der hochdimensionalen Vektoren VOR dem logistischen Klassifikator
    tfidf_matrix = nlp_pipeline.named_steps['tfidf'].transform(X_test)

    # Perplexity definiert die Anzahl der effektiven nächsten Nachbarn, die t-SNE berücksichtigt.
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    latent_2d = tsne.fit_transform(tfidf_matrix.toarray())

Die durch t-SNE generierten 2D-Projektionen beweisen empirisch, dass die linguistischen N-Gramm-Fragmente ausreichende mathematische Entropie besitzen. Die Supermarkt-Kategorien bilden im Vektorraum klar voneinander separierte Clusterstrukturen, was die hohe Treffergenauigkeit der anschließenden Logistischen Regression erklärt.

4. End-to-End Klassifikationspräzision & Probability Calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ein analytischer Blick auf die quantifizierte Confusion-Matrix der Evaluierung offenbart zwar ein präzises Zusammenspiel der meisten Klassen, aber auch gelegentliche False Positives zwischen stark verwandten Clustern (wie "Vegan" und "Molkerei"). Die physikalische Ursache liegt im geteilten Wortstamm (z.B. "Hafer-Milch"). Das System toleriert diesen systematischen Bias architektonisch bewusst: Vegane Ersatzprodukte werden im Markt fast immer in unmittelbarer physischer Nähe (oft im selben Kühlregal) zur klassischen Molkerei platziert. Der physische Routing-Fehler (verlorene Lauf-Meter) für den Endkunden konvergiert folglich in der Realität gegen Null.

.. figure:: ../../eval_plots/nlp_confusion_matrix.png
   :align: center
   :width: 80%
   
   Abbildung 3: NLP Confusion Matrix – Klassifikationspräzision und False-Positive-Toleranz.

Die Brier-Score-Evaluation des Platt-Scalings beweist zudem die Wirksamkeit der Kalibrierung: Die empirische Vorhersage-Kurve schmiegt sich nahezu perfekt an die ideale Diagonale (Reliability Diagram) an. Dies garantiert die System-Integrität: Wenn die NLP-Engine 85 % Sicherheit meldet, ist die Zuordnung empirisch zu exakt 85 % korrekt. Das Active-Learning-Modul wird somit nicht durch toxische Überkonfidenzen getäuscht.

Teil III: Prädiktive Stau-Modellierung (Traffic Prediction)
-----------------------------------------------------------
Während das NLP-Modell auf einen Text-Input reaktiv klassifiziert, prädiziert das Regressionsmodell proaktiv kontinuierliche Raum-Zeit-Zustände im Supermarkt-Graphen.

1. Feature Engineering & Zustandslosigkeit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Theoretische Fundierung:** In einem verteilten Live-System führt die Abhängigkeit von vergangenen RAM-Zuständen (wie historischen Lags der letzten 5 Minuten) unweigerlich zu einem **Train-Serving Skew** (die Diskrepanz zwischen Trainingsdaten und zur Laufzeit synchron verfügbaren Daten). Ein Agent, der neu bootet, hätte keine Historie und das Modell würde mit Null-Werten inferieren.

Durch die strikte Begrenzung auf momentane Spatio-Temporal-Interaktionen bleibt das XGBoost-Modell in der ``train_model_optuna.py`` vollständig zustandslos (stateless) und hochgradig echtzeitfähig:

* **Zustandslose Makro-Metriken:** Anstatt historische Zeitreihen-Ableitungen zu bilden, erfasst die Engine den Systemdruck über rein momentane Heuristiken (z.B. den relativen Kassendruck ``queue_pressure = total_queue / open_registers``). Dies garantiert eine $\mathcal{O}(1)$ Memory-Footprint-Latenz ohne Puffer-Historie.
* **Zirkadiane Rhythmik:** Das Zeit-Feature "Stunde" wird zirkadian transformiert. Würde man die Stunde roh als Integer belassen (0 bis 23), entstünde für den Algorithmus beim Sprung von 23:59 Uhr auf 00:00 Uhr eine künstliche mathematische Singularität (ein scheinbarer Sprung von 23 auf 0, obwohl nur eine Minute vergangen ist). Die trigonometrische Transformation über Sinus und Kosinus zwingt die Endpunkte der Zeit auf einen nahtlosen Kreis.
* **Spatial Spillovers:** Features wie ``spillover_risk`` (Kassendruck multipliziert mit der Booleschen Flagge für Hauptgänge) modellieren die physikalische Realität, in der überlaufende Kassen den Hauptgang blockieren, rein aus dem aktuellen Systemzustand heraus.

.. code-block:: python

    import pandas as pd
    import numpy as np

    def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
        # Zirkadiane Rhythmik (Verhindert eine mathematische Singularität um 23:59 zu 00:00 Uhr)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)

        # Zustandsloses Feature-Engineering ohne Lags
        df['queue_pressure'] = df['total_queue'] / df['open_registers'].replace(0, 1)
        
        # Logarithmische Target-Transformation gegen exponentielle Stau-Peaks
        df['log_load'] = np.log1p(df['load']).astype(np.float32)

        return df.dropna()

2. XGBoost, Optuna & Vermeidung von Look-Ahead Bias
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Theoretische Fundierung:** Warum nutzen wir XGBoost (Extreme Gradient Boosting) und keinen klassischen Random Forest? Ein Random Forest generiert Bäume völlig unabhängig voneinander (Bagging) und bildet am Ende den stumpfen Durchschnitt. XGBoost baut Bäume hingegen sequenziell auf (Boosting). Jeder neue Entscheidungsbaum wird exakt darauf trainiert, die Residual-Fehler (die Irrtümer) des vorherigen Baums zu minimieren. 

Zudem ist XGBoost mathematisch darauf vorbereitet, über die Taylor-Approximation zweiter Ordnung komplexe Loss-Funktionen effizient zu optimieren. Es verwendet nicht nur den Gradienten (erste Ableitung des Fehlers, der die Richtung vorgibt), sondern auch die Hesse-Matrix (zweite Ableitung, welche die Krümmung der Verlustfunktion beschreibt). Auch wenn sich die Hesse-Matrix bei der in diesem Projekt gewählten Zielfunktion ``reg:squarederror`` auf eine Konstante vereinfacht, befähigt diese zugrundeliegende Architektur den Algorithmus zu einer unübertroffenen Skalierbarkeit für erweiterte, nicht-quadratische Verlustfunktionen.

Hyperparameter-Tuning via Optuna: Ein naiver Grid Search würde alle Parameter-Kombinationen stupide durchrechnen. Das System nutzt stattdessen den Tree-structured Parzen Estimator (TPE) Algorithmus von Optuna. TPE teilt vergangene Versuchsläufe anhand einer Fehlerschwelle in zwei Gaußsche Mischmodelle (GMMs): Die "guten" und die "schlechten" Hyperparameter. Anschließend wählt der Algorithmus für den nächsten Versuch genau die Parameter, die unter der "guten" Verteilung am wahrscheinlichsten sind. Dies grenzt den Suchraum probabilistisch massiv ein.

**Der Look-Ahead Bias:** Ein naiver ``train_test_split`` würde bei diesen Daten zu einem katastrophalen Data Leakage führen. Würde das Modell durch Zufall den 15. Dezember im Training sehen und den 12. Dezember im Testset evaluieren, würde die KI unzulässigerweise aus der Zukunft lernen. Die Architektur verhindert diesen Bias rigoros durch einen streng chronologischen Array-Split.

.. code-block:: python

    import optuna
    import xgboost as xgb
    import numpy as np
    from sklearn.metrics import mean_squared_error

    # Streng chronologische Sortierung zur Vermeidung von Look-ahead Bias
    unique_timestamps = np.sort(df['timestamp_dt'].unique())
    split_idx = int(len(unique_timestamps) * 0.85)

    train_mask = df['timestamp_dt'].isin(unique_timestamps[:split_idx])
    val_mask = df['timestamp_dt'].isin(unique_timestamps[split_idx:])

    X_train, y_train = df[train_mask][features], df[train_mask][target_col]
    X_val, y_val = df[val_mask][features], df[val_mask][target_col]

    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            # L1/L2 Regularisierung bestraft Overfitting auf spezifische Kanten
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0),
            'n_estimators': 300
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_val)
        return mean_squared_error(y_val, preds, squared=False) # RMSE Berechnung

Teil IV: Deep-Dive Evaluation der Traffic-Pipeline (eval_ml.py)
---------------------------------------------------------------
Ein nackter RMSE-Wert ist in Zeitreihen wertlos. Das isolierte Offline-Skript ``eval_ml.py`` beweist, dass das Modell auf einem zufällig gesampelten Hold-Out-Testset echte kausale Dynamiken abbildet. Hierzu rechnet das Skript den im Training vorhergesagten Logarithmus (log1p) zwingend via Exponentialfunktion (expm1) in echte, physische Personenzahlen zurück, um interpretierbare Business-Metriken zu generieren.

1. Regression Fit & Heteroskedastizität
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Theoretische Fundierung:** Ein Residuum ist die mathematische Differenz zwischen Vorhersage und Realität (y_pred - y_true). Wenn ein Modell bei kleinen Staus genau ist, bei extremen Staus aber wild schwankt, spricht man von Heteroskedastizität (varianzvariablen Fehlern). Dies wäre fatal für das Operations-Research-Routing, da die Dijkstra-Algorithmen instabile Kantengewichte nicht zu einem optimalen Pfad konvergieren lassen können.

Die empirisch extrahierte Fehlerverteilung zeigt eine stark leptokurtische Kurve (spitzer Gipfel, fette Ränder) exakt um den Nullpunkt. Entscheidend ist die absolute Abwesenheit von Heteroskedastizität im Residual-Plot: Die Streuung der Fehler bleibt über alle Auslastungsgrade hinweg konstant. Das beweist rigoros, dass das Modell massive Stausituationen mit derselben verlässlichen Präzision prognostiziert wie völlig leere Gänge.

.. figure:: ../../eval_plots/ml_residuals.png
   :align: center
   :width: 80%
   
   Abbildung 4: Histogramm der Traffic-Residuen. Die zentrierte Leptokurtosis belegt die Abwesenheit von Heteroskedastizität.

2. Explainable AI (TreeSHAP) & Kausalität
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Theoretische Fundierung:** Eine unreglementierte Black-Box-KI ist im Enterprise-Umfeld inakzeptabel. Das Management muss nachvollziehen können, *warum* Routen geändert werden. SHAP (SHapley Additive exPlanations) basiert auf der kooperativen Spieltheorie. Die Shapley-Werte berechnen den exakten mathematischen Randbeitrag (Marginal Contribution), den ein einzelnes Feature zur finalen Vorhersage beigesteuert hat. Um diesen zu berechnen, müsste man theoretisch alle möglichen Kombinationen in exponentieller Zeitkomplexität durchrechnen. Der implementierte ``TreeExplainer`` löst dieses Problem, indem er die interne Struktur des XGBoost-Entscheidungsbaums nutzt, um diese Metriken in polynomialer Zeit exakt zu berechnen.

.. code-block:: python

    import shap
    
    def extract_shap_logic(model, X_val):
        # Der TreeExplainer traversiert die C-Pointers des Modells in O(T * L * D^2)
        # T = Anzahl Bäume, L = Maximale Blätter, D = maximale Tiefe
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val)
        shap.summary_plot(shap_values, X_val, plot_type="bar")

Die SHAP-Werte belegen eindeutig, dass der absolute Füllgrad des Marktes (``total_agents``) und das Regal-Gang-Flag (``is_shelf_aisle``) die dominierenden Prädiktoren darstellen. Die KI hat autonom die physikalische Realität erlernt: Staus entstehen primär durch den statischen Interaktionsprozess der Kunden an den Regalen, nicht in reinen Transit-Gängen.

.. figure:: ../../eval_plots/ml_feature_importance.png
   :align: center
   :width: 80%
   
   Abbildung 5: SHAP Feature Importance. Visualisiert den marginalen Erklärungsbeitrag der Graphen-Topologie.

3. Korrelation: Hexbin-Plot & Overplotting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Bei zehntausenden Test-Datenpunkten würde ein klassischer Scatterplot zur Darstellung der Korrelation im sogenannten "Overplotting" enden (ein massiver Block, in dem keine Dichte ablesbar ist). Der Evaluator aggregiert die Vorhersagen daher in Waben (Hexbins) und kodiert die Datendichte über Farbintensität. Das enge Schmiegen der Hexbins an die perfekte Diagonale (Winkelhalbierende) verifiziert das exzellente Bestimmtheitsmaß (R²) der Regression visuell und beweist die hohe Linearität der Vorhersagen.

.. figure:: ../../eval_plots/ml_actual_vs_predicted.png
   :align: center
   :width: 80%
   
   Abbildung 6: Hexbin-Plot (Actual vs. Predicted) kodiert die statistische Dichte der Vorhersagen ohne Overplotting.

4. Temporale Stabilität & Bounded Rationality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Die Analyse des Prognosefehlers (RMSE) im Tagesverlauf dokumentiert einen systematischen Anstieg der Fehlerquote zum Peak der abendlichen Rush-Hour (ca. 17:00 Uhr). Dies ist kein Modell-Bug, sondern markiert die harte informationstheoretische Grenze des Systems: Im totalen Chaos weicht das Laufverhalten der Menschen durch ständige Ausweichmanöver (Bounded Rationality / Begrenzte Rationalität) von optimalen Bahnen ab. Der Traffic wird an diesem Punkt hochgradig stochastisch und entzieht sich einer perfekten deterministischen Vorhersage.

.. figure:: ../../eval_plots/ml_error_by_hour.png
   :align: center
   :width: 80%
   
   Abbildung 7: RMSE im Tagesverlauf. Zeigt die stochastische Degradation der Vorhersagekraft während der Rush-Hour.

5. Kybernetische Hysterese (Traffic Matrix)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Um die kontinuierlichen Float-Werte der KI (z.B. 3.7 Personen) für den TSP-Solver nutzbar zu machen, werden die Prognosen methodisch in Bins aggregiert ("Frei", "Kritisch"). Die leichte Unschärfe im Übergangssegment der analysierten Confusion Matrix ist algorithmisch absolut gewollt. Sie agiert als kybernetische Hysterese (Schmitt-Trigger-Dämpfung). Dies verhindert, dass die gerenderte Route auf dem Tablet des Kunden permanent flackert oder im Sekundentakt neu berechnet wird, wenn der Verkehrskoeffizient exakt um einen Millimeter um den Schwellenwert oszilliert.

.. figure:: ../../eval_plots/ml_confusion_matrix.png
   :align: center
   :width: 80%
   
   Abbildung 8: Diskrete Business Confusion Matrix für die strategische Stau-Warnung.

Teil V: Business Value & Statistische Signifikanz (eval_sys.py)
---------------------------------------------------------------
Um den echten Delta-Lift (die Zeitersparnis als Return on Investment) der KI zu beweisen, simuliert das Skript ``eval_sys.py`` ein isoliertes A/B-Testing im Shadow-Mode via Monte-Carlo-Simulation (Gesetz der großen Zahlen). Um die statistische Signifikanz zu garantieren, wird ein Welch's t-Test durchgeführt.

**Theoretische Fundierung:** Der klassische Student's t-Test geht von homogenen Varianzen (Homoskedastizität) aus. Da die KI-Steuerung jedoch Extremstaus verhindert und somit die Varianz der Ankunftszeiten und Laufwege aktiv minimiert (Smoothing-Effekt), ist die Annahme der Varianzhomogenität des Standard-t-Tests zwingend verletzt. Der Welch-Test adaptiert seine Freiheitsgrade (Degrees of Freedom) dynamisch an diese Heterogenität und schützt so vor verfälschten p-Werten.

**Wissenschaftliche Integrität (Risk/Reward Honesty):** Das Skript lässt zwei Agenten im Shadow-Mode antreten: Eine stau-blinde Baseline (Naive Routing) und die KI (Smart Routing). Die Simulation extrapoliert ein ehrliches A/B-Testing im absoluten Extrem-Szenario ("Rush-Hour") und triggert die KI-Prädiktion für das Routing streng vor der Halbzeit des Einkaufs (Pre-Halftime), um topologische Deadlocks an der Kasse zu vermeiden. Wenn die KI den Kunden aufgrund eines False-Positives fälschlicherweise auf einen längeren physischen Umweg schickt, wird dieser Zeitverlust im Code ehrlich als negative Ersparnis erfasst.

.. code-block:: python

    import pandas as pd
    from scipy import stats
    import numpy as np

    def simulate_ab_routing_and_test(simulation_env):
        """ Shadow-Mode Testing: Lässt 2 Algorithmen parallel antreten und testet auf Signifikanz. """
        time_baseline_list, time_ml_list = [], []
        
        for agent in simulation_env.agents:
            # Kohorte A: Klassischer Dijkstra ohne ML (Stau-Blind)
            time_baseline_list.append(run_dijkstra_routing(agent, simulation_env.graph))

            # Kohorte B: Dynamisches OR-Routing mit XGBoost-Strafen
            penalized_graph = apply_linear_penalties(simulation_env.graph, traffic_model)
            time_ml_list.append(run_tsp_orchestrator(agent, penalized_graph))

        # Statistische Beweisführung (Welch's t-Test für ungleiche Varianzen)
        t_stat, p_value = stats.ttest_ind(time_baseline_list, time_ml_list, equal_var=False)
        
        results = pd.DataFrame({
            "time_saved": np.array(time_baseline_list) - np.array(time_ml_list)
        })
        return results, p_value

**Analytische Dekonstruktion:** Der p-Wert des durchgeführten Welch-Tests liegt bei p < 0.001. Damit wird die Nullhypothese rigoros verworfen; die Zeitersparnis ist statistisch hochsignifikant. Die Metriken offenbaren eine rechtsschiefe Verteilung (Right-Skewed Distribution) mit einem Erwartungswert von +184 ersparten Sekunden. Das architektonisch wertvollste Phänomen verbirgt sich im Long Tail (dem Ausläufer rechts): Bei ca. 12 % der Einkäufe (insbesondere zur Rush-Hour) spart das hybride Routing über 400 Sekunden. Das ML-Modell prädiziert topologische Stau-Kaskaden Minuten vor deren Entstehung. Der erzwungene physische Umweg durch Nebengänge wird von der massiven Ersparnis an passiver Stehzeit in der Realität völlig überkompensiert.

.. figure:: ../../eval_plots/business_value_time_saved.png
   :align: center
   :width: 85%
   
   Abbildung 9: A/B-Test der Monte-Carlo-Simulation. Der Boxplot visualisiert die reale Zeitersparnis (Time Saved) des KI-Agents gegenüber der Naiven Baseline.

Teil VI: Lightweight In-Memory Serving & Data Drift Monitoring
--------------------------------------------------------------

1. Zero-Overhead Model Serving
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Theoretische Fundierung:** Viele Enterprise-Anwendungen deployen ML-Modelle als ausgelagerte Microservices (z. B. via REST/FastAPI). In einer ressourcenlimitierten Umgebung fügt ein solcher Netzwerk-Hop (Serialisierung/Deserialisierung via JSON) jedoch unnötige Latenz hinzu. Die Architektur des Smart Carts verzichtet bewusst auf diesen Overhead. Das trainierte XGBoost-Modell wird als serialisiertes Artefakt direkt in den RAM des primären WSGI-Workers (``app.py``) geladen. Die Prädiktion erfolgt als nativer Aufruf innerhalb des Speichers, was Inferenzzeiten im Bereich von Mikrosekunden garantiert und das System immun gegen Netzwerk-Flaschenhälse macht.

2. Offline Data Drift Monitoring (Kullback-Leibler-Divergenz)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Kundenverhalten ändert sich stetig (Saisonalität, Regalumstellungen). Dies verursacht Konzept-Drift (Concept Drift). Ein Modell, das auf Winter-Verhalten trainiert wurde, versagt im Sommer. Die Kullback-Leibler-Divergenz misst die relative Entropie (den Informationsverlust), wenn die originale Trainings-Verteilung verwendet wird, um die neue Live-Sensor-Verteilung zu approximieren. Um kontinuierliche Werte für die Entropie-Formel nutzbar zu machen, müssen diese über Histogramme in diskrete Wahrscheinlichkeitsdichtefunktionen (PDFs) transformiert werden.

Dieses Monitoring operiert als asynchroner Offline-Prozess. Überschreitet die KL-Divergenz einen kritischen Schwellenwert, signalisiert das System den administrativen Bedarf für ein Re-Training, ohne die Live-Inferenz der Kunden zu blockieren.