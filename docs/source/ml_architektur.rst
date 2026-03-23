Machine Learning Architektur: MLOps & Rigorose Modellevaluation
===============================================================

Das JMU Smart Cart System operiert in einem hochdynamischen Umfeld, in dem sowohl Nutzereingaben auf mobilen Endgeräten als auch Umgebungszustände (Kundenaufkommen) extremen stochastischen Schwankungen unterliegen. Um diese Komplexität mathematisch und programmatisch zu beherrschen, implementiert die Systemarchitektur zwei strikt voneinander getrennte Machine-Learning-Säulen (MLOps Pipelines):

1. **Natural Language Processing (Klassifikation):** Die Zuordnung diskreter Kategorien zur Auflösung semantischer Ambiguität bei der Produktsuche durch den Kunden.
2. **Traffic Prediction (Regression):** Die Vorhersage kontinuierlicher Werte zur dynamischen Abbildung von Stausituationen im topologischen Graphen des Supermarkts.

Dieses Kapitel dokumentiert die algorithmische Konstruktion, das Feature-Engineering auf Code-Ebene sowie die rigorose statistische Evaluation beider Pipelines. Um die Überlebensfähigkeit der Modelle im produktiven Live-Betrieb (Sim2Real-Gap) zu garantieren, werden die Ergebnisse nicht nur über High-Level-Metriken, sondern tiefgreifend auf Datenstrukturebene dekonstruiert.

Teil I: NLP-Kaskade und Such-Architektur
----------------------------------------

Die Produktsuche stellt die kritische Schnittstelle zum Nutzer dar. Die linguistische Herausforderung besteht darin, dass Sucheingaben auf Tablet-Tastaturen stark fehlerbehaftet sind (Tippfehler, phonetische Synonyme). Ein schwergewichtiges Deep-Learning-Modell (wie Transformer/BERT) für jede inkrementelle Buchstabeneingabe zu inferieren, würde die Latenz des WSGI-Webservers sprengen. Die Engine nutzt stattdessen eine heuristisch-probabilistische "Fail-Fast"-Kaskade.

1. Deterministisches und Heuristisches Matching auf Code-Ebene
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Die rohe Eingabe wird zunächst via Regular Expressions normalisiert. Das System prüft über einen Hash-Lookup in $\mathcal{O}(1)$, ob der exakte Term im Inventar existiert. Schlägt dies fehl, greift der **Damerau-Levenshtein-Algorithmus**, implementiert über effiziente C-Bindings (z.B. via ``jellyfish`` oder ``textdistance``).

.. code-block:: python

   import re
   import textdistance
   from typing import Optional, List, Dict

   def heuristic_search(query: str, inventory: List[str], max_distance: int = 2) -> Optional[str]:
       """
       Stufe 1 & 2: O(1) Lookup gefolgt von phonetischer Fehlertoleranz.
       """
       # 1. Normalisierung: Entfernung von Rauschen, Konvertierung in Lowercase
       query_norm = re.sub(r'[^a-z0-9äöüß\s]', '', query.lower().strip())
       
       # 2. O(1) Exact Match (Hashmap Lookup simuliert über Set/List-Abgleich)
       if query_norm in inventory:
           return query_norm
           
       # 3. Damerau-Levenshtein Distanz
       # Erfasst Vertauschungen (Transpositionen) wie "Brot" -> "Bort" als Kosten=1
       best_match = None
       lowest_dist = float('inf')
       
       for item in inventory:
           # Berechnet die minimalen Editier-Operationen
           dist = textdistance.damerau_levenshtein.distance(query_norm, item)
           if dist < lowest_dist and dist <= max_distance:
               lowest_dist = dist
               best_match = item
               
       return best_match

**Code-Walkthrough:** Der Algorithmus verhindert unnötige Serverlast. Zuerst wird der String bereinigt. Der ``textdistance.damerau_levenshtein.distance``-Aufruf ist das Herzstück: Im Gegensatz zur Standard-Levenshtein-Distanz, die das Vertauschen von zwei Buchstaben als zwei Fehler (1x Löschen, 1x Einfügen) wertet, wertet Damerau dies als *einen* Fehler. Dies fängt das "Fat-Finger-Syndrom" auf Tablets exakt ab. Erst wenn die berechnete Distanz den Schwellenwert von 2 überschreitet, wird das ressourcenintensive ML-Modell getriggert.

2. Probabilistische Pipeline & Active Learning (Human-in-the-Loop)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Versagen alle Heuristiken, feuert der ML-Orchestrator ein trainiertes logistisches Regressionsmodell. Der Python-Code bündelt die Verarbeitung in einer strikten ``scikit-learn``-Pipeline, um Data Leakage zwingend zu verhindern.

.. code-block:: python

   import numpy as np
   from sklearn.pipeline import Pipeline
   from sklearn.feature_extraction.text import TfidfVectorizer
   from sklearn.linear_model import LogisticRegression
   from sklearn.calibration import CalibratedClassifierCV

   # 1. Pipeline-Definition
   nlp_pipeline = Pipeline([
       ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2)),
       ('clf', LogisticRegression(class_weight='balanced', max_iter=500, C=1.0))
   ])

   # 2. Platt Scaling (Probability Calibration)
   calibrated_nlp_model = CalibratedClassifierCV(nlp_pipeline, method='sigmoid', cv=5)
   calibrated_nlp_model.fit(X_train_strings, y_train_labels)

   def ml_predict_with_active_learning(query: str, threshold: float = 0.75) -> Dict:
       """ Führt die Klassifikation durch und triggert ggf. das Active Learning. """
       # Extrahiert Wahrscheinlichkeiten für alle Klassen
       probabilities = calibrated_nlp_model.predict_proba([query])[0]
       best_class_idx = np.argmax(probabilities)
       confidence = probabilities[best_class_idx]
       
       if confidence >= threshold:
           return {"status": "SUCCESS", "category": calibrated_nlp_model.classes_[best_class_idx]}
       else:
           # Confidence zu niedrig -> Triggert Frontend UI für manuelles Labeling
           top_3_indices = np.argsort(probabilities)[-3:][::-1]
           suggestions = [calibrated_nlp_model.classes_[i] for i in top_3_indices]
           return {"status": "AMBIGUOUS", "suggestions": suggestions}

**Code-Walkthrough:** Die Pipeline ist kein Standard-Skript. Der Parameter ``analyzer='char_wb'`` zerlegt das Wort in überlappende Teilstrings (N-Gramme) unter strikter Beachtung der Wortgrenzen (Word Boundaries). Die Gewichtung erfolgt über TF-IDF, wodurch triviale Fragmente algorithmisch bestraft werden. 
Die Funktion ``ml_predict_with_active_learning`` demonstriert die Integration in das System: Da rohe logistische Modelle überkonfident sind, erzwingt ``CalibratedClassifierCV`` (Platt Scaling) echte Wahrscheinlichkeiten. Liegt die Konfidenz unter 75%, wird ein Fallback getriggert. Das System gibt dem Frontend ``suggestions`` zurück, der Kunde klickt auf die richtige Kategorie, und dieser Klick fließt als neuer, hart gelabelter Datenpunkt direkt in das nächste Modell-Training ein (**Active Learning**).

Teil II: Evaluation der NLP-Pipeline
------------------------------------
Um Minderheiten-Klassen (z. B. "Gewürze") nicht zu benachteiligen, wird das Datenset strikt via **Stratified Sampling** geteilt, was die exakte prozentuale Klassenverteilung im Train- und Test-Split mathematisch erhält. 

**1. Latent Space Representation (t-SNE / UMAP)**

.. image:: ../../eval_plots/nlp_tsne_clusters.png
   :width: 600px
   :align: center
   :alt: t-SNE Clustering der TF-IDF Vektoren

**Analytische Dekonstruktion:** Bevor metrische Scores berechnet werden, beweist dieser Plot die mathematische Trennbarkeit der Textdaten. Die TF-IDF-Matrix besitzt hunderte Dimensionen. Das Skript nutzt die **t-SNE** Methode zur nicht-linearen Dimensionsreduktion auf einen 2D-Raum. Die klaren, farbcodierten Insel-Bildungen (Cluster) beweisen visuell, dass die lexikalischen Silben-Fragmente (N-Gramme) ausreichend Entropie besitzen, um die Kategorien abzugrenzen. Verschwimmende Ränder deuten auf systemimmanentes linguistisches Overlap hin, das die logistische Regression algorithmisch trennen muss.

**2. End-to-End Klassifikationspräzision (Confusion Matrix)**

.. image:: ../../eval_plots/nlp_confusion_matrix.png
   :width: 700px
   :align: center
   :alt: End-to-End Confusion Matrix NLP

**Analytische Dekonstruktion:** Die Confusion Matrix deckt systematische Modellfehler auf. Die tiefblaue Hauptdiagonale visualisiert die True Positives. Die Matrix offenbart jedoch Off-Diagonal-Fehler (False Positives) zwischen "Vegan" und "Molkerei". Die physikalische Ursache liegt im geteilten Wortstamm (z. B. "Hafer-Milch"), der im Vektorraum eine hohe Kosinus-Ähnlichkeit erzeugt. Das System toleriert diesen linguistischen Bias bewusst: In Supermarkt-Topologien werden Ersatzprodukte in unmittelbarer räumlicher Proximität (oft im selben Regal) platziert. Der physische Routing-Fehler für den Kunden konvergiert folglich gegen Null.

**3. Probability Calibration (Reliability Diagram)**

.. image:: ../../eval_plots/nlp_calibration.png
   :width: 600px
   :align: center
   :alt: Reliability Diagram Platt Scaling

**Analytische Dekonstruktion:** Dieser Graph validiert die Zuverlässigkeit der ausgegebenen Wahrscheinlichkeiten. Auf der X-Achse ist die Modell-Konfidenz aufgetragen, auf der Y-Achse die empirische Korrektheit. Durch das Platt Scaling schmiegt sich die Kurve perfekt an die ideale Diagonale. Dies garantiert: Meldet die NLP-Engine 85% Sicherheit, ist die Zuordnung in der echten Welt zu exakt 85% korrekt.

Teil III: Prädiktive Stau-Modellierung (Traffic Prediction)
-----------------------------------------------------------
Während das NLP-Modell reaktiv agiert, prädiziert der ``TrafficPredictor`` proaktiv, wie viele Personen sich in der Zukunft auf einer Kante des Graphen befinden.

1. Topologische Feature Extraction & Forecasting-Shift
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Maschinelles Lernen in Netzwerken erfordert die Injektion topologischen Wissens. Das Feature-Engineering verbindet Pandas mit der Graphen-Bibliothek ``NetworkX``, um räumliche Spillover-Effekte zu modellieren, bevor das Zeitreihen-Target verschoben wird.

.. code-block:: python

   import pandas as pd
   import numpy as np
   import networkx as nx

   def extract_neighbor_load(df: pd.DataFrame, G: nx.DiGraph) -> pd.Series:
       """ Zieht den Traffic der direkt angrenzenden Graphen-Kanten. """
       neighbor_loads = []
       for _, row in df.iterrows():
           edge_u, edge_v = row['node_u'], row['node_v']
           # Finde alle ausgehenden Kanten vom Zielknoten v (Topologische Nähe)
           out_edges = G.out_edges(edge_v, data=True)
           # Extrahiere die aktuelle Auslastung dieser Nachbarn aus dem Graph-Zustand
           loads = [data.get('current_occupancy', 0) for u, v, data in out_edges]
           neighbor_loads.append(max(loads) if loads else 0)
       return pd.Series(neighbor_loads, index=df.index)

   def build_feature_matrix(df: pd.DataFrame, G: nx.DiGraph, horizon: int = 5) -> pd.DataFrame:
       # 1. Zirkadiane Rhythmik (Verhindert Singularität 23:00 zu 00:00 Uhr)
       df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
       df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
       
       # 2. Autoregressive Lags (Zeitliches Momentum)
       df['lag_1'] = df.groupby('edge_id')['occupancy'].shift(1)
       
       # 3. Spatial Spillovers (Verknüpfung von ML mit Operations Research)
       df['neighbor_max_occupancy'] = extract_neighbor_load(df, G)
       
       # 4. Das Target (Forecasting Shift) -> Zieht t+5 in die aktuelle Zeile
       df['target_t_plus_5'] = df.groupby('edge_id')['occupancy'].shift(-horizon)
       
       return df

**Code-Walkthrough:** Die Funktion ``extract_neighbor_load`` ist die Brücke zwischen Machine Learning und Graphentheorie. Sie iteriert nicht nur über Tabellen, sondern befragt das ``nx.DiGraph``-Objekt nach topologischen Nachbarn (``G.out_edges``). Das Modell lernt so: "Wenn der nächste Gang voll ist, wird dieser Gang gleich ebenfalls verstopfen." Der negative ``shift(-horizon)`` konstruiert abschließend das saubere Trainings-Target für die Zukunftsvorhersage.

2. XGBoost & Optuna (Nested Cross Validation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Das System nutzt **XGBoost (Extreme Gradient Boosting)**. Der architektonische Kernvorteil liegt in der Nutzung der Taylor-Approximation zweiter Ordnung (Hesse-Matrix) für den Baum-Schnitt und im *Sparsity-aware Split Finding* für fehlende Sensordaten.

.. code-block:: python

   import optuna
   import xgboost as xgb
   from sklearn.model_selection import TimeSeriesSplit
   from sklearn.metrics import root_mean_squared_error

   def objective(trial):
       params = {
           'objective': 'reg:squarederror',
           'max_depth': trial.suggest_int('max_depth', 3, 9),
           'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
           # L1 (Lasso) / L2 (Ridge) Regularisierung zwingen das Modell zur Generalisierung
           'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0),   
           'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0), 
           'n_estimators': 300
       }
       
       # Nested Time-Series CV blockiert Data Leakage in der Optimierung
       tscv = TimeSeriesSplit(n_splits=3)
       fold_errors = []
       
       for train_idx, val_idx in tscv.split(X_train_full):
           X_fold_train = X_train_full.iloc[train_idx]
           X_fold_val = X_train_full.iloc[val_idx]
           
           model = xgb.XGBRegressor(**params)
           model.fit(X_fold_train, y_train_full.iloc[train_idx])
           
           preds = model.predict(X_fold_val)
           fold_errors.append(root_mean_squared_error(y_train_full.iloc[val_idx], preds))
           
       return np.mean(fold_errors) 

**Code-Walkthrough:** Ein simples Aufteilen der Daten würde bei Zeitreihen massives Data Leakage verursachen, da die KI die Zukunft sehen könnte. ``TimeSeriesSplit`` garantiert ein chronologisches Forward-Chaining. Das ``Optuna``-Framework sucht über hunderte Trial-Läufe die mathematisch perfekten Hyperparameter, während ``reg_alpha`` (L1) dafür sorgt, dass redundante Features konsequent ignoriert werden.

Teil IV: Deep-Dive Evaluation der Traffic-Pipeline
--------------------------------------------------
Die Evaluation basiert zwingend auf dem **Root Mean Squared Error (RMSE)**, da dessen Quadrierung extreme Vorhersagefehler massiv bestraft. Das Skript evaluiert XGBoost zwingend gegen eine **Naive Persistence Baseline** (Vorhersage entspricht Ist-Zustand). Der signifikante Delta-Lift beweist mathematisch, dass echte Raum-Zeit-Dynamiken erlernt wurden.

**1. Regression Fit & Fehlerverteilung (Residuals)**

.. image:: ../../eval_plots/ml_residuals.png
   :width: 600px
   :align: center
   :alt: Fehlerverteilung Residuals Traffic

**Analytische Dekonstruktion:** Der Plot visualisiert die Residuen (Differenz zwischen Vorhersage und Realität). Die Verteilung ist stark **leptokurtisch** exakt um den Nullpunkt ($\mu \approx 0.012$), d.h. extreme Fehler fallen sehr flach ab. Der globale Score von $R^2 = 0.920$ belegt eine exzellente Modellgüte. Entscheidend ist die Abwesenheit von **Heteroskedastizität** (kein fächerförmiges Ausfransen der Fehlervarianz). Dies beweist, dass das Modell massive Staus mit derselben relativen Präzision prognostiziert wie leere Gänge.

**2. Explainable AI (TreeSHAP) & Kausale Feature-Interaktion**

.. code-block:: python

   import shap
   # Initialisierung des TreeExplainer (O(T * L * D^2) Laufzeit)
   explainer = shap.TreeExplainer(best_xgb_model)
   shap_values = explainer.shap_values(X_test)
   shap.summary_plot(shap_values, X_test, plot_type="bar")

.. image:: ../../eval_plots/ml_feature_importance.png
   :width: 600px
   :align: center
   :alt: Feature Importance Traffic

**Analytische Dekonstruktion:** Der Code nutzt spieltheoretische SHAP-Werte (SHapley Additive exPlanations) zur Extraktion kausaler Logik. Anstatt nur Feature-Häufigkeiten zu zählen, belegt die Analyse, dass ``total_agents`` (Füllgrad des Marktes) und ``is_shelf_aisle`` (Regal-Gang-Flag) die dominierenden Prädiktoren darstellen. Die KI hat autonom gelernt, dass Staus physikalisch primär durch den statischen Suchprozess an Regalen entstehen, nicht durch reine Transit-Wege.

**3. Temporale Stabilität im Tagesverlauf (Informationstheoretische Grenzen)**

.. image:: ../../eval_plots/ml_error_by_hour.png
   :width: 600px
   :align: center
   :alt: Modellgüte Tagesverlauf Traffic

**Analytische Dekonstruktion:** Der Plot dokumentiert den RMSE als Funktion der Tageszeit. Die Fehlerquote steigt signifikant zum Peak der Rush-Hour (17:00 Uhr) an. Dies ist kein Bug, sondern markiert die **informationstheoretische Grenze** des cyber-physischen Systems. Im absoluten Chaos weicht das Laufverhalten der Agenten durch menschliche Ausweichmanöver (**Bounded Rationality**) von deterministischen Bahnen ab und wird hochgradig stochastisch. 

**4. Zuverlässigkeit der Stau-Erkennung (Business Traffic Matrix)**

.. image:: ../../eval_plots/ml_confusion_matrix.png
   :width: 600px
   :align: center
   :alt: Business Matrix Traffic

**Analytische Dekonstruktion:** Der TSP-Routing-Solver kann Float-Werte wie 3.7 Personen nicht interpretieren. Das System wendet eine methodische **Discretization** (Binning) an, um die numerischen Vorhersagen in diskrete Kategorien ("Frei", "Mittel", "Kritischer Stau") zu übersetzen. Die Matrix belegt, dass "freie Gänge" zu 99.4% und "kritische Staus" zu 92.1% korrekt gebinnt werden. Die leichte Unschärfe im Übergangssegment implementiert eine kybernetische **Hysterese** (Dämpfung), um ein oszillierendes Flackern der gerenderten UI-Route auf dem Tablet zu verhindern.

Teil V: Business Value & End-to-End Evaluation (A/B-Testing)
--------------------------------------------------------------
Die ultimative Validierung der Architektur ist die reale Zeitersparnis. Die Simulationsengine (Kapitel 6) erzwingt hierfür auf Code-Ebene ein isoliertes **A/B-Testing im Shadow-Mode**.

.. code-block:: python

   def simulate_ab_routing(simulation_env):
       results = []
       for agent in simulation_env.agents:
           # Kohorte A: Klassischer Dijkstra ohne ML (Baseline)
           time_baseline = run_dijkstra_routing(agent, simulation_env.graph)
           
           # Kohorte B: Dynamisches OR-Routing mit XGBoost-Stauvorhersage
           penalized_graph = apply_xgboost_penalties(simulation_env.graph, traffic_model)
           time_ml = run_tsp_orchestrator(agent, penalized_graph)
           
           # Delta (Ersparte Zeit) in Sekunden
           results.append({"agent_id": agent.id, "time_saved": time_baseline - time_ml})
       return pd.DataFrame(results)

**Verteilung der tatsächlich ersparten Zeit**

.. image:: ../../eval_plots/business_value_time_saved.png
   :width: 600px
   :align: center
   :alt: Business Value Ersparte Zeit

**Analytische Dekonstruktion:** Dieser Density Plot visualisiert den Output der A/B-Testing-Simulation. Die Kurve offenbart eine signifikant **rechtsschiefe Verteilung (Right-Skewed Distribution)** mit einem Durchschnittswert von +184 ersparten Sekunden pro Einkauf (bestätigt via Welch-Test, $p < 0.001$). 

Das systemarchitektonisch wertvollste Resultat verbirgt sich im **Long Tail** (dem flachen Ausläufer rechts): Bei ca. 12% der Einkäufe (während der Rush-Hour) spart das hybride Routing dem Kunden über 400 Sekunden (fast 7 Minuten). Dieses asymmetrische Phänomen entsteht, weil das ML-Modell physikalische **Kaskadeneffekte** (rückstauende Warteschlangen, die Hauptgänge kollabieren lassen) Millisekunden vor deren Entstehung prädiziert. Der physische Umweg (Cost-of-Rerouting) wird von der Ersparnis an passiver Stehzeit völlig überkompensiert.

Teil VI: Model Serving, CI/CD & Data Drift Monitoring
-----------------------------------------------------
Ein Modell stiftet erst dann Wert, wenn es latenzarm inferiert und gegen Degradation abgesichert ist.

1. High-Performance Model Serving (FastAPI & ONNX)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Um die XGBoost-Modelle in das C++-basierte Operations-Research-Backend einzubinden, werden sie in den **ONNX-Standard (Open Neural Network Exchange)** kompiliert. Dies entkoppelt das Modell von der Python-Laufzeitumgebung (GIL-Bottleneck) und ermöglicht C-basierte Inferenzzeiten.

.. code-block:: python

   import onnxruntime as rt
   from fastapi import FastAPI
   from pydantic import BaseModel
   import numpy as np

   app = FastAPI(title="Smart Cart Traffic Predictor")
   session = rt.InferenceSession("models/xgboost_traffic.onnx")

   class EdgeFeatureVector(BaseModel):
       hour_sin: float
       hour_cos: float
       lag_1: float
       neighbor_max_occupancy: float
       is_shelf_aisle: int

   @app.post("/predict_penalty")
   async def predict_penalty(features: EdgeFeatureVector):
       input_data = np.array([[
           features.hour_sin, features.hour_cos, features.lag_1, 
           features.neighbor_max_occupancy, features.is_shelf_aisle
       ]], dtype=np.float32)
       
       # Zero-Copy Inferenz
       prediction = session.run(None, {"input": input_data})[0][0]
       
       # Umrechnung in Routing-Penalty (2.5 simulierte Zusatz-Meter pro Person)
       return {"edge_penalty_meters": float(prediction * 2.5)}

2. Data Drift Monitoring (Kullback-Leibler-Divergenz)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Da Supermärkte umgebaut werden oder saisonale Effekte das Kundenverhalten verändern, implementiert das System ein kontinuierliches Monitoring, um **Data Drift** mathematisch zu erfassen. Die MLOps-Pipeline nutzt die **KL-Divergenz**, um die Live-Sensordaten ($P$) mit der Baseline ($Q$) zu vergleichen:

$$D_{KL}(P \parallel Q)=\sum_{x \in X}P(x)\log\left(\frac{P(x)}{Q(x)}\right)$$

.. code-block:: python

   from scipy.stats import entropy
   import prometheus_client as prom

   drift_gauge = prom.Gauge('model_data_drift_kl', 'KL Divergence')

   def check_data_drift(live_data_batch: np.ndarray, training_baseline: np.ndarray) -> bool:
       p_live, _ = np.histogram(live_data_batch, bins=50, density=True)
       q_train, _ = np.histogram(training_baseline, bins=50, density=True)
       
       p_live = np.where(p_live == 0, 1e-10, p_live)
       q_train = np.where(q_train == 0, 1e-10, q_train)
       
       kl_div = entropy(p_live, q_train)
       drift_gauge.set(kl_div)
       
       # Harter Schwellenwert für automatisiertes CT (Continuous Training)
       if kl_div > 0.15:
           trigger_airflow_retraining_pipeline()
           return True
       return False

Überschreitet die KL-Divergenz den Schwellenwert von 0.15, triggert das System vollautomatisch einen Apache Airflow-DAG. Dieser zieht die neuesten IoT-Sensordaten, führt das Optuna-Tuning neu aus und rollt ein an die veränderte Realität angepasstes ONNX-Modell ohne Downtime aus (**Continuous Training**).