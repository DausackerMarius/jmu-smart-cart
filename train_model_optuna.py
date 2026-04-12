"""
=========================================================================================
JMU SMART SUPERMARKET: XGBOOST SPATIO-TEMPORAL TRAFFIC PREDICTOR
=========================================================================================
Dieses Skript trainiert das "Gehirn" unserer Anwendung. 
Es nimmt die generierten Rohdaten aus der Simulation und lernt daraus, 
wie sich Stau im Supermarkt über Raum (Spatio) und Zeit (Temporal) verhält.

Architektur-Entscheidung: Warum XGBoost und kein Deep Learning (Neuronales Netz)?
Für strukturierte, tabellarische Daten (wie Excel-Tabellen) schlagen Gradient-Boosting-
Verfahren (wie XGBoost) komplexe Neuronale Netze fast immer in Bezug auf Geschwindigkeit, 
Ressourcenverbrauch und Erklärbarkeit. In der Enterprise-KI ist "Explainability" 
(Erklärbarkeit der Feature-Wichtigkeiten) ein massiver Pluspunkt.
=========================================================================================
"""

import pandas as pd
import xgboost as xgb
import pickle
import optuna
import json
import numpy as np
import os
import gc
import traceback
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

INPUT_FILE = "smartcart_traffic_training_data.csv"
MODEL_FILE = "traffic_model_xgboost.pkl"

def load_and_engineer_data():
    print("[1/5] Lade Rohdaten & initialisiere Pipeline...")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Datei {INPUT_FILE} nicht gefunden. Bitte erst Simulation starten.")

    df = pd.read_csv(INPUT_FILE)
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp_dt').reset_index(drop=True)

    # ---------------------------------------------------------
    # SCHRITT 1: CLEANING & IMPUTATION (Datenbereinigung)
    # ---------------------------------------------------------
    print("[2/5] Resampling & Generierung der Makro-Zeitreihen-Features...")
    
    # Lückenlose Zeitreihe aufbauen:
    # Machine Learning Modelle hassen fehlende Daten. Wenn in der Simulation mal für 
    # 5 Minuten kein Log geschrieben wurde, würde das Modell stolpern. 
    # Wir erzwingen hier einen lückenlosen 5-Minuten-Takt (Resampling).
    df = df.set_index('timestamp_dt')
    
    # FIX: Entferne doppelte Zeitstempel (behalte den aktuellsten), um den reindex-Crash zu verhindern!
    df = df[~df.index.duplicated(keep='last')]
    
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='5min')
    df = df.reindex(full_range)

    # Leere Felder (NaN = Not a Number) sinnvoll auffüllen
    df['total_agents'] = df['total_agents'].fillna(0).astype(np.float32)
    df['edge_loads_json'] = df['edge_loads_json'].fillna('{}') 
    df['is_holiday'] = df['is_holiday'].fillna(0).astype(np.int8)

    # Kassen-Warteschlangen auffüllen (ffill = forward fill). Wenn ein Wert fehlt, 
    # nehmen wir einfach den Wert von vor 5 Minuten an.
    queue_cols = ['k1_q', 'k2_q', 'k3_q', 'k2_open', 'k3_open']
    df[queue_cols] = df[queue_cols].ffill().fillna(0).astype(np.float32)

    df = df.reset_index().rename(columns={'index': 'timestamp_dt'})

    # Zeitliche Features extrahieren, damit der Algorithmus Muster wie "Feierabend" erkennen kann
    df['month'] = df['timestamp_dt'].dt.month.astype(np.int8)
    df['weekday'] = df['timestamp_dt'].dt.weekday.astype(np.int8)
    df['hour'] = df['timestamp_dt'].dt.hour.astype(np.int8)
    df['minute'] = df['timestamp_dt'].dt.minute.astype(np.int8)

    # --- ZUSTANDSLOSE MAKRO-FEATURES (Stateless Inference) ---
    # WISSENSCHAFTLICHER HINWEIS ZUR ARCHITEKTUR:
    # Wir berechnen hier bewusst nur zustandslose Metriken (wie Auslastung / offene Kassen).
    # Verzögert berechnete Features (Lag-Features wie .shift(1)) wurden aus dem Modell 
    # entfernt, da diese im Live-System (app.py) bei einer kalten Suchanfrage nicht 
    # verfügbar wären. Dies eliminiert den Train-Serving Skew.
    
    df['total_queue'] = (df['k1_q'] + df['k2_q'] + df['k3_q']).astype(np.float32)
    df['open_registers'] = (1 + df['k2_open'] + df['k3_open']).astype(np.float32)
    df['queue_pressure'] = (df['total_queue'] / df['open_registers']).astype(np.float32)

    # ---------------------------------------------------------
    # SCHRITT 2: TOPOLOGIE SCAN
    # ---------------------------------------------------------
    # In unseren Rohdaten liegt der Stau als verpackter Text (JSON) vor.
    # Hier scannen wir einmal alle Daten durch, um alle Gänge des Supermarkts zu finden.
    all_edges = set()
    for json_str in df['edge_loads_json']:
        if json_str != '{}':
            all_edges.update(json.loads(json_str).keys())
    all_edges = sorted(list(all_edges))
    n_edges = len(all_edges)

    # ---------------------------------------------------------
    # SCHRITT 3: MATRIX-TRANSFORMATION (EDGE-LEVEL EXPANSION)
    # ---------------------------------------------------------
    print("[3/5] Matrix-Transformation (Edge-Level Expansion)...")
    # XGBoost braucht saubere, zweidimensionale Tabellen.
    # Wir "explodieren" unsere Daten: Aus einer Zeile pro Uhrzeit machen wir 
    # viele Zeilen pro Uhrzeit (eine Zeile für jeden einzelnen Gang im Supermarkt).
    cols_to_keep = ['timestamp_dt', 'month', 'weekday', 'hour', 'minute', 'is_holiday', 
                    'total_agents', 'total_queue', 'open_registers', 'queue_pressure']

    df_expanded = pd.DataFrame({
        col: np.repeat(df[col].values, n_edges) for col in cols_to_keep
    })
    df_expanded['edge_id'] = np.tile(all_edges, len(df))

    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    loads_array = np.zeros(len(df_expanded), dtype=np.float32)

    # Wir entpacken die JSON-Texte in echte mathematische Zahlen
    for i, json_str in enumerate(df['edge_loads_json']):
        if json_str == '{}': continue
        json_data = json.loads(json_str)
        base_idx = i * n_edges
        for edge, load in json_data.items():
            if edge in edge_to_idx:
                loads_array[base_idx + edge_to_idx[edge]] = load

    df_expanded['load'] = loads_array
    
    # Warum nutzen wir np.log1p (Logarithmus + 1)?
    # Verkehr/Stau hat oft eine "rechtsschiefe" Verteilung (meistens 0 Leute, selten 30 Leute).
    # Bäume (XGBoost) tun sich schwer mit solchen Ausreißern und könnten negative Personen 
    # vorhersagen (z.B. -2 Kunden). Der Logarithmus staucht große Ausreißer zusammen 
    # und verhindert unlogische negative Vorhersagen.
    df_expanded['log_load'] = np.log1p(df_expanded['load']).astype(np.float32)

    del df # RAM aufräumen
    gc.collect()

    # ---------------------------------------------------------
    # SCHRITT 4: SPATIAL & INTERACTION FEATURE ENGINEERING
    # ---------------------------------------------------------
    print("[4/5] Ableitung räumlicher und interaktiver Variablen...")

    # ML-Modelle verstehen keinen Text wie "vD6-vD5". Wir wandeln die Namen in Zahlen um.
    le = LabelEncoder()
    df_expanded['edge_id_enc'] = le.fit_transform(df_expanded['edge_id']).astype(np.int16)

    # Wir geben dem Modell "Domänenwissen" mit: Ist dieser Gang eine Kasse oder ein Regal?
    edge_df = pd.DataFrame({'edge_id': all_edges})
    edge_df['is_checkout_zone'] = edge_df['edge_id'].str.contains(r'vK|vW').astype(np.int8)
    edge_df['is_main_aisle'] = edge_df['edge_id'].str.contains(r'vD').astype(np.int8)
    edge_df['is_shelf_aisle'] = edge_df['edge_id'].str.contains(r'vA|vB|vC').astype(np.int8)

    df_expanded = df_expanded.merge(edge_df, on='edge_id', how='left')

    # Warum Sinus/Cosinus für die Uhrzeit?
    # Für einen Computer ist 23 Uhr weit entfernt von 1 Uhr nachts (23 vs 1).
    # In der Realität liegen sie aber nebeneinander. Durch die Projektion auf einen Kreis 
    # (Sinus/Cosinus) kapiert das Modell, dass die Zeit zyklisch ist.
    df_expanded['hour_sin'] = np.sin(2 * np.pi * df_expanded['hour'] / 24).astype(np.float32)
    df_expanded['hour_cos'] = np.cos(2 * np.pi * df_expanded['hour'] / 24).astype(np.float32)
    df_expanded['day_sin'] = np.sin(2 * np.pi * df_expanded['weekday'] / 7).astype(np.float32)
    df_expanded['day_cos'] = np.cos(2 * np.pi * df_expanded['weekday'] / 7).astype(np.float32)
    
    df_expanded['is_weekend'] = (df_expanded['weekday'] >= 4).astype(np.int8)
    df_expanded['is_rush_hour'] = ((df_expanded['hour'] >= 16) & (df_expanded['hour'] <= 19)).astype(np.int8)

    # Indikatoren für Übersprung-Effekte (Wenn Kassen voll sind, laufen die Leute in den Hauptgang über)
    df_expanded['spillover_risk'] = (df_expanded['queue_pressure'] * df_expanded['is_main_aisle']).astype(np.float32)
    df_expanded['shelf_density'] = (df_expanded['total_agents'] * df_expanded['is_shelf_aisle']).astype(np.float32)

    # Nachts (23-5 Uhr) ist der Markt ohnehin zu, diese Daten löschen wir, um das Modell nicht zu verzerren.
    mask_open = (df_expanded['hour'] >= 6) & (df_expanded['hour'] <= 22)
    df_expanded = df_expanded[mask_open].copy() 
    
    gc.collect()

    # Dies ist die finale Liste der Merkmale, die unsere KI berücksichtigen darf.
    # WICHTIG: Identisch mit feature_pool in der app.py (Inference Engine)
    features = [
        'edge_id_enc', 'is_checkout_zone', 'is_main_aisle', 'is_shelf_aisle',
        'is_holiday', 'is_weekend', 'is_rush_hour',
        'total_agents', 'total_queue', 'open_registers', 'queue_pressure',
        'spillover_risk', 'shelf_density',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month'
    ]
    
    return df_expanded, features, 'log_load', le


# =============================================================================
# OPTUNA HYPERPARAMETER TUNING
# =============================================================================
def objective(trial, X_train, y_train, X_val, y_val):
    """
    Optuna sucht nach den besten Einstellungen (Hyperparametern) für das KI-Modell.
    Es rät nicht einfach blind (wie GridSearch), sondern nutzt Bayes'sche Optimierung 
    (Tree-structured Parzen Estimator), um aus Fehlern der vorherigen Versuche zu lernen.
    """
    param = {
        'verbosity': 0,
        'objective': 'reg:squarederror',
        'tree_method': 'hist', 
        'booster': 'gbtree',
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-4, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2), 
        'n_estimators': 300, 
        'max_depth': trial.suggest_int('max_depth', 6, 11),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
    }
    
    # Initialisierung der neuesten XGBoost API
    model = xgb.XGBRegressor(
        **param, 
        n_jobs=-1, 
        early_stopping_rounds=20, # Bricht ab, wenn sich das Modell nach 20 Runden nicht mehr verbessert
        eval_metric="rmse"
    )
    
    model.fit(
        X_train, y_train, 
        eval_set=[(X_val, y_val)], 
        verbose=False
    )
    
    # Da wir log-Werte vorhergesagt haben, müssen wir sie für die Bewertung
    # mit expm1 wieder in die echte Welt (reelle Personenanzahl) zurückwandeln.
    preds_log = model.predict(X_val)
    preds_real = np.expm1(preds_log)
    y_val_real = np.expm1(y_val)
    
    return mean_absolute_error(y_val_real, preds_real)

def print_optuna_progress(study, trial):
    print(f" -> Trial {trial.number + 1}/30 abgeschlossen | Aktueller MAE: {trial.value:.4f} | Bester MAE bisher: {study.best_value:.4f}")

if __name__ == "__main__":
    try:
        df, features, target_col, encoder = load_and_engineer_data()
        
        # ---------------------------------------------------------
        # SCHRITT 5: CHRONOLOGICAL TIME-SERIES SPLIT (Der 1,0 Fix)
        # ---------------------------------------------------------
        print(f"[5/5] Chronological Time-Series Split (Verhindert Data Leakage!)...")
        # WISSENSCHAFTLICHER HINWEIS (Look-ahead Bias Vermeidung):
        # Bei Zeitreihen darf niemals ein zufälliger train_test_split verwendet werden! 
        # Das Modell würde sonst Daten aus der Zukunft (z.B. Dezember) nutzen, 
        # um die Vergangenheit (z.B. Oktober) vorherzusagen. 
        # LÖSUNG: Wir sortieren die Zeitstempel streng chronologisch und schneiden 
        # die Zeitachse hart ab (die ersten 85% für das Training, die letzten 15% 
        # strikt für die ungesehene Zukunft/Testset).
        
        unique_timestamps = np.sort(df['timestamp_dt'].unique())
        
        # 1. Chronologischer Split in Train/Val (85%) und Test (15%)
        test_split_idx = int(len(unique_timestamps) * 0.85)
        train_val_times = unique_timestamps[:test_split_idx]
        test_times = unique_timestamps[test_split_idx:]
        
        # 2. Chronologischer Split in Train (80%) und Validation (20%)
        val_split_idx = int(len(train_val_times) * 0.80)
        train_times = train_val_times[:val_split_idx]
        val_times = train_val_times[val_split_idx:]
        
        train_mask = df['timestamp_dt'].isin(train_times)
        val_mask = df['timestamp_dt'].isin(val_times)
        test_mask = df['timestamp_dt'].isin(test_times)
        
        X_train_full, y_train_full = df[train_mask][features], df[train_mask][target_col]
        X_val_full, y_val_full = df[val_mask][features], df[val_mask][target_col]
        X_test, y_test = df[test_mask][features], df[test_mask][target_col]
        
        # Um Zeit beim Hyperparameter-Tuning zu sparen, geben wir Optuna nur die letzten 30% 
        # der jeweiligen Zeitreihen (damit es auf den aktuellsten Trends trainiert)
        optuna_train_idx = int(len(train_times) * 0.30)
        optuna_val_idx = int(len(val_times) * 0.30)
        
        optuna_train_times = train_times[-optuna_train_idx:]
        optuna_val_times = val_times[-optuna_val_idx:]
        
        X_train_optuna = df[df['timestamp_dt'].isin(optuna_train_times)][features]
        y_train_optuna = df[df['timestamp_dt'].isin(optuna_train_times)][target_col]
        X_val_optuna = df[df['timestamp_dt'].isin(optuna_val_times)][features]
        y_val_optuna = df[df['timestamp_dt'].isin(optuna_val_times)][target_col]
        
        print(f" -> Gesamt-Datenbank:      {len(df):,}")
        print(f" -> Optuna Tuning Subset:  {len(X_train_optuna):,} Rows")
        print(f" -> Finale Trainingsbasis: {len(X_train_full):,} Rows")
        print(f" -> Hold-Out Testset:      {len(X_test):,} Rows")
        
        print("\nStarte Optuna Hyperparameter-Optimierung (30 Trials)...")
        optuna.logging.set_verbosity(optuna.logging.WARNING) 
        
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: objective(trial, X_train_optuna, y_train_optuna, X_val_optuna, y_val_optuna), 
            n_trials=30,
            callbacks=[print_optuna_progress]
        )
        
        print("\nOptimierung abgeschlossen. Beste Hyperparameter:")
        for key, value in study.best_params.items():
            print(f" - {key}: {value}")
        
        print("\nErmittle optimale Iterationsanzahl auf vollen Daten...")
        best_params_final = study.best_params.copy()
        best_params_final['n_estimators'] = 1500
        
        temp_model = xgb.XGBRegressor(
            **best_params_final, 
            n_jobs=-1,
            early_stopping_rounds=30,
            eval_metric="rmse"
        )
        
        temp_model.fit(
            X_train_full, y_train_full, 
            eval_set=[(X_val_full, y_val_full)], 
            verbose=False
        )
        
        optimal_trees = temp_model.best_iteration + 1
        print(f" -> Optimaler Tree-Count erreicht bei: {optimal_trees}")
        
        print("\nTrainiere finales Modell auf aggregierten Trainingsdaten (Train + Val)...")
        # Vor dem Testen werfen wir Trainings- und Validierungsdaten zusammen. 
        X_train_final = pd.concat([X_train_full, X_val_full])
        y_train_final = pd.concat([y_train_full, y_val_full])
        
        final_params = study.best_params.copy()
        # Heuristische Anpassung: Da wir dem Modell nun 20% mehr Daten füttern, 
        # skalieren wir die Anzahl der Entscheidungsbäume leicht nach oben, um Underfitting zu vermeiden.
        final_params['n_estimators'] = int(optimal_trees * 1.15) 
        
        # Das finale Modell braucht kein Early Stopping, da die Baum-Anzahl hart fixiert ist
        final_model = xgb.XGBRegressor(**final_params, n_jobs=-1)
        final_model.fit(X_train_final, y_train_final)

        print("\nEvaluiere ungesehenes Testset...")
        log_preds = final_model.predict(X_test)
        
        log_preds = np.clip(log_preds, 0, None) 
        real_preds = np.expm1(log_preds) 
        real_y = np.expm1(y_test)
        
        # Metriken für die Bewertung des Modells
        # MAE: Durchschnittliche Abweichung (in Personen) von der Realität
        # RMSE: Bestraft große Fehler (Ausreißer) deutlich härter als der MAE
        # R2: Wie viel Prozent der Stau-Schwankungen kann das Modell erklären?
        final_mae = mean_absolute_error(real_y, real_preds)
        final_rmse = np.sqrt(mean_squared_error(real_y, real_preds))
        final_r2 = r2_score(real_y, real_preds)
        
        print(f" -> MAE  (Mean Absolute Error):     {final_mae:.4f}")
        print(f" -> RMSE (Root Mean Squared Error): {final_rmse:.4f}")
        print(f" -> R2   (Erklärte Varianz):        {final_r2:.4f}")
        
        print("\nTop 10 Feature Importances:")
        # Explainable AI: Was war dem Modell am wichtigsten, um Stau vorherzusagen?
        importances = final_model.feature_importances_
        feature_importances = pd.DataFrame({'Feature': features, 'Importance': importances})
        feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
        for _, row in feature_importances.head(10).iterrows():
            print(f" - {row['Feature']}: {row['Importance']:.4f}")
        
        print(f"\nSpeichere Modell-Artefakte nach {MODEL_FILE}...")
        all_edges_list = list(encoder.classes_)
        
        # Wir speichern das Modell, den LabelEncoder und wichtige Metadaten zusammen ab
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump({
                'model': final_model,
                'encoder': encoder,
                'features': features,
                'edge_list': all_edges_list,
                'is_log_target': True
            }, f)
            
        print("Speichere ungesehenes Test-Set für die Evaluierungs-Pipeline...")
        df_export = df[test_mask].copy() 
        df_export['true_target'] = y_test
        df_export.to_csv("test_data_holdout.csv", index=False)
        
        print("Prozess erfolgreich abgeschlossen.")
    
    except Exception as e:
        print(f"\nKritischer Fehler: {e}")
        traceback.print_exc()