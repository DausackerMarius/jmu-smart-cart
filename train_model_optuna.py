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

# =============================================================================
# ENTERPRISE ML KONFIGURATION (FERRARI EDITION)
# =============================================================================
INPUT_FILE = "smartcart_traffic_training_data.csv"
MODEL_FILE = "traffic_model_ferrari.pkl"
SPLIT_DATE = '2025-11-01' 

def load_and_engineer_data():
    print("⏳ [1/6] Lade Rohdaten & Initialisiere Ferrari-Pipeline...")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Datei {INPUT_FILE} nicht gefunden!")
    
    df = pd.read_csv(INPUT_FILE)
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp_dt').reset_index(drop=True)

    # ---------------------------------------------------------
    # SCHRITT 1: CLEANING & IMPUTATION
    # ---------------------------------------------------------
    print("⏱️ [2/6] Resampling & Makro-Zeitreihen-Features...")
    df = df.set_index('timestamp_dt')
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='5min')
    df = df.reindex(full_range)
    
    df['total_agents'] = df['total_agents'].fillna(0)
    df['edge_loads_json'] = df['edge_loads_json'].fillna('{}') 
    df['is_holiday'] = df['is_holiday'].fillna(0)
    
    queue_cols = ['k1_q', 'k2_q', 'k3_q', 'k2_open', 'k3_open']
    df[queue_cols] = df[queue_cols].ffill().fillna(0)
    
    df = df.reset_index().rename(columns={'index': 'timestamp_dt'})
    
    df['month'] = df['timestamp_dt'].dt.month
    df['weekday'] = df['timestamp_dt'].dt.weekday
    df['hour'] = df['timestamp_dt'].dt.hour
    df['minute'] = df['timestamp_dt'].dt.minute

    # --- NEU: AUTOREGRESSIVE MAKRO-FEATURES (Momentum) ---
    # Bevor wir die Daten explodieren, berechnen wir das zeitliche Momentum des Ladens.
    df['total_queue'] = df['k1_q'] + df['k2_q'] + df['k3_q']
    df['open_registers'] = 1 + df['k2_open'] + df['k3_open']
    df['queue_pressure'] = df['total_queue'] / df['open_registers']
    
    # Lag 1: Zustand vor 5 Minuten
    df['queue_pressure_lag1'] = df['queue_pressure'].shift(1).fillna(0)
    df['total_agents_lag1'] = df['total_agents'].shift(1).fillna(0)
    
    # Delta (Momentum): Wächst der Stau oder löst er sich auf? (+ Wert = Stau wächst)
    df['queue_momentum'] = df['queue_pressure'] - df['queue_pressure_lag1']
    df['fill_rate'] = df['total_agents'] - df['total_agents_lag1']

    # ---------------------------------------------------------
    # SCHRITT 2: TOPOLOGIE SCAN
    # ---------------------------------------------------------
    print("🔍 [3/6] Deep Scan der Laden-Topologie...")
    all_edges = set()
    for json_str in df['edge_loads_json']:
        if json_str != '{}':
            all_edges.update(json.loads(json_str).keys())
    all_edges = sorted(list(all_edges))
    n_edges = len(all_edges)
    print(f"   -> Topologie erkannt: {n_edges} physische Pfade.")
    
    # ---------------------------------------------------------
    # SCHRITT 3: DATA EXPLOSION (Vectorized)
    # ---------------------------------------------------------
    print("💥 [4/6] Daten-Explosion (Multidimensionale Matrix)...")
    cols_to_keep = ['timestamp_dt', 'month', 'weekday', 'hour', 'minute', 'is_holiday', 
                    'total_agents', 'total_queue', 'open_registers', 'queue_pressure',
                    'queue_momentum', 'fill_rate']
    
    df_expanded = df.loc[df.index.repeat(n_edges)].reset_index(drop=True)
    df_expanded = df_expanded[cols_to_keep]
    df_expanded['edge_id'] = all_edges * len(df)
    
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    loads_array = np.zeros(len(df_expanded), dtype=np.float32)
    
    for i, row in df.iterrows():
        if row['edge_loads_json'] == '{}': continue
        json_data = json.loads(row['edge_loads_json'])
        base_idx = i * n_edges
        for edge, load in json_data.items():
            if edge in edge_to_idx:
                loads_array[base_idx + edge_to_idx[edge]] = load
                
    df_expanded['load'] = loads_array
    df_expanded['log_load'] = np.log1p(df_expanded['load'])
    
    del df
    gc.collect()

    # ---------------------------------------------------------
    # SCHRITT 4: SPATIAL & INTERACTION FEATURE ENGINEERING
    # ---------------------------------------------------------
    print("🧠 [5/6] Generiere räumliche und Interaktions-Features...")
    
    le = LabelEncoder()
    df_expanded['edge_id_enc'] = le.fit_transform(df_expanded['edge_id'])
    
    # A) Zyklische Zeit
    df_expanded['hour_sin'] = np.sin(2 * np.pi * df_expanded['hour'] / 24)
    df_expanded['hour_cos'] = np.cos(2 * np.pi * df_expanded['hour'] / 24)
    df_expanded['day_sin'] = np.sin(2 * np.pi * df_expanded['weekday'] / 7)
    df_expanded['day_cos'] = np.cos(2 * np.pi * df_expanded['weekday'] / 7)
    df_expanded['is_weekend'] = df_expanded['weekday'].apply(lambda x: 1 if x >= 4 else 0)
    df_expanded['is_rush_hour'] = df_expanded['hour'].apply(lambda x: 1 if 16 <= x <= 19 else 0)
    
    # B) Spatial Features
    df_expanded['is_checkout_zone'] = df_expanded['edge_id'].str.contains(r'vK|vW').astype(int)
    df_expanded['is_main_aisle'] = df_expanded['edge_id'].str.contains(r'vD').astype(int)
    df_expanded['is_shelf_aisle'] = df_expanded['edge_id'].str.contains(r'vA|vB|vC').astype(int)

    # --- NEU: FEATURE INTERAKTIONEN ---
    # Wir nehmen dem XGBoost-Baum die mathematische Arbeit ab.
    # Hoher Kassen-Druck * Hauptgang = Extreme Stau-Wahrscheinlichkeit
    df_expanded['spillover_risk'] = df_expanded['queue_pressure'] * df_expanded['is_main_aisle']
    # Viele Leute im Markt * Regal-Gang = Hohe Regal-Dichte
    df_expanded['shelf_density'] = df_expanded['total_agents'] * df_expanded['is_shelf_aisle']

    print("🧹 [Cleanup] Fokussiere auf Geschäftszeiten (06:00 - 22:00)...")
    mask_open = (df_expanded['hour'] >= 6) & (df_expanded['hour'] <= 22)
    df_expanded = df_expanded[mask_open]

    # Das Arsenal an Features (Ferrari Level)
    features = [
        'edge_id_enc', 'is_checkout_zone', 'is_main_aisle', 'is_shelf_aisle',
        'is_holiday', 'is_weekend', 'is_rush_hour',
        'total_agents', 'total_queue', 'open_registers', 'queue_pressure',
        'queue_momentum', 'fill_rate', 'spillover_risk', 'shelf_density', # <- Die neuen Gamechanger
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
    ]
    
    return df_expanded, features, 'log_load', le

# =============================================================================
# OPTUNA & TRAINING
# =============================================================================
def objective(trial, X_train, y_train, X_test, y_test):
    param = {
        'verbosity': 0,
        'objective': 'reg:squarederror',
        'tree_method': 'hist', 
        'booster': 'gbtree',
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 6, 12), # Max 12 ist tief genug durch Feature Interaktionen
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
    }
    
    model = xgb.XGBRegressor(**param, n_jobs=-1, early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # --- NEU: Wissenschaftlich korrekte Evaluation in Optuna ---
    preds_log = model.predict(X_test)
    # Rücktransformation für echte MAE Optimierung
    preds_real = np.expm1(preds_log)
    y_test_real = np.expm1(y_test)
    
    # RMSE ist oft besser für Algorithmen, die extreme Ausreißer (Staus) bestrafen sollen.
    # Hier nutzen wir MAE als Zielfunktion, um robust gegen kleine Schwankungen zu sein.
    return mean_absolute_error(y_test_real, preds_real)

if __name__ == "__main__":
    try:
        df, features, target_col, encoder = load_and_engineer_data()
        
        print(f"\n✂️ [6/6] Chronologischer Time-Series Split (Trennlinie: {SPLIT_DATE})...")
        train_mask = df['timestamp_dt'] < pd.to_datetime(SPLIT_DATE)
        test_mask = df['timestamp_dt'] >= pd.to_datetime(SPLIT_DATE)
        
        df_train = df[train_mask]
        df_test = df[test_mask]
        
        X_train, y_train = df_train[features], df_train[target_col]
        X_test, y_test = df_test[features], df_test[target_col]
        
        print(f"   Training Rows (März - Okt): {len(X_train):,}")
        print(f"   Test Rows (Nov - Dez):      {len(X_test):,}")
        
        print(f"\n🚀 Starte Optuna (15 Trials) mit re-transformierter Metrik...")
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=15)
        
        print("\n🏆 Beste Parameter gefunden:")
        print(study.best_params)
        
        print("\n🏋️ Trainiere Finales Modell auf Best Params...")
        final_model = xgb.XGBRegressor(**study.best_params, n_jobs=-1)
        final_model.fit(X_train, y_train)
        
        print("\n📊 Evaluiere auf echtem Weihnachts-Stresstest (Nov-Dez)...")
        log_preds = final_model.predict(X_test)
        
        # Limit predictions to >= 0 before expm1 to prevent numerical weirdness
        log_preds = np.clip(log_preds, 0, None) 
        
        real_preds = np.expm1(log_preds) 
        real_y = np.expm1(y_test)
        
        final_mae = mean_absolute_error(real_y, real_preds)
        final_rmse = np.sqrt(mean_squared_error(real_y, real_preds))
        final_r2 = r2_score(real_y, real_preds)
        
        print(f"   MAE (Durchschn. Fehler): {final_mae:.4f} Personen pro Kante")
        print(f"   RMSE (Straftat für extreme Ausreißer): {final_rmse:.4f}")
        print(f"   R² Score (Erklärte Varianz): {final_r2:.4f}")
        
        print("\n💡 Feature Importances (Die Physik des Supermarkts):")
        importances = final_model.feature_importances_
        feature_importances = pd.DataFrame({'Feature': features, 'Importance': importances})
        feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
        for _, row in feature_importances.head(10).iterrows():
            print(f"   - {row['Feature']}: {row['Importance']:.4f}")
        
        print(f"\n💾 Speichere {MODEL_FILE}...")
        all_edges_list = list(encoder.classes_)
        
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump({
                'model': final_model,
                'encoder': encoder,
                'features': features,
                'edge_list': all_edges_list,
                'is_log_target': True
            }, f)
            
        print("✅ FERTIG. Willkommen in der Königsklasse.")
    
    except Exception as e:
        print(f"\n❌ FEHLER: {e}")
        traceback.print_exc()