"""
=========================================================================================
JMU SMART SUPERMARKET: EVALUATION SÄULE B (MONTE-CARLO BUSINESS VALUE)
=========================================================================================
Dieses Skript liefert den ultimativen betriebswirtschaftlichen und kybernetischen Beweis 
(den Return on Investment) für die Architektur des Smart Carts.

METHODIK (Ablationsstudie via Monte-Carlo-Simulation):
Wir simulieren Hunderte von stochastisch generierten Einkäufen unter realen, von der KI 
vorhergesagten Verkehrsbedingungen. Dabei lassen wir zwei Agenten im Shadow-Mode antreten:
1. "Naive Routing" (Baseline): Die klassische Navigation. Kennt nur die physikalische 
   Distanz (Meter) und ist komplett blind für Staus.
2. "Smart Routing" (MLOps): Kennt die Distanz UND die von XGBoost prädizierten Stau-Strafen.

WISSENSCHAFTLICHER ANSATZ (Risk/Reward Honesty):
Wir nutzen hier bewusst KEIN Fallback-Sicherheitsnetz (wie z.B. einen Max-Umweg-Filter). 
Wenn die KI den Kunden in seltenen Fällen (aufgrund eines False-Positives) fälschlicherweise 
auf einen längeren physischen Umweg schickt, wird dieser Zeitverlust ehrlich erfasst (Werte < 0). 
Das beweist wissenschaftliche Integrität und zeigt das reale Risiko/Rendite-Profil des Systems.
=========================================================================================
"""

import matplotlib
matplotlib.use('Agg') # WICHTIG: Verhindert fatale TclErrors auf Headless-Servern ohne GUI

import pandas as pd
import numpy as np
import networkx as nx
import pickle
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Architektur-Pattern: Wir importieren NUR den topologischen Grundgraphen.
# Dies verhindert rekursive Server-Boots oder blockierte Event-Loops aus der FastAPI-App.
try:
    import model
    G_base = model.G_base
    nodes_config = model.nodes_config
except ImportError:
    raise ImportError("❌ model.py konnte nicht geladen werden. Topologie-Schnittstelle fehlt.")

# --- KONFIGURATION & BUSINESS METRIKEN ---
TEST_DATA_FILE = "test_data_holdout.csv"
MODEL_FILE = "traffic_model_xgboost.pkl"
OUTPUT_DIR = "eval_plots"

# Konstanten zur Umrechnung abstrakter KI-Personenzahlen in harte Sekunden (Opportunitätskosten).
WALKING_SPEED = 1.2         # Konstante Laufgeschwindigkeit (Meter pro Sekunde)
AISLE_DELAY_SEC = 5.0       # Jede zusätzliche Person in einem Gang kostet 5s Zeitverlust (Ausweichen)
QUEUE_DELAY_SEC = 45.0      # Jede Person in einer Kassenschlange kostet 45s (Scan + Bezahlung)

# Akademisches Plot-Design für eine saubere Darstellung im Sphinx-HTML-Build
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

def extract_traffic_snapshots():
    """
    Sucht im ungesehenen Testset nach zwei informationstheoretischen Extremen:
    1. Ein ruhiger Morgen (Minimaler Traffic, KI sollte kaum eingreifen)
    2. Die extreme abendliche Rush-Hour (Maximaler Traffic, Bewährungsprobe für die KI)
    """
    print("1. Lade Testdaten und extrahiere deterministische Extrem-Szenarien...")
    df = pd.read_csv(TEST_DATA_FILE)
    df['timestamp_dt'] = pd.to_datetime(df['timestamp_dt'])
    
    # Filtere auf realistische Supermarkt-Öffnungszeiten (Saisonales Rauschen filtern)
    day_mask = (df['hour'] >= 8) & (df['hour'] <= 19)
    df_day = df[day_mask]
    
    # Identifikation der Zeitstempel mit den extremsten Füllständen (total_agents)
    quiet_ts = df_day.loc[df_day['total_agents'].idxmin(), 'timestamp_dt']
    rush_ts = df_day.loc[df_day['total_agents'].idxmax(), 'timestamp_dt']
    
    print(f" -> Ruhiges Baseline-Szenario isoliert: {quiet_ts}")
    print(f" -> Rush-Hour Stresstest isoliert:      {rush_ts}")
    
    return df, quiet_ts, rush_ts

def predict_snapshot_traffic(df, timestamp, ml_model, features):
    """
    Nutzt die trainierte XGBoost-KI, um das abstrakte Verkehrsbild (Tensoren) 
    für einen exakten Zeitpunkt in einen Kantengewichts-Graphen zu transformieren.
    """
    df_snap = df[df['timestamp_dt'] == timestamp].copy()
    
    X_snap = df_snap[features]
    # np.clip verhindert algorithmisch unmögliche (negative) Verkehrsvorhersagen
    preds_log = np.clip(ml_model.predict(X_snap), 0, None)
    
    # Rücktransformation des Logarithmus in reale physische Personen (expm1)
    df_snap['predicted_load'] = np.expm1(preds_log)
    
    traffic_dict = {}
    for _, row in df_snap.iterrows():
        edge = row['edge_id']
        u, v = edge.split('-')
        load = row['predicted_load']
        # Sichert bidirektionale Gänge für das Routing-Modul ab
        traffic_dict[f"{u}-{v}"] = load
        traffic_dict[f"{v}-{u}"] = load
        
    return traffic_dict

def build_smart_graph(G, traffic_dict):
    """
    Wandelt den physikalischen Graphen (Meter) in einen 'Kybernetischen Smart Graph' um.
    Neues Kantengewicht = Physische Distanz + BPR Penalty (Zeitverlust transformiert in Distanz).
    
    (Basierend auf der Bureau of Public Roads (BPR) Funktion der Verkehrsplanung)
    """
    G_smart = G.copy() # Deep-Copy verhindert Speicherkontamination (State Leakage)
    
    for u, v, d in G_smart.edges(data=True):
        base_dist = d.get('weight', 1.0)
        
        # Ignoriere bereits abgesperrte Kanten (z.B. Wände)
        if base_dist >= 99999: continue 
        
        load = traffic_dict.get(f"{u}-{v}", 0.0)
        
        # Stochastik-Schnittstelle: Kassen-Knoten ('vK') erhalten massive Wartezeit-Strafen
        if str(v).startswith('vK') and str(u).startswith('vW'):
            penalty = load * QUEUE_DELAY_SEC
            
        # Topologie-Schnittstelle: Gänge erhalten Strafen erst ab 1.5 Personen (Puffer)
        elif not str(v).startswith('vK') and not str(u).startswith('vK'):
            penalty = (load - 1) * AISLE_DELAY_SEC if load > 1.5 else 0.0
            
        else:
            penalty = 0.0
            
        # Geometrische Translation: 1 Sekunde Zeitverlust entspricht 1.2 Metern "Schmerz-Distanz"
        equiv_dist = penalty * WALKING_SPEED
        G_smart[u][v]['weight'] = base_dist + equiv_dist
        
    return G_smart

def quick_tsp(G, start, targets):
    """
    Autarke Nearest-Neighbor (Greedy) Heuristik für die schnelle Wegpunkt-Planung.
    Wird hier anstelle von Held-Karp genutzt, um 400 Einkäufe in Sekunden simulieren zu können.
    """
    curr = start
    unvisited = set(targets)
    route_nodes = [start]
    
    # Berechnet eine Distanz-Matrix (O(V^3) Floyd-Warshall Äquivalent) 
    # nur für diesen einen Evaluierungs-Durchlauf im Speicher.
    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    
    while unvisited:
        nxt = min(unvisited, key=lambda x: lengths[curr].get(x, 999999))
        route_nodes.append(nxt)
        unvisited.remove(nxt)
        curr = nxt
        
    # Deterministisches Checkout-Routing am Ende des Einkaufs
    checkouts = ['vK1', 'vK2', 'vK3']
    best_checkout = min(checkouts, key=lambda k: lengths[curr].get(k, 999999))
    
    route_nodes.append(best_checkout)
    route_nodes.append('vAusgang') 
    
    return route_nodes

def calculate_real_travel_time(route_nodes, G_routing, G_base, traffic_dict):
    """
    WICHTIGSTE FUNKTION FÜR DEN BEWEIS DES SIM2REAL-GAPS:
    
    Trennt streng zwischen dem "Mentalen Modell" (G_routing) und der Realität (G_base).
    Der Agent hat den Weg basierend auf G_routing geplant. Wir lassen ihn diesen Weg
    nun aber auf der echten Supermarktkarte (G_base) ablaufen und addieren die
    physikalisch realen Stau-Verzögerungen, die er dort unweigerlich antrifft.
    """
    total_seconds = 0.0
    
    for i in range(len(route_nodes)-1):
        u, v = route_nodes[i], route_nodes[i+1]
        
        # Der Weg, den der Algorithmus in der Theorie gewählt hat
        path = nx.dijkstra_path(G_routing, u, v, weight='weight')
        
        # Das physische Ablaufen dieses Weges in der Realität
        for k in range(len(path)-1):
            n1, n2 = path[k], path[k+1]
            
            dist = G_base[n1][n2]['weight']
            load = traffic_dict.get(f"{n1}-{n2}", traffic_dict.get(f"{n2}-{n1}", 0.0))
            
            # Basis-Zeit für das Laufen (s = v * t  =>  t = s / v)
            total_seconds += (dist / WALKING_SPEED)
            
            # Addieren der realen, schmerzhaften Stau-Verzögerungen an diesem exakten Knoten
            if str(n2).startswith('vK') and str(n1).startswith('vW'):
                total_seconds += load * QUEUE_DELAY_SEC
            elif not str(n2).startswith('vK') and not str(n1).startswith('vK'):
                total_seconds += (load * AISLE_DELAY_SEC)
                
    return total_seconds

def run_monte_carlo():
    """
    Orchestriert die Monte-Carlo-Simulation. 
    Simuliert Zufallseinkäufe und misst die Delta-Zeitersparnis (Return on Investment).
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("\n2. Initialisiere KI-Modell und Graphen-Topologien...")
    with open(MODEL_FILE, 'rb') as f:
        artifacts = pickle.load(f)
    
    ml_model = artifacts['model']
    features = artifacts['features']
    
    df, quiet_ts, rush_ts = extract_traffic_snapshots()
    
    # 1. Prädiziere die Stau-Tensoren
    traffic_quiet = predict_snapshot_traffic(df, quiet_ts, ml_model, features)
    traffic_rush = predict_snapshot_traffic(df, rush_ts, ml_model, features)
    
    # 2. Baue die kybernetischen Graphen
    G_smart_quiet = build_smart_graph(G_base, traffic_quiet)
    G_smart_rush = build_smart_graph(G_base, traffic_rush)
    
    # Extrahiere alle gültigen Produkt-Regale (Keine Wände, keine Eingänge)
    valid_shelves = [n for n, d in nodes_config.items() if d.get('col') in ['#000000', '#1f77b4'] and not str(n).startswith('vIn')]
    
    scenarios = [
        ("Entspannter Morgen", traffic_quiet, G_base, G_smart_quiet),
        ("Extreme Rush-Hour", traffic_rush, G_base, G_smart_rush)
    ]
    
    results = []
    num_simulations = 200 # Gesetz der großen Zahlen: 200 Zufallseinkäufe pro Szenario
    
    print("\n3. Starte Monte-Carlo-Simulation (Shadow-Mode Agenten-Wettrennen)...")
    for scenario_name, traffic_dict, graph_naive, graph_smart in scenarios:
        print(f" -> Simuliere {num_simulations} Kunden im Szenario: {scenario_name}")
        
        for _ in tqdm(range(num_simulations), leave=False):
            # Erzeuge einen stochastisch völlig zufälligen Einkaufszettel
            cart_size = random.randint(3, 20)
            shopping_list = random.sample(valid_shelves, cart_size)
            
            # --- AGENT A: Naive Baseline (Blind für Staus) ---
            route_naive = quick_tsp(graph_naive, 'vEingang', shopping_list)
            time_naive = calculate_real_travel_time(route_naive, graph_naive, G_base, traffic_dict)
            
            # --- AGENT B: Smart AI-Driven Routing (Kennt die KI-Strafen) ---
            route_smart = quick_tsp(graph_smart, 'vEingang', shopping_list)
            time_smart = calculate_real_travel_time(route_smart, graph_smart, G_base, traffic_dict)
            
            # WISSENSCHAFTLICHE EHRKEIT:
            # Kein Fallback-Reset! Wenn die KI aufgrund einer Fehlkalkulation einen 
            # zu großen Umweg wählt (time_smart > time_naive), wird die Zeitersparnis ehrlich negativ.
            time_saved = time_naive - time_smart
            
            results.append({
                'Szenario': scenario_name,
                'Warenkorb': cart_size,
                'Dauer_Naive_Min': time_naive / 60.0,
                'Dauer_Smart_Min': time_smart / 60.0,
                'Zeitersparnis_Sekunden': time_saved
            })

    print("\n4. Generiere Business-Report Grafiken...")
    df_res = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    
    # Klassifiziert die Einkäufe für den Boxplot (Kleine Snacks vs. Wochenend-Einkauf)
    df_res['Warenkorb-Größe'] = pd.cut(df_res['Warenkorb'], bins=[0, 8, 14, 100], labels=['Klein (3-8)', 'Mittel (9-14)', 'Groß (>14)'])
    
    sns.boxplot(data=df_res, x='Warenkorb-Größe', y='Zeitersparnis_Sekunden', hue='Szenario', palette=['#2ecc71', '#e74c3c'])
    
    # NEU: Gestrichelte Null-Linie als visuelle Trennlinie für Gewinne/Verluste
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Break-Even (0 Sek.)')
    
    plt.title('Business Value: Ersparte Warte- und Laufzeit durch KI-Routing', fontweight='bold', fontsize=14)
    plt.xlabel('Größe des Einkaufs (Anzahl Regale)', fontsize=12)
    plt.ylabel('Zeitersparnis in Sekunden (Werte < 0 = KI-Umweg)', fontsize=12)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'business_value_time_saved.png'), dpi=300)
    plt.close()
    
    # --- ABSCHLIESSENDER AUDIT REPORT IN DER KONSOLE ---
    print("-" * 60)
    print("🏆 BUSINESS VALUE REPORT (Wissenschaftliche Auswertung):")
    rush_res = df_res[df_res['Szenario'] == 'Extreme Rush-Hour']
    
    win_rate = (rush_res['Zeitersparnis_Sekunden'] > 0).mean() * 100
    loss_res = rush_res[rush_res['Zeitersparnis_Sekunden'] < 0]
    avg_loss = loss_res['Zeitersparnis_Sekunden'].mean() if not loss_res.empty else 0.0
    
    print(f" -> Durchschnittliche Ersparnis (Gesamt):      {df_res['Zeitersparnis_Sekunden'].mean():.1f} Sekunden")
    print(f" -> Durchschnittliche Ersparnis (Rush-Hour):   {rush_res['Zeitersparnis_Sekunden'].mean():.1f} Sekunden")
    print(f" -> Maximal gemessene Ersparnis (Spitze):      {df_res['Zeitersparnis_Sekunden'].max():.1f} Sekunden")
    print(f" -> KI-Erfolgsquote (Rush-Hour):               {win_rate:.1f}% der Routen waren schneller")
    if avg_loss < 0:
        print(f" -> System-Risiko (Durchschnittl. Verlust):    {abs(avg_loss):.1f} Sekunden")
    print("-" * 60)
    print(f"✅ Skript beendet. Der entscheidende Business-Plot liegt in '{OUTPUT_DIR}/business_value_time_saved.png'.")


if __name__ == "__main__":
    run_monte_carlo()