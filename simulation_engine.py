"""
=========================================================================================
JMU SMART SUPERMARKET: ABM TRAFFIC GENERATOR vFINAL (Queue Spillover Edition)
=========================================================================================
Dieses Skript ist ein "Agent-Based Model" (ABM) zur Generierung unserer Trainingsdaten.
Da wir keine echten Kamera-Bewegungsdaten von 200.000 Supermarktkunden haben, 
simulieren wir sie hier. Jeder Kunde (Agent) hat einen eigenen Warenkorb, läuft 
physikalisch durch den virtuellen Laden, steht an der Kasse an und blockiert 
den Weg für andere Kunden.

Diese Simulation läuft für ein komplettes simuliertes Jahr und generiert die CSV-Datei, 
auf der unser XGBoost-Traffic-Modell später trainiert wird.
=========================================================================================
"""

import random
import networkx as nx
import numpy as np
import datetime
import csv
import json
import sys
import os
from collections import deque
import math

# =============================================================================
# 0. MODEL IMPORT & INTEGRITÄTSCHECK
# =============================================================================
# Wir importieren den leeren Supermarkt-Graphen aus unserer model.py.
# So garantieren wir, dass die Simulation auf exakt demselben Grundriss läuft 
# wie das spätere Live-Dashboard.
try:
    import model
    G_base = model.G_base
    nodes_config = model.nodes_config
    edges_config = model.edges_config
    C_BLACK = model.CONFIG.COLOR_BLACK
    C_BLUE = model.CONFIG.COLOR_BLUE
except ImportError as e:
    print(f"❌ KRITISCH: 'model.py' nicht gefunden oder fehlerhaft. Details: {e}")
    sys.exit(1)

def edge_key_func(u, v):
    """
    Hilfsfunktion: Erzeugt einen eindeutigen String für jeden Gang (Kante).
    Wichtig: Der Gang von A nach B ist identisch mit dem Gang von B nach A. 
    Durch sorted() fassen wir beides zusammen.
    """
    a, b = sorted([str(u), str(v)])
    return f"{a}-{b}"

# =============================================================================
# 1. ENTERPRISE KONFIGURATION (Die Parameter der Simulation)
# =============================================================================
# Hier definieren wir die Spielregeln der Simulation (Stochastik & Physik)
class Config:
    OUTPUT_FILE = "smartcart_traffic_training_data.csv"
    CSV_BUFFER_SIZE = 1000 # Schreibt erst ab 1000 Zeilen auf die Festplatte (für Performance)
    TICK_SEC = 5           # Ein Simulationsschritt entspricht 5 echten Sekunden
    SNAPSHOT_INTERVAL_MIN = 5 # Alle 5 Minuten speichern wir ein Foto vom aktuellen Stau
    
    # Simulationszeitraum: Wir simulieren ein komplettes Jahr (2025)
    START_DATE = datetime.datetime(2025, 1, 1, 6, 0, 0)
    END_DATE   = datetime.datetime(2026, 1, 1, 0, 0, 0)

    OPEN_HOUR  = 7
    CLOSE_HOUR = 20
    MAX_STORE_CAPACITY = 200 # Brandschutz/Physik: Bei 200 Leuten ist der Markt dicht

    # --- KUNDEN-GENERIERUNG (Spawn-Raten) ---
    # Wie viele Kunden kommen pro Wochentag? (0 = Montag, 4 = Freitag, 5 = Samstag, 6 = Sonntag/Geschlossen)
    BASE_DAILY_CUSTOMERS = {0: 800, 1: 800, 2: 800, 3: 800, 4: 1100, 5: 1500, 6: 0}
    
    # Feiertage (Der Markt bleibt geschlossen)
    HOLIDAYS = {
        datetime.date(2025, 1, 1), datetime.date(2025, 1, 6), 
        datetime.date(2025, 4, 18), datetime.date(2025, 4, 21), 
        datetime.date(2025, 5, 1), datetime.date(2025, 5, 29), 
        datetime.date(2025, 6, 9), datetime.date(2025, 10, 3), 
        datetime.date(2025, 11, 1), datetime.date(2025, 12, 25), datetime.date(2025, 12, 26)
    }
    
    # Saisonale Effekte (Im Winter gehen mehr Leute drinnen einkaufen als im Hochsommer)
    MONTH_FACTOR = {1: 0.8, 2: 0.8, 3: 0.9, 4: 0.95, 5: 1.0, 6: 1.0, 7: 1.0, 8: 0.9, 9: 1.0, 10: 1.1, 11: 1.2, 12: 1.6}
    CHRISTMAS_BOOST_DAYS = (20, 21, 22, 23, 24)
    CHRISTMAS_BOOST_FACTOR = 1.5 # 50% mehr Kunden vor Weihnachten
    
    # Tages-Verlauf (Eine typische Gauss/Bimodal-Kurve für Supermärkte)
    # Peaks um 11-12 Uhr (Mittagspause) und 17-18 Uhr (Feierabend)
    HOURLY_FACTOR = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.5, 1.4, 1.5, 1.8, 2.2, 2.0, 1.5, 0.8, 0.0, 0.0, 0.0]

    # --- KUNDEN-VERHALTEN & PHYSIK ---
    # Lognormalverteilung für die Anzahl der gekauften Produkte.
    # Lognormal ist perfekt hierfür, da es keine "negativen" Warenkörbe gibt und
    # die meisten Kunden wenig kaufen (z.B. 5 Teile), aber wenige Kunden extrem viel (Wocheneinkauf).
    ITEMS_LOG_MEAN = 2.8 
    ITEMS_LOG_SIGMA = 0.6
    
    # Ein Kunde packt im Schnitt 2.5 Produkte aus einem Regal ein, bevor er zum nächsten geht
    ITEMS_PER_SHELF = 2.5 
    
    WALK_SPEED_BASE = 1.2 # Meter pro Sekunde (Normales Gehtempo)
    LOCAL_DENSITY_PENALTY = 0.15 # Wenn viele Leute in einem Gang sind, wird der Agent langsamer

    # --- KASSEN-STOCHASTIK (Service-Zeiten) ---
    CASHIER_SPEEDS = {1: 25.0, 2: 20.0, 3: 18.0} # Kasse 1 ist die Schnellkasse (25 Artikel/Min)
    PAYMENT_DELAY_BASE = 35.0 # Bezahlen dauert pauschal 35 Sekunden
    
    # 5% Chance, dass etwas schiefgeht (z.B. EC-Gerät defekt, Kunde sucht Kleingeld) -> Kostet 120 Extrasekunden
    PAYMENT_ANOMALY_PROB = 0.05 
    PAYMENT_ANOMALY_DELAY = 120.0 
    
    # Ab wie vielen Wartenden rufen wir Kasse 2 oder 3 auf?
    OPEN_K3_TRIGGER = 6
    OPEN_K2_TRIGGER = 3
    CLOSE_K2_TRIGGER = 1
    CLOSE_K3_TRIGGER = 0


def compute_daily_customers(date):
    """Ermittelt das absolute Limit an Kunden für den aktuellen Tag (Wochentag + Saison + Feiertag)."""
    weekday = date.weekday()
    if weekday == 6 or date.date() in Config.HOLIDAYS: 
        return 0
    base = Config.BASE_DAILY_CUSTOMERS[weekday]
    month_factor = Config.MONTH_FACTOR[date.month]
    if date.month == 12 and date.day in Config.CHRISTMAS_BOOST_DAYS:
        month_factor *= Config.CHRISTMAS_BOOST_FACTOR
    return base * month_factor

def hourly_rate(hour, daily_total):
    """
    Bricht die tägliche Kundenmenge auf die aktuelle Stunde und Minute herunter.
    Damit füttern wir später die Poisson-Verteilung, um die Leute durch die Tür spawnen zu lassen.
    """
    if hour < Config.OPEN_HOUR or hour >= Config.CLOSE_HOUR: 
        return 0.0
    total_factor = sum(Config.HOURLY_FACTOR[Config.OPEN_HOUR:Config.CLOSE_HOUR])
    fraction = Config.HOURLY_FACTOR[hour] / total_factor
    return daily_total * fraction / 60.0

# =============================================================================
# 2. DIE WELT & DER RÄUMLICHE SPILLOVER
# =============================================================================

class StoreEnvironment:
    """
    Verwaltet den Graphen und das physische "Zurückstauen" (Spillover) der Kassen.
    In simplen Modellen stehen an Kasse 1 unendlich viele Leute auf einem Punkt (Knoten).
    In der Realität stauen sie sich physikalisch in den Gang 'vD6' oder 'vC7' zurück
    und blockieren dort Kunden, die eigentlich nur einkaufen wollen.
    """
    def __init__(self):
        self.G = G_base.copy()
        self.shopping_nodes = []
        
        # Identifiziere alle echten Regale (ignorieren Eingänge, Ausgänge, Kassen)
        for node, data in nodes_config.items():
            if (data.get('col') == C_BLACK or data.get('col') == C_BLUE) and not str(node).startswith('vIn') and not str(node).startswith('vW') and not str(node).startswith('vK'):
                if self.G.has_node(node):
                    self.shopping_nodes.append(node)
                    
        # Dijkstra All-Pairs vorberechnen, damit die Agenten nicht in jedem Frame neu rechnen müssen
        self.paths = dict(nx.all_pairs_dijkstra_path(self.G, weight='weight'))
        self.dists = dict(nx.all_pairs_dijkstra_path_length(self.G, weight='weight'))
        self._generate_spillover_paths()
        
    def _generate_spillover_paths(self):
        """
        Berechnet den "Fluchtweg" einer Kassen-Warteschlange. 
        Wenn Kasse 1 voll ist, staut sich die Schlange in den Hauptgang (vD6) zurück.
        """
        deep_nodes = {1: 'vD6', 2: 'vD6', 3: 'vC7'}
        self.spillover_edges = {1: [], 2: [], 3: []}
        
        for lane, deep_node in deep_nodes.items():
            kasse_node = f'vK{lane}'
            if self.G.has_node(deep_node) and self.G.has_node(kasse_node):
                # Wir drehen den Pfad um (Von der Kasse zurück ins Innere des Marktes)
                path = self.paths[deep_node][kasse_node]
                spillover_path = path[::-1] 
                
                for i in range(len(spillover_path)-1):
                    u, v = spillover_path[i], spillover_path[i+1]
                    edge_key = edge_key_func(u, v)
                    weight = self.G[u][v]['weight'] if self.G.has_edge(u, v) else 3.0
                    
                    # Kapazität der Kante berechnen (Ein Einkaufswagen braucht Platz)
                    # Wir gehen von 1.5 Personen pro Laufmeter aus.
                    cap = max(2, int(weight * 1.5)) 
                    self.spillover_edges[lane].append((edge_key, cap))
            else:
                # Fallback, falls Kasse nicht erreichbar ist
                self.spillover_edges[lane].append((edge_key_func(f'vW{lane}', kasse_node), 999))

    def get_spillover_edge(self, lane, queue_index):
        """
        Gibt zurück, in welchem Gang man physikalisch steht, wenn man der 
        X-te Kunde (queue_index) in der Schlange von Kasse Y ist.
        """
        edges = self.spillover_edges[lane]
        current_cap = 0
        for edge_key, cap in edges:
            current_cap += cap
            # Passen wir noch in diesen Gangabschnitt, oder müssen wir eins weiter zurück rücken?
            if queue_index < current_cap:
                return edge_key
        # Wenn der Stau extrem lang ist, stellen wir den Kunden in den allerletzten definierten Gang
        return edges[-1][0]

# =============================================================================
# 3. DIE KI-AGENTEN (KUNDEN)
# =============================================================================

class CustomerAgent:
    """
    Ein autonomer Agent (Kunde) mit eigenem Einkaufszettel und eigenem Gehirn (State Machine).
    """
    def __init__(self, uid, env: StoreEnvironment, current_time):
        self.id = uid
        self.env = env
        
        # 1. Ziehe Lognormalverteilung für Warenkorbgröße
        self.items = max(1, int(np.random.lognormal(Config.ITEMS_LOG_MEAN, Config.ITEMS_LOG_SIGMA)))
        num_stops = max(1, int(self.items / Config.ITEMS_PER_SHELF))
        
        # 2. Zufällige Regale auswählen (Der "Einkaufszettel")
        raw_targets = random.sample(self.env.shopping_nodes, min(num_stops, len(self.env.shopping_nodes)))
        
        # 3. Greedy-TSP für eine grobe Route (Da die Agenten dumm sein sollen, nutzen sie kein Held-Karp)
        self.target_sequence = []
        curr = 'vEingang'
        unvisited = set(raw_targets)
        while unvisited:
            nxt = min(unvisited, key=lambda x: self.env.dists[curr][x])
            self.target_sequence.append(nxt)
            unvisited.remove(nxt)
            curr = nxt
            
        self.state = 'SHOPPING_WALK' 
        self.curr_node = 'vEingang'
        self.edge_progress = 0.0 # Wie weit bin ich schon auf der aktuellen Kante gelaufen?
        self.active_edge = None
        self.wait_timer = 0.0
        
        self.route = deque()
        self._load_route_to_next_target()
        self.lane = None
        self.queue_node = None
        self.checkout_node = None

    def _load_route_to_next_target(self):
        if not self.target_sequence: return False
        target = self.target_sequence.pop(0)
        full_path = self.env.paths[self.curr_node][target]
        self.route = deque(full_path[1:])
        return True

    def _route_to_node(self, destination: str):
        full_path = self.env.paths[self.curr_node][destination]
        self.route = deque(full_path[1:])
        self.edge_progress = 0.0

    def update(self, dt, mgr, local_edge_loads):
        """
        Wird in jedem 5-Sekunden-Tick aufgerufen. Steuert die State-Machine (Zustände) des Kunden.
        """
        if self.state == 'DONE' or self.state == 'QUEUE': return 

        # --- ZUSTAND: Der Agent steht vor einem Regal und packt ein ---
        if self.state == 'SHOPPING_ACTION':
            self.wait_timer -= dt
            if self.wait_timer <= 0: # Fertig mit Einpacken
                # Geht es zum nächsten Regal, oder ist der Zettel leer?
                if self._load_route_to_next_target():
                    self.state = 'SHOPPING_WALK'
                else:
                    # Einkauf abgeschlossen -> Schnellste Kasse suchen!
                    self.lane = mgr.get_best_lane()
                    self.queue_node = f'vW{self.lane}'
                    self.checkout_node = f'vK{self.lane}'
                    self._route_to_node(self.queue_node)
                    self.state = 'CHECKOUT_WALK'
            return

        # --- ZUSTAND: Der Agent läuft einen Gang entlang ---
        if self.state in ['SHOPPING_WALK', 'CHECKOUT_WALK', 'EXITING']:
            
            # Zielknoten erreicht?
            if not self.route:
                if self.state == 'SHOPPING_WALK':
                    self.state = 'SHOPPING_ACTION'
                    self.wait_timer = random.uniform(15, 45) # Suchen/Einpacken dauert 15-45 Sek.
                elif self.state == 'CHECKOUT_WALK':
                    self.state = 'QUEUE' # In die Kassenschlange einreihen
                    mgr.join_queue(self)
                elif self.state == 'EXITING':
                    self.state = 'DONE' # Markt komplett verlassen
                    self.active_edge = None
                return

            # Physik-Berechnung: Fortbewegung auf der Kante
            nxt_node = self.route[0]
            edge_weight = self.env.G[self.curr_node][nxt_node]['weight']
            self.active_edge = edge_key_func(self.curr_node, nxt_node)

            # Physik: Harter Stau wenn überfüllt!
            # Je mehr Leute in diesem Gangabschnitt stehen, desto langsamer kann der Agent gehen.
            agents_on_edge = local_edge_loads.get(self.active_edge, 0)
            local_speed_factor = max(0.05, 1.0 - (agents_on_edge * Config.LOCAL_DENSITY_PENALTY))
            
            walk_distance = Config.WALK_SPEED_BASE * local_speed_factor * dt
            remaining_edge = edge_weight - self.edge_progress

            # Haben wir den Knotenpunkt überschritten?
            if walk_distance >= remaining_edge:
                self.curr_node = self.route.popleft()
                self.edge_progress = 0.0 # Auf der neuen Kante fangen wir bei 0 Metern an
            else:
                self.edge_progress += walk_distance
            return

        # --- ZUSTAND: Bezahlen ---
        if self.state == 'PAYING':
            # Blockiert physikalisch den Bereich direkt an der Kasse
            self.active_edge = edge_key_func(self.checkout_node, 'vAusgang')
            return

    def leave_store(self):
        """Zustands-Wechsel nach dem Bezahlvorgang: Der Kunde geht zur Tür."""
        self.curr_node = self.checkout_node
        self._route_to_node('vAusgang')
        self.state = 'EXITING'

# =============================================================================
# 4. KASSEN-MANAGEMENT (Der Queue-Controller)
# =============================================================================

class CheckoutManager:
    """Verwaltet die Öffnung und Schließung der Kassen sowie die Abfertigungszeiten."""
    def __init__(self):
        self.queues = {1: deque(), 2: deque(), 3: deque()}
        # Kasse 1 ist immer offen, die anderen werden bei Bedarf gerufen.
        self.open = {1: True, 2: False, 3: False}
        self.processing = {1: None, 2: None, 3: None}
        self.timer = {1: 0.0, 2: 0.0, 3: 0.0}

    def get_best_lane(self) -> int:
        """Gibt die Kasse mit der aktuell kürzesten Schlange zurück."""
        open_lanes = [k for k, v in self.open.items() if v]
        if not open_lanes: return 1
        return min(open_lanes, key=lambda x: len(self.queues[x]))

    def join_queue(self, agent: CustomerAgent):
        self.queues[agent.lane].append(agent)

    def update(self, dt):
        total_queue = sum(len(q) for q in self.queues.values())

        # Kassen-Öffnung (Filialleiter-Logik)
        if not self.open[2] and total_queue >= Config.OPEN_K2_TRIGGER: self.open[2] = True
        if not self.open[3] and total_queue >= Config.OPEN_K3_TRIGGER: self.open[3] = True

        # Kassen schließen (Wenn niemand mehr ansteht und die kritische Grenze unterschritten ist)
        if self.open[2] and len(self.queues[2]) == 0 and self.processing[2] is None and total_queue <= Config.CLOSE_K2_TRIGGER:
            self.open[2] = False
        if self.open[3] and len(self.queues[3]) == 0 and self.processing[3] is None and total_queue <= Config.CLOSE_K3_TRIGGER:
            self.open[3] = False

        # Abfertigung der Kunden simulieren
        for k in [1, 2, 3]:
            if not self.open[k]: continue
            
            # Holt den nächsten Kunden aufs Band
            if self.processing[k] is None and self.queues[k]:
                cust = self.queues[k].popleft()
                self.processing[k] = cust
                cust.state = 'PAYING'
                
                # Scanzeit berechnen (Gauß-Verteilung für realistische Abweichungen)
                speed = max(10.0, random.gauss(Config.CASHIER_SPEEDS[k], 3.0))
                scan_time = (cust.items / speed) * 60.0
                delay = Config.PAYMENT_DELAY_BASE
                
                # Bezahlanomalie (z.B. Kunde findet Münze nicht)
                if random.random() < Config.PAYMENT_ANOMALY_PROB:
                    delay += Config.PAYMENT_ANOMALY_DELAY
                self.timer[k] = scan_time + random.gauss(delay, 5.0)

            # Zieht die Zeit ab. Wenn Timer abgelaufen, darf der Kunde gehen.
            if self.processing[k] is not None:
                self.timer[k] -= dt
                if self.timer[k] <= 0:
                    self.processing[k].leave_store()
                    self.processing[k] = None

    def update_queues_physics(self, env):
        """
        Der wichtigste Teil der Simulation: Ordnet wartende Kunden auf physikalisch korrekte Gänge zu.
        Eine Schlange von 20 Personen passt nicht auf den Kassen-Knoten, 
        sondern blockiert die Regalgänge (Spillover-Effekt).
        """
        for lane, q in self.queues.items():
            for idx, agent in enumerate(q):
                agent.active_edge = env.get_spillover_edge(lane, idx)

    def reset(self):
        """Wird nachts aufgerufen, um den Markt für den nächsten Tag vorzubereiten."""
        self.queues = {1: deque(), 2: deque(), 3: deque()}
        self.processing = {1: None, 2: None, 3: None}
        self.open = {1: True, 2: False, 3: False}
        self.timer = {1: 0.0, 2: 0.0, 3: 0.0}

# =============================================================================
# 5. SIMULATIONS-HAUPTLOOP (Die Uhr läuft)
# =============================================================================
def run():
    print("\n" + "=" * 70)
    print("🚀 JMU SMART SUPERMARKET: ABM TRAFFIC GENERATOR (SPILLOVER EDITION)")
    print("=" * 70)
    
    # Schreibe den CSV-Header, falls die Datei noch nicht existiert
    if not os.path.exists(Config.OUTPUT_FILE):
        with open(Config.OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["timestamp", "month", "weekday", "hour", "minute", "is_holiday",
                             "total_agents", "k1_q", "k2_q", "k3_q", "k2_open", "k3_open", "edge_loads_json"])

    env = StoreEnvironment()
    mgr = CheckoutManager()
    agents = []
    next_id = 0
    buffer = []
    interval_loads = {}

    current_time = Config.START_DATE
    total_minutes = (Config.END_DATE - Config.START_DATE).total_seconds() / 60.0

    try:
        while current_time < Config.END_DATE:
            # 1. Ist der Markt geschlossen?
            is_closed = (current_time.hour < Config.OPEN_HOUR or current_time.hour >= Config.CLOSE_HOUR)
            is_holiday_today = (current_time.date() in Config.HOLIDAYS or current_time.weekday() == 6)

            if is_closed or is_holiday_today:
                agents = []
                mgr.reset()
                current_time += datetime.timedelta(minutes=1)
                continue

            # 2. Neue Kunden per Poisson-Verteilung spawnen (Passiert 1x pro Minute)
            if current_time.second == 0:
                daily = compute_daily_customers(current_time)
                rate = hourly_rate(current_time.hour, daily)
                # Poisson ist die perfekte mathematische Funktion für zufällige Ankünfte an einer Tür
                n_new = np.random.poisson(rate)
                # Brandschutz-Limit durchsetzen
                free_slots = max(0, Config.MAX_STORE_CAPACITY - len(agents))
                n_new = min(n_new, free_slots)
                
                for _ in range(n_new):
                    next_id += 1
                    agents.append(CustomerAgent(next_id, env, current_time))

            # 3. Sammle lokale Dichte (Wie viele Leute stehen gerade auf welchem Gang?)
            current_edge_loads = {}
            for a in agents:
                if a.active_edge:
                    current_edge_loads[a.active_edge] = current_edge_loads.get(a.active_edge, 0) + 1

            # 4. Update aller Agenten (Sie reagieren auf den Stau)
            for a in agents:
                a.update(Config.TICK_SEC, mgr, current_edge_loads)
                
            # 5. Kassen-Status aktualisieren & Überlauf (Spillover) berechnen
            mgr.update(Config.TICK_SEC)
            mgr.update_queues_physics(env) 

            # 6. Aggregation der Spitzen-Werte (Max-Pooling) für diesen Tick
            # Wir interessieren uns in der Simulation für den maximalen Stau im Intervall
            tick_loads = {}
            for a in agents:
                if a.active_edge:
                    tick_loads[a.active_edge] = tick_loads.get(a.active_edge, 0) + 1
                    
            for edge, load in tick_loads.items():
                interval_loads[edge] = max(interval_loads.get(edge, 0), load)

            # Agenten löschen, die den Markt verlassen haben
            agents = [a for a in agents if a.state != 'DONE']

            # 7. CSV-Snapshot abspeichern (Z.B. alle 5 Minuten)
            if current_time.minute % Config.SNAPSHOT_INTERVAL_MIN == 0 and current_time.second == 0:
                q1 = len(mgr.queues[1]) + (1 if mgr.processing[1] else 0)
                q2 = len(mgr.queues[2]) + (1 if mgr.processing[2] else 0)
                q3 = len(mgr.queues[3]) + (1 if mgr.processing[3] else 0)

                json_str = json.dumps(interval_loads)

                row = [
                    current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    current_time.month, current_time.weekday(), current_time.hour, current_time.minute,
                    1 if is_holiday_today else 0, len(agents),
                    q1, q2, q3, 1 if mgr.open[2] else 0, 1 if mgr.open[3] else 0,
                    json_str
                ]
                buffer.append(row)
                interval_loads = {} # Reset für das nächste 5-Minuten Fenster

                # Fortschrittsanzeige im Terminal (Wichtig für lange Simulationsläufe)
                elapsed = (current_time - Config.START_DATE).total_seconds() / 60.0
                progress = (elapsed / total_minutes) * 100
                sys.stdout.write(f"\r⏳ {current_time.strftime('%Y-%m-%d %H:%M')} | Fortschritt: {progress:>5.2f}% | Agenten: {len(agents):>3d} | Q-Total: {q1+q2+q3:>2d}")
                sys.stdout.flush()

            # Pufferung auf die Festplatte (Ram-schonend)
            if len(buffer) >= Config.CSV_BUFFER_SIZE:
                with open(Config.OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                    writer.writerows(buffer)
                buffer = []

            current_time += datetime.timedelta(seconds=Config.TICK_SEC)

    except KeyboardInterrupt:
        # Falls der Nutzer Strg+C drückt, sauber beenden und Puffer leeren
        print("\n\n⚠️ Simulation durch Benutzer abgebrochen. Speichere Zwischenstand...")
    except Exception as e:
        print(f"\n\n❌ Fataler Fehler: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Garantiert, dass die restlichen Daten im Puffer nicht verloren gehen
        if buffer:
            with open(Config.OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                writer.writerows(buffer)
            print(f"\n💾 Letzte {len(buffer)} Zeilen erfolgreich gesichert.")

if __name__ == "__main__":
    run()
    
