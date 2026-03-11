"""
=========================================================================================
JMU SMART SUPERMARKET: ABM TRAFFIC GENERATOR vFINAL (Queue Spillover Edition)
=========================================================================================
Erzeugt die perfekten Trainingsdaten für das XGBoost Traffic-Modell.
NEU: Physischer "Queue Spillover" - Überlaufende Kassen blockieren reale Regalgänge!
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
    a, b = sorted([str(u), str(v)])
    return f"{a}-{b}"

# =============================================================================
# 1. ENTERPRISE KONFIGURATION
# =============================================================================
class Config:
    OUTPUT_FILE = "smartcart_traffic_training_data.csv"
    CSV_BUFFER_SIZE = 1000 
    TICK_SEC = 5 
    SNAPSHOT_INTERVAL_MIN = 5
    START_DATE = datetime.datetime(2025, 1, 1, 6, 0, 0)
    END_DATE   = datetime.datetime(2026, 1, 1, 0, 0, 0)

    OPEN_HOUR  = 7
    CLOSE_HOUR = 20
    MAX_STORE_CAPACITY = 200

    BASE_DAILY_CUSTOMERS = {0: 800, 1: 800, 2: 800, 3: 800, 4: 1100, 5: 1500, 6: 0}
    HOLIDAYS = {
        datetime.date(2025, 1, 1), datetime.date(2025, 1, 6), 
        datetime.date(2025, 4, 18), datetime.date(2025, 4, 21), 
        datetime.date(2025, 5, 1), datetime.date(2025, 5, 29), 
        datetime.date(2025, 6, 9), datetime.date(2025, 10, 3), 
        datetime.date(2025, 11, 1), datetime.date(2025, 12, 25), datetime.date(2025, 12, 26)
    }
    MONTH_FACTOR = {1: 0.8, 2: 0.8, 3: 0.9, 4: 0.95, 5: 1.0, 6: 1.0, 7: 1.0, 8: 0.9, 9: 1.0, 10: 1.1, 11: 1.2, 12: 1.6}
    CHRISTMAS_BOOST_DAYS = (20, 21, 22, 23, 24)
    CHRISTMAS_BOOST_FACTOR = 1.5
    HOURLY_FACTOR = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.5, 1.4, 1.5, 1.8, 2.2, 2.0, 1.5, 0.8, 0.0, 0.0, 0.0]

    ITEMS_LOG_MEAN = 2.8 
    ITEMS_LOG_SIGMA = 0.6
    ITEMS_PER_SHELF = 2.5 
    WALK_SPEED_BASE = 1.2 
    LOCAL_DENSITY_PENALTY = 0.15 

    CASHIER_SPEEDS = {1: 25.0, 2: 20.0, 3: 18.0}
    PAYMENT_DELAY_BASE = 35.0
    PAYMENT_ANOMALY_PROB = 0.05 
    PAYMENT_ANOMALY_DELAY = 120.0 
    
    OPEN_K3_TRIGGER = 6
    OPEN_K2_TRIGGER = 3
    CLOSE_K2_TRIGGER = 1
    CLOSE_K3_TRIGGER = 0


def compute_daily_customers(date):
    weekday = date.weekday()
    if weekday == 6 or date.date() in Config.HOLIDAYS: return 0
    base = Config.BASE_DAILY_CUSTOMERS[weekday]
    month_factor = Config.MONTH_FACTOR[date.month]
    if date.month == 12 and date.day in Config.CHRISTMAS_BOOST_DAYS:
        month_factor *= Config.CHRISTMAS_BOOST_FACTOR
    return base * month_factor

def hourly_rate(hour, daily_total):
    if hour < Config.OPEN_HOUR or hour >= Config.CLOSE_HOUR: return 0.0
    total_factor = sum(Config.HOURLY_FACTOR[Config.OPEN_HOUR:Config.CLOSE_HOUR])
    fraction = Config.HOURLY_FACTOR[hour] / total_factor
    return daily_total * fraction / 60.0

class StoreEnvironment:
    def __init__(self):
        self.G = G_base.copy()
        self.shopping_nodes = []
        for node, data in nodes_config.items():
            if (data.get('col') == C_BLACK or data.get('col') == C_BLUE) and not str(node).startswith('vIn') and not str(node).startswith('vW') and not str(node).startswith('vK'):
                if self.G.has_node(node):
                    self.shopping_nodes.append(node)
                    
        self.paths = dict(nx.all_pairs_dijkstra_path(self.G, weight='weight'))
        self.dists = dict(nx.all_pairs_dijkstra_path_length(self.G, weight='weight'))
        self._generate_spillover_paths()
        
    def _generate_spillover_paths(self):
        """Berechnet dynamisch, in welche Gänge sich Schlangen physikalisch zurückstauen."""
        deep_nodes = {1: 'vD6', 2: 'vD6', 3: 'vC7'}
        self.spillover_edges = {1: [], 2: [], 3: []}
        
        for lane, deep_node in deep_nodes.items():
            kasse_node = f'vK{lane}'
            if self.G.has_node(deep_node) and self.G.has_node(kasse_node):
                path = self.paths[deep_node][kasse_node]
                spillover_path = path[::-1] # Kasse -> Hinten
                for i in range(len(spillover_path)-1):
                    u, v = spillover_path[i], spillover_path[i+1]
                    edge_key = edge_key_func(u, v)
                    weight = self.G[u][v]['weight'] if self.G.has_edge(u, v) else 3.0
                    cap = max(2, int(weight * 1.5)) # 1.5 Personen mit Wagen pro Meter
                    self.spillover_edges[lane].append((edge_key, cap))
            else:
                self.spillover_edges[lane].append((edge_key_func(f'vW{lane}', kasse_node), 999))

    def get_spillover_edge(self, lane, queue_index):
        """Mappt den Index in der logischen Queue auf eine physische Kante im Markt."""
        edges = self.spillover_edges[lane]
        current_cap = 0
        for edge_key, cap in edges:
            current_cap += cap
            if queue_index < current_cap:
                return edge_key
        return edges[-1][0]

class CustomerAgent:
    def __init__(self, uid, env: StoreEnvironment, current_time):
        self.id = uid
        self.env = env
        self.items = max(1, int(np.random.lognormal(Config.ITEMS_LOG_MEAN, Config.ITEMS_LOG_SIGMA)))
        num_stops = max(1, int(self.items / Config.ITEMS_PER_SHELF))
        
        raw_targets = random.sample(self.env.shopping_nodes, min(num_stops, len(self.env.shopping_nodes)))
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
        self.edge_progress = 0.0 
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
        if self.state == 'DONE' or self.state == 'QUEUE': return # QUEUE wird vom Manager gesetzt

        # --- Warten am Regal ---
        if self.state == 'SHOPPING_ACTION':
            self.wait_timer -= dt
            if self.wait_timer <= 0:
                if self._load_route_to_next_target():
                    self.state = 'SHOPPING_WALK'
                else:
                    self.lane = mgr.get_best_lane()
                    self.queue_node = f'vW{self.lane}'
                    self.checkout_node = f'vK{self.lane}'
                    self._route_to_node(self.queue_node)
                    self.state = 'CHECKOUT_WALK'
            return

        # --- Gehen ---
        if self.state in ['SHOPPING_WALK', 'CHECKOUT_WALK', 'EXITING']:
            if not self.route:
                if self.state == 'SHOPPING_WALK':
                    self.state = 'SHOPPING_ACTION'
                    self.wait_timer = random.uniform(15, 45) 
                elif self.state == 'CHECKOUT_WALK':
                    self.state = 'QUEUE'
                    mgr.join_queue(self)
                elif self.state == 'EXITING':
                    self.state = 'DONE'
                    self.active_edge = None
                return

            nxt_node = self.route[0]
            edge_weight = self.env.G[self.curr_node][nxt_node]['weight']
            self.active_edge = edge_key_func(self.curr_node, nxt_node)

            # Physik: Harter Stau wenn überfüllt!
            agents_on_edge = local_edge_loads.get(self.active_edge, 0)
            local_speed_factor = max(0.05, 1.0 - (agents_on_edge * Config.LOCAL_DENSITY_PENALTY))
            
            walk_distance = Config.WALK_SPEED_BASE * local_speed_factor * dt
            remaining_edge = edge_weight - self.edge_progress

            if walk_distance >= remaining_edge:
                self.curr_node = self.route.popleft()
                self.edge_progress = 0.0
            else:
                self.edge_progress += walk_distance
            return

        # --- Bezahlen an der Kasse ---
        if self.state == 'PAYING':
            self.active_edge = edge_key_func(self.checkout_node, 'vAusgang')
            return

    def leave_store(self):
        self.curr_node = self.checkout_node
        self._route_to_node('vAusgang')
        self.state = 'EXITING'


class CheckoutManager:
    def __init__(self):
        self.queues = {1: deque(), 2: deque(), 3: deque()}
        self.open = {1: True, 2: False, 3: False}
        self.processing = {1: None, 2: None, 3: None}
        self.timer = {1: 0.0, 2: 0.0, 3: 0.0}

    def get_best_lane(self) -> int:
        open_lanes = [k for k, v in self.open.items() if v]
        if not open_lanes: return 1
        return min(open_lanes, key=lambda x: len(self.queues[x]))

    def join_queue(self, agent: CustomerAgent):
        self.queues[agent.lane].append(agent)

    def update(self, dt):
        total_queue = sum(len(q) for q in self.queues.values())

        if not self.open[2] and total_queue >= Config.OPEN_K2_TRIGGER: self.open[2] = True
        if not self.open[3] and total_queue >= Config.OPEN_K3_TRIGGER: self.open[3] = True

        if self.open[2] and len(self.queues[2]) == 0 and self.processing[2] is None and total_queue <= Config.CLOSE_K2_TRIGGER:
            self.open[2] = False
        if self.open[3] and len(self.queues[3]) == 0 and self.processing[3] is None and total_queue <= Config.CLOSE_K3_TRIGGER:
            self.open[3] = False

        for k in [1, 2, 3]:
            if not self.open[k]: continue
            
            if self.processing[k] is None and self.queues[k]:
                cust = self.queues[k].popleft()
                self.processing[k] = cust
                cust.state = 'PAYING'
                
                speed = max(10.0, random.gauss(Config.CASHIER_SPEEDS[k], 3.0))
                scan_time = (cust.items / speed) * 60.0
                delay = Config.PAYMENT_DELAY_BASE
                if random.random() < Config.PAYMENT_ANOMALY_PROB:
                    delay += Config.PAYMENT_ANOMALY_DELAY
                self.timer[k] = scan_time + random.gauss(delay, 5.0)

            if self.processing[k] is not None:
                self.timer[k] -= dt
                if self.timer[k] <= 0:
                    self.processing[k].leave_store()
                    self.processing[k] = None

    def update_queues_physics(self, env):
        """Platziert Agenten physikalisch korrekt in den Gängen (Spillover)"""
        for lane, q in self.queues.items():
            for idx, agent in enumerate(q):
                agent.active_edge = env.get_spillover_edge(lane, idx)

    def reset(self):
        self.queues = {1: deque(), 2: deque(), 3: deque()}
        self.processing = {1: None, 2: None, 3: None}
        self.open = {1: True, 2: False, 3: False}
        self.timer = {1: 0.0, 2: 0.0, 3: 0.0}

# =============================================================================
# 4. SIMULATIONS-HAUPTLOOP
# =============================================================================
def run():
    print("\n" + "=" * 70)
    print("🚀 JMU SMART SUPERMARKET: ABM TRAFFIC GENERATOR vFINAL (SPILLOVER)")
    print("=" * 70)
    
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
            is_closed = (current_time.hour < Config.OPEN_HOUR or current_time.hour >= Config.CLOSE_HOUR)
            is_holiday_today = (current_time.date() in Config.HOLIDAYS or current_time.weekday() == 6)

            if is_closed or is_holiday_today:
                agents = []
                mgr.reset()
                current_time += datetime.timedelta(minutes=1)
                continue

            if current_time.second == 0:
                daily = compute_daily_customers(current_time)
                rate = hourly_rate(current_time.hour, daily)
                n_new = np.random.poisson(rate)
                free_slots = max(0, Config.MAX_STORE_CAPACITY - len(agents))
                n_new = min(n_new, free_slots)
                
                for _ in range(n_new):
                    next_id += 1
                    agents.append(CustomerAgent(next_id, env, current_time))

            # 1. Sammle lokale Dichte VOR dem Move
            current_edge_loads = {}
            for a in agents:
                if a.active_edge:
                    current_edge_loads[a.active_edge] = current_edge_loads.get(a.active_edge, 0) + 1

            # 2. Physics-Update
            for a in agents:
                a.update(Config.TICK_SEC, mgr, current_edge_loads)
                
            # 3. Kassen-Update & Queue-Spillover Mapping
            mgr.update(Config.TICK_SEC)
            mgr.update_queues_physics(env) 

            # 4. Final Aggregation für diesen Tick (Nachdem Spillover greift)
            tick_loads = {}
            for a in agents:
                if a.active_edge:
                    tick_loads[a.active_edge] = tick_loads.get(a.active_edge, 0) + 1
                    
            for edge, load in tick_loads.items():
                interval_loads[edge] = max(interval_loads.get(edge, 0), load)

            agents = [a for a in agents if a.state != 'DONE']

            # Snapshot Speichern 
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
                interval_loads = {} 

                elapsed = (current_time - Config.START_DATE).total_seconds() / 60.0
                progress = (elapsed / total_minutes) * 100
                sys.stdout.write(f"\r⏳ {current_time.strftime('%Y-%m-%d %H:%M')} | Fortschritt: {progress:>5.2f}% | Agenten: {len(agents):>3d} | Q-Total: {q1+q2+q3:>2d}")
                sys.stdout.flush()

            if len(buffer) >= Config.CSV_BUFFER_SIZE:
                with open(Config.OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                    writer.writerows(buffer)
                buffer = []

            current_time += datetime.timedelta(seconds=Config.TICK_SEC)

    except KeyboardInterrupt:
        print("\n\n⚠️ Simulation durch Benutzer abgebrochen.")
    except Exception as e:
        print(f"\n\n❌ Fataler Fehler: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if buffer:
            with open(Config.OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                writer.writerows(buffer)
            print(f"\n💾 Letzte {len(buffer)} Zeilen gesichert.")

if __name__ == "__main__":
    run()