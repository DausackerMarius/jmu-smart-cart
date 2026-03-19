"""
=========================================================================================
PROJEKT: JMU SMART SUPERMARKET - DASHBOARD FRONTEND
=========================================================================================
Dieses Skript ist das User Interface (UI) unserer Anwendung, geschrieben in Plotly Dash.
Es importiert die Berechnungen aus der 'model.py' (Backend) und visualisiert sie.

Architektur-Hinweis für die Verteidigung:
Die App ist nach dem MVC-Prinzip (Model-View-Controller) aufgebaut:
- Model: Ausgelagert in die 'model.py' (Datenhaltung, Graphen, ML-Pipeline)
- View: Das HTML-Layout in der Funktion 'serve_layout()'
- Controller: Die @app.callback Funktionen, die auf Klicks reagieren
=========================================================================================
"""

import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import networkx as nx
import datetime
import traceback
import pandas as pd
import numpy as np
import pickle
import os
import glob
import hashlib

# =============================================================================
# 1. IMPORT AUS DEM BACKEND (model.py)
# =============================================================================
# Wir importieren hier nur die fertigen Objekte und Methoden aus unserem Backend.
# Das sorgt für eine saubere Trennung und hält das Frontend-Skript übersichtlich.
from model import (
    nodes_config, 
    edges_config, 
    shelves, 
    inv_manager, 
    G_base,
    COLOR_SHELF, 
    COLOR_SHELF_BORDER, 
    COLOR_HIGHLIGHT,
    calculate_hybrid_route,
    ml_predictor,
    QueuingModel,
    node_to_category
)

# Konfiguration der Kassenlogik: Ab wie vielen wartenden Personen öffnen wir eine neue Kasse?
THRESHOLD_OPEN_K3 = 4.0  # Ab 4 Wartenden an Kasse 1 öffnet Kasse 3
THRESHOLD_OPEN_K2 = 8.0  # Ab 8 Wartenden an Kasse 1 öffnet Kasse 2

# Statischer Start- und Zielknoten, da jeder Kunde am Eingang startet und am Ausgang endet
G = G_base
s_node = 'vEingang'
e_node = 'vAusgang'

# Wir bereiten hier einmalig die Dropdown-Liste für das Admin-Panel vor.
# Wir filtern versteckte Knoten (wie 'vInCorner') heraus, damit der Admin keine 
# Produkte in unsichtbare Regale sortiert.
GLOBAL_NODE_OPTIONS = [
    {'label': f"Regal/Knoten: {k}", 'value': k} 
    for k in nodes_config.keys() 
    if 'pos' in nodes_config[k] and not nodes_config[k].get('hidden', False)
]

# =============================================================================
# 2. BUSINESS LOGIC LAYER (Stau-Simulation via XGBoost)
# =============================================================================

class TrafficSimulationEngine:
    """
    Diese Klasse kapselt unser trainiertes Machine Learning Modell (XGBoost) für die Stau-Vorhersage.
    Warum liegt das hier und nicht in der model.py?
    Weil das Traffic-Modell stark an das Frontend gekoppelt ist (der User wählt im 
    Dashboard eine Uhrzeit aus, und diese Klasse berechnet sofort den passenden Stau dazu).
    """
    _model_cache = None

    @classmethod
    def initialize(cls):
        """
        Sucht beim Starten des Servers nach der neuesten Modelldatei (.pkl).
        Das Modell wird in den RAM geladen, damit die Vorhersagen später in Millisekunden ablaufen.
        """
        pkl_files = glob.glob("*.pkl")
        traffic_models = [f for f in pkl_files if "traffic" in f]

        if traffic_models:
            # Wir greifen immer dynamisch das aktuellste Modell ab (falls wir es neu trainiert haben)
            best_model_path = max(traffic_models, key=os.path.getmtime)
            try:
                with open(best_model_path, "rb") as f:
                    cls._model_cache = pickle.load(f)
                print(f"✅ KI-MODELL GELADEN: {best_model_path}")
            except Exception as e:
                print(f"❌ Fehler beim Laden des Models: {e}")
        else:
            print("❌ KEIN TRAFFIC-MODELL GEFUNDEN! Bitte erst das Training starten.")

    @classmethod
    def get_traffic_state(cls, sim_dt: datetime.datetime) -> tuple[dict, int]:
        """
        Füttert das XGBoost-Modell mit der gewählten Uhrzeit und berechnet den Stau.
        
        Args:
            sim_dt (datetime): Der vom User im UI-Kalender ausgewählte Zeitpunkt.
            
        Returns:
            tuple[dict, int]: Ein Dictionary mit der vorhergesagten Personenanzahl 
                              pro Gangkante und die Gesamtzahl der Kunden im Markt.
        """
        if not cls._model_cache:
            return {}, 0

        model = cls._model_cache['model']
        encoder = cls._model_cache['encoder']
        features_list = cls._model_cache['features']
        edge_list = cls._model_cache['edge_list']

        # 1. Feature Engineering: Zeitmerkmale extrahieren
        month = sim_dt.month
        day = sim_dt.day
        hour = sim_dt.hour
        weekday = sim_dt.weekday()

        is_weekend = 1 if weekday >= 4 else 0
        is_rush_hour = 1 if 16 <= hour <= 19 else 0
        # Spezielle Regel für Weihnachten (Peak-Saison)
        is_christmas = 1 if (month == 12 and day >= 20) else 0

        # 2. Heuristische Agenten-Simulation (Wie voll ist der Markt grob?)
        if is_christmas and 9 <= hour <= 20:
            total_agents = 200
            total_queue = 180
            open_registers = 3
        elif is_rush_hour:
            total_agents = 120
            total_queue = 50
            open_registers = 2
        else:
            total_agents = 30
            total_queue = 3
            open_registers = 1

        queue_pressure = total_queue / open_registers
        
        # 3. Räumliche Merkmale aufbauen (Ist der Gang eine Kasse oder ein Regal?)
        df_pred = pd.DataFrame({'edge_id': edge_list})
        df_pred['edge_id_enc'] = encoder.transform(df_pred['edge_id'])
        df_pred['is_checkout_zone'] = df_pred['edge_id'].str.contains(r'vK|vW').astype(int)
        df_pred['is_main_aisle'] = df_pred['edge_id'].str.contains(r'vD').astype(int)
        df_pred['is_shelf_aisle'] = df_pred['edge_id'].str.contains(r'vA|vB|vC').astype(int)

        # 4. Feature Pool zusammenbauen
        # WICHTIG: Wir kodieren die Zeit zyklisch über Sinus/Cosinus. 
        # Warum? Für einen Baum-Algorithmus ist 23 Uhr mathematisch weit weg von 1 Uhr. 
        # Durch den Sinus-Kreis "versteht" das Modell, dass diese Zeiten eigentlich direkt nebeneinander liegen.
        feature_pool = {
            'edge_id_enc': df_pred['edge_id_enc'],
            'is_checkout_zone': df_pred['is_checkout_zone'],
            'is_main_aisle': df_pred['is_main_aisle'],
            'is_shelf_aisle': df_pred['is_shelf_aisle'],
            'is_holiday': is_christmas,
            'is_weekend': is_weekend,
            'is_rush_hour': is_rush_hour,
            'total_agents': total_agents,
            'total_queue': total_queue,
            'open_registers': open_registers,
            'queue_pressure': queue_pressure,
            'queue_momentum': 5.0 if queue_pressure > 15 else 0.0,
            'fill_rate': 2.0 if total_agents > 100 else 0.0,
            'spillover_risk': queue_pressure * df_pred['is_main_aisle'],
            'shelf_density': total_agents * df_pred['is_shelf_aisle'],
            'hour_sin': np.sin(2 * np.pi * hour / 24), 
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'day_sin': np.sin(2 * np.pi * weekday / 7), 
            'day_cos': np.cos(2 * np.pi * weekday / 7)
        }

        # 5. Vorhersage generieren
        X_pred = pd.DataFrame({f: feature_pool.get(f, 0) for f in features_list})
        
        # Das Modell sagt logarithmierte Werte voraus (um unlogische negative Kundenzahlen zu vermeiden).
        # Wir wandeln diese mit expm1 wieder in greifbare Personenanzahlen um.
        log_preds = np.clip(model.predict(X_pred), 0, None) 
        real_preds = np.expm1(log_preds)
        
        return dict(zip(edge_list, real_preds)), total_agents

# Modell laden, bevor der erste User die Website betritt
TrafficSimulationEngine.initialize()

# =============================================================================
# 3. APP LAYOUT (Frontend-Struktur)
# =============================================================================
# Wir nutzen Bootstrap (DBC), um ein responsives Layout zu bauen, das auf 
# PCs und Handys funktioniert.

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX, dbc.icons.FONT_AWESOME])
app.title = "JMU Market AI: Optimization Dashboard"

def serve_layout():
    """
    Generiert das HTML-Layout.
    ARCHITEKTUR-HINWEIS: Wir verpacken das Layout in eine Funktion, anstatt es 
    einer statischen Variable zuzuweisen. Warum?
    Damit die aktuelle Uhrzeit (datetime.now()) bei JEDEM Seiten-Reload (F5) 
    frisch gezogen wird und nicht auf der Startzeit des Servers einfriert.
    """
    current_dt_str = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
    
    return html.Div([
        dcc.Location(id='url', refresh=False),
        
        # ---------------------------------------------------------
        # STATELESS ARCHITECTURE (Zustandslosigkeit)
        # ---------------------------------------------------------
        # Wir speichern den Warenkorb (cart-store) im Browser des Nutzers (Client-Side).
        # Würden wir dafür eine Python-Variable auf dem Server nutzen, würden sich alle 
        # Besucher der Webseite denselben Warenkorb teilen und es käme zum Chaos.
        # So bleibt unser Server "zustandslos" und ist theoretisch unendlich skalierbar.
        dcc.Store(id='cart-store', data=[]),
        dcc.Store(id='mode-store', data='static'),

        # --- Navbar (Kopfzeile) ---
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink([html.I(className="fa-solid fa-route me-1"), "1. Statische Baseline"], href="/", id="nav-static")),
                dbc.NavItem(dbc.NavLink([html.I(className="fa-solid fa-brain me-1"), "2. Dynamische KI-Simulation"], href="/ai", id="nav-ai")),
            ],
            brand=html.Span([html.I(className="fa-solid fa-shopping-cart me-2"), "JMU Market AI Optimizer"]),
            color="dark",
            dark=True,
            className="mb-4 shadow-sm sticky-top"
        ),

        # --- Haupt-Container ---
        dbc.Container([
            dbc.Row([   
                # LINKE SPALTE: Steuerung und Warenkorb (1/3 der Breite)
                dbc.Col([
                    html.H4("Warenkorb & Routing", className="text-primary fw-bold mb-3"),
                    
                    dbc.Card([
                        dbc.CardBody([
                            html.H6(html.B("1. Produkt hinzufügen")),
                            dbc.InputGroup([
                                dbc.Input(id='input-prod-name', type='text', placeholder='z.B. Milch, Brot...', debounce=False),
                                dbc.Button([html.I(className="fa-solid fa-plus")], id='btn-add-item', color="primary"),
                            ], className="mb-2"),
                            html.Div(id='add-status-msg', className="text-muted small mb-3"),
                            
                            html.Hr(),
                            
                            html.H6(html.B("2. Aktueller Warenkorb")),
                            html.Div(
                                id='cart-list-container', 
                                className="bg-light border rounded p-2 mb-2 shadow-sm",
                                style={'maxHeight': '260px', 'overflowY': 'auto'} # Ermöglicht Scrollen bei vielen Items
                            ),
                            
                            dbc.Row([
                                dbc.Col(dbc.Button([html.I(className="fa-solid fa-rotate-left me-1"), "Undo"], id='btn-undo-cart', color="outline-secondary", size="sm", className="w-100"), width=6),
                                dbc.Col(dbc.Button([html.I(className="fa-solid fa-trash me-1"), "Leeren"], id='btn-clear-cart', color="outline-danger", size="sm", className="w-100"), width=6),
                            ], className="mb-3"),
                            
                            # Zeit-Auswahl (wird im statischen Modus über Callbacks unsichtbar gemacht)
                            html.Div(id="ai-time-wrapper", children=[
                                html.Hr(),
                                html.H6(html.B([html.I(className="fa-regular fa-clock me-1"), "Simulations-Zeitpunkt"])),
                                dbc.Input(id='sim-date-picker', type='datetime-local', value=current_dt_str, className="mb-3 shadow-sm"),
                            ]),
                            
                            dbc.Button([html.I(className="fa-solid fa-person-walking-arrow-right me-2"), "Route optimieren"], 
                                       id='btn-calc-route', color="success", className="w-100 fw-bold shadow-lg mb-2", size="lg"),
                            
                            html.Hr(className="my-4"),
                            
                            # --- Admin Menü (Versteckt für normale User) ---
                            html.Div(
                                dbc.Button([html.I(className="fa-solid fa-gear me-1"), "Admin Workspace"], 
                                           id="btn-admin-modal-open", color="link", size="sm", className="text-muted text-decoration-none p-0"),
                                className="text-end"
                            ),
                            
                            dbc.Modal([
                                dbc.ModalHeader(dbc.ModalTitle([html.I(className="fa-solid fa-lock me-2"), "Admin Authentifizierung"])),
                                dbc.ModalBody([
                                    dbc.Input(id="admin-password-input", type="password", placeholder="Passwort eingeben...", autoFocus=True),
                                    html.Div(id="admin-login-error", className="text-danger small mt-2 fw-bold")
                                ]),
                                dbc.ModalFooter(dbc.Button("Login", id="btn-admin-login", color="dark")),
                            ], id="modal-admin-login", is_open=False, centered=True),

                            dbc.Collapse([
                                dbc.Card([
                                    dbc.CardHeader("KI-gestütztes Produktmanagement", className="bg-dark text-white small p-2 fw-bold"),
                                    dbc.CardBody([
                                        dbc.Tabs([
                                            dbc.Tab(label="Neues Produkt", children=[
                                                html.Br(),
                                                html.Label("Produktname", className="small fw-bold text-muted mb-0"),
                                                dbc.InputGroup([
                                                    dbc.Input(id="admin-add-name", placeholder="z.B. Veganes Schnitzel"),
                                                    dbc.Button(
                                                        [dbc.Spinner(html.Div(id="loading-spinner", className="d-none"), size="sm", spinner_class_name="me-1"), "KI-VORSCHLAG"], 
                                                        id="btn-ai-suggest", color="info", className="fw-bold text-white"
                                                    ),
                                                ], className="mb-2 shadow-sm"),
                                                
                                                html.Div(id="ai-suggest-result", className="mb-3"),
                                                
                                                html.Label("Physisches Regal (Knoten)", className="small fw-bold text-muted mb-0"),
                                                dbc.Select(id="admin-add-node", options=GLOBAL_NODE_OPTIONS, placeholder="Manuelle Auswahl...", className="mb-2"),
                                                
                                                dbc.Row([
                                                    dbc.Col([
                                                        html.Label("Marke", className="small fw-bold text-muted mb-0"),
                                                        dbc.Input(id="admin-add-brand", placeholder="z.B. Rügenwalder", className="mb-2")
                                                    ], width=6),
                                                    dbc.Col([
                                                        html.Label("Kategorie", className="small fw-bold text-muted mb-0"),
                                                        dbc.Input(id="admin-add-cat", placeholder="z.B. Vegan", className="mb-2")
                                                    ], width=6)
                                                ]),
                                                
                                                dbc.Checkbox(id="admin-needs-review", label="Review erforderlich", value=False, className="mb-3 text-warning fw-bold"),
                                                dbc.Button("In Datenbank speichern", id="btn-admin-save", color="success", size="sm", className="w-100 fw-bold shadow-sm"),
                                                html.Div(id="admin-add-feedback", className="mt-2 small text-center")
                                            ]),
                                            dbc.Tab(label="Löschen", children=[
                                                html.Br(),
                                                dcc.Dropdown(id="admin-delete-select", placeholder="Produkt suchen...", className="mb-3"),
                                                dbc.Button([html.I(className="fa-solid fa-trash me-2"), "Dauerhaft löschen"], id="btn-admin-delete", color="danger", size="sm", className="w-100 shadow-sm"),
                                                html.Div(id="admin-delete-feedback", className="mt-2 small text-center")
                                            ])
                                        ])
                                    ], className="p-2")
                                ], className="mt-3 border-dark shadow")
                            ], id="admin-panel-collapse", is_open=False),
                        ])
                    ], className="shadow-sm border-0 bg-white"),
                ], width=4),
                
                # RECHTE SPALTE: Plotly Graphik und Kartendarstellung (2/3 der Breite)
                dbc.Col([
                    html.Div(id="ai-map-wrapper", children=[
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label([html.I(className="fa-solid fa-eye me-1"), "Karten-Ansicht:"], className="fw-bold me-2 text-primary"),
                                        dbc.RadioItems(
                                            id="view-mode",
                                            options=[
                                                {"label": " KI vs Baseline", "value": "all"},
                                                {"label": " Nur Baseline", "value": "base"},
                                                {"label": " Nur KI", "value": "smart"},
                                                {"label": " Stau-Heatmap", "value": "heat"}
                                            ],
                                            value="all", inline=True, className="d-inline-block fw-bold small"
                                        )
                                    ], width=7, className="d-flex align-items-center border-end"),
                                    
                                    dbc.Col([
                                        html.Label("Stau-Toleranz (Ausweichen ab X Personen):", className="fw-bold text-danger mb-0 small"),
                                        dcc.Slider(
                                            id='heatmap-threshold', min=1, max=15, step=1, value=4,
                                            marks={i: str(i) for i in range(0, 16, 3)}, className="p-0 mt-1"
                                        )
                                    ], width=5)
                                ])
                            ], className="p-2 bg-white")
                        ], className="mb-3 shadow-sm border-0")
                    ]),

                    dbc.Card([
                        dcc.Graph(id='floor-plan', style={'height': '60vh'}, config={'displayModeBar': False})
                    ], className="shadow-lg border-0 mb-3"),
                    
                    # Platzhalter für die dynamischen Text-Ergebnisse (Routen-Vergleich)
                    html.Div(id='route-info-container'),
                    html.Div(id='path-details-container', className="mt-3")
                ], width=8)
            ])
        ], fluid=True, className="pt-3 pb-5")
    ], className="bg-light", style={"minHeight": "100vh"})

app.layout = serve_layout


# =============================================================================
# 4. CONTROLLER: URL-ROUTING & WARENKORB-LOGIK (Callbacks)
# =============================================================================

@app.callback(
    [Output('mode-store', 'data'), Output('ai-time-wrapper', 'style'),
     Output('ai-map-wrapper', 'style'), Output('nav-static', 'active'), Output('nav-ai', 'active')],
    [Input('url', 'pathname')]
)
def handle_routing(pathname):
    """
    Steuert unsere Single-Page-Application (SPA).
    Schaltet zwischen dem dummen Modus (Baseline) und dem KI-Modus um, 
    indem Menüs ein- oder ausgeblendet werden, OHNE die Seite neu zu laden.
    """
    if pathname == '/ai': 
        return 'dynamic', {'display': 'block'}, {'display': 'block'}, False, True
    else: 
        return 'static', {'display': 'none'}, {'display': 'none'}, True, False

@app.callback(
    [Output('cart-store', 'data'), Output('input-prod-name', 'value'), Output('add-status-msg', 'children')],
    [Input('btn-add-item', 'n_clicks'), Input('btn-clear-cart', 'n_clicks'),
     Input('btn-undo-cart', 'n_clicks'), Input('input-prod-name', 'n_submit')],
    [State('input-prod-name', 'value'), State('cart-store', 'data')]
)
def manage_cart(n_add, n_clear, n_undo, n_submit, text_input, current_cart):
    """
    Verwaltet den Warenkorb-Status. Greift auf die model.py zu, um Produkte 
    mit Fehlertoleranz (Levenshtein) zu suchen.
    """
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    current_cart = current_cart or []

    if trigger_id == 'btn-clear-cart':
        return [], "", dbc.Alert("Warenkorb geleert.", color="info", className="py-1 px-2 small")
        
    if trigger_id == 'btn-undo-cart':
        if current_cart:
            removed = current_cart.pop()
            return current_cart, "", dbc.Alert(f"Rückgängig: '{removed['name']}' entfernt.", color="warning", className="py-1 px-2 small")
        return current_cart, "", dbc.Alert("Warenkorb ist leer.", color="info", className="py-1 px-2 small")
    
    if (trigger_id == 'btn-add-item' or trigger_id == 'input-prod-name') and text_input:
        # Backend-Suche aktivieren
        found = inv_manager.find_product(text_input)
        if found:
            node_id, real_name, brand = found
            
            # Robuste Preis-Sicherung (Da JSON-Datenbanken oft unsaubere Datentypen liefern)
            price_str = "0.00 €"
            search_target = str(real_name).strip().lower()
            
            for items_list in inv_manager.stock.values():
                for p in items_list:
                    if str(p.get('name', '')).strip().lower() == search_target:
                        raw_price = p.get('price', '0.00')
                        if isinstance(raw_price, (int, float)):
                            price_str = f"{raw_price:.2f} €"
                        else:
                            price_str = str(raw_price).strip()
                            if "€" not in price_str:
                                price_str += " €"
                        break
                if price_str != "0.00 €":
                    break
            
            # Duplikat-Schutz
            if any(item['name'] == real_name for item in current_cart):
                return current_cart, "", dbc.Alert(f"Bereits im Wagen.", color="info", className="py-1 px-2 small")
            
            current_cart.append({'node': node_id, 'name': real_name, 'brand': brand, 'slot': node_id, 'price': price_str})
            return current_cart, "", dbc.Alert(f"'{real_name}' hinzugefügt", color="success", className="py-1 px-2 small")
        else:
            return current_cart, text_input, dbc.Alert(f"Nicht gefunden.", color="danger", className="py-1 px-2 small")
            
    return current_cart, dash.no_update, ""


# =============================================================================
# 5. HILFSFUNKTIONEN FÜR DIE VISUALISIERUNG (Graphen-Mathematik)
# =============================================================================

def calculate_turn_direction(p1, p2, p3):
    """
    Berechnet die Abbiegerichtung des Kunden an einer Gabelung mittels 2D-Kreuzprodukt.
    
    Wir wandeln die Laufwege in Vektoren um. Das Kreuzprodukt dieser Vektoren 
    gibt uns eine Z-Achse. Ist das Vorzeichen der Z-Achse positiv, rotiert der Pfad 
    gegen den Uhrzeigersinn (Links). Ist es negativ, rotiert er mit dem Uhrzeigersinn (Rechts).
    Das ist robuster als reine Winkelberechnungen.
    """
    v1 = (p2[0]-p1[0], p2[1]-p1[1])
    v2 = (p3[0]-p2[0], p3[1]-p2[1])
    
    cp = v1[0]*v2[1] - v1[1]*v2[0]
    if cp > 0.5: return "Links abbiegen ↩️"
    elif cp < -0.5: return "Rechts abbiegen ↪️"
    return "Geradeaus ⬆️"

def build_true_plot_path(sequence, graph):
    """
    Die TSP-Algorithmen geben uns nur die Reihenfolge der Regale.
    Diese Funktion füllt die Strecken zwischen den Regalen mit echten Wegpunkten (Dijkstra)
    auf, damit die gezeichnete Linie im Frontend sauber durch die Gänge läuft und nicht 
    durch Wände schneidet.
    """
    if not sequence: return [], []
    px, py = [], []
    
    for i in range(len(sequence)-1):
        u, v = sequence[i], sequence[i+1]
        try:
            sub_path = nx.dijkstra_path(graph, u, v, weight='weight')
            for j, node in enumerate(sub_path):
                # Verhindert, dass Verbindungsknoten doppelt in die Liste rutschen
                if i > 0 and j == 0: continue 
                
                pos = nodes_config[node]['pos']
                px.append(pos[0])
                py.append(pos[1])
        except: pass
    return px, 

# =============================================================================
# 6. KERN-LOGIK: VISUALISIERUNG & ROUTEN-VERGLEICH (Baseline vs. KI)
# =============================================================================

@app.callback(
    [Output('cart-list-container', 'children'), Output('floor-plan', 'figure'),
     Output('route-info-container', 'children'), Output('path-details-container', 'children')], 
    [Input('cart-store', 'data'), Input('btn-calc-route', 'n_clicks'),
     Input('view-mode', 'value'), Input('heatmap-threshold', 'value'), 
     Input('mode-store', 'data'), Input('sim-date-picker', 'value')] 
)
def update_visuals(cart_data, n_clicks, view_mode, heat_thresh, current_mode, date_value):
    """
    Diese Funktion ist der Haupt-Render-Loop des Dashboards. 
    Sie wird immer dann getriggert, wenn der Nutzer mit der Oberfläche interagiert 
    (z.B. ein Produkt in den Korb legt oder den 'Route optimieren'-Button drückt).
    Hier berechnen wir die Routen, vergleichen die KI mit der Baseline und 
    bauen den Plotly-Graphen auf.
    """
    
    # Schutz vor Initialisierungsfehlern: Falls der Korb leer ist, setzen wir eine leere Liste.
    cart_data = cart_data or []
    
    # Wir extrahieren alle Ziel-Regale aus dem Warenkorb. 
    # dict.fromkeys() ist ein bewährter Python-Trick, um Duplikate zu entfernen 
    # (falls der User 3x Milch kauft, müssen wir das Regal trotzdem nur 1x besuchen), 
    # behält aber im Gegensatz zu set() die chronologische Reihenfolge bei.
    target_nodes = list(dict.fromkeys([item['node'] for item in cart_data]))
    
    def format_path_steps(seq, cart, is_ai=False):
        """
        Übersetzt die abstrakten mathematischen Knoten-IDs (z.B. 'vA10') in 
        eine menschenlesbare, schrittweise Turn-by-Turn Navigation für das UI.
        """
        steps = []
        for i in range(len(seq)):
            n = seq[i]
            lbl = n
            desc = "Folge dem Gang"
            direction = ""
            
            # Richtungsanweisung an Gabelungen berechnen.
            # Wir rufen hier die Vektor-Hilfsfunktion (Kreuzprodukt) auf, um zu prüfen, 
            # ob der Weg am aktuellen Knoten einen Knick nach links oder rechts macht.
            if i > 0 and i < len(seq)-1:
                p1 = nodes_config[seq[i-1]]['pos']
                p2 = nodes_config[n]['pos']
                p3 = nodes_config[seq[i+1]]['pos']
                direction = calculate_turn_direction(p1, p2, p3)
                if direction != "Geradeaus ⬆️": 
                    desc = html.Span(direction, className="text-primary fw-bold")

            # UX-Optimierung: Dem Nutzer anzeigen, was er am jeweiligen Wegpunkt tun muss.
            items_here = [item for item in cart if item['node'] == n]
            if items_here:
                lbl = html.Span(f"Regal {n}", className="text-warning fw-bold")
                item_names = ", ".join([i['name'] for i in items_here])
                desc = html.Span([html.I(className="fa-solid fa-hand-holding-hand me-1"), f"Einpacken: {item_names}"], className="text-success fw-bold")
            elif n in ['vK1', 'vK2', 'vK3']: 
                lbl = f"Kasse {n}"
                desc = html.Span([html.I(className="fa-solid fa-credit-card me-1"), "Bezahlen"], className="text-danger fw-bold")
            elif n == 'vEingang': 
                lbl, desc = "Start", "Markt betreten"
            elif n == 'vAusgang': 
                lbl, desc = "Ziel", "Einkauf abgeschlossen"
            
            # Generiert das finale HTML-Snippet für diesen konkreten Schritt
            steps.append(html.Div([
                html.Span(f"{i+1}.", className="badge bg-secondary me-2"),
                html.B(lbl), html.Span([" - ", desc], className="text-muted small ms-1")
            ], className="mb-2 border-bottom pb-1"))
        return steps

    # ---------------------------------------------------------
    # DATA CLEANSING: Robustes Preis-Parsing
    # ---------------------------------------------------------
    # In der Praxis sind Datenbankeinträge oft unsauber gepflegt (z.B. "1.99€ " 
    # statt eines reinen Floats). Wenn wir blind addieren, stürzt die App ab. 
    # Wir säubern den String, filtern Buchstaben heraus und korrigieren 
    # landesspezifische Kommasetzungen, bevor wir konvertieren.
    total_sum = 0.0
    list_items = []
    
    for i in cart_data:
        p_str = i.get('price', '0.00 €')
        try:
            # Behalte nur Zahlen, Punkte und Kommas im String
            clean_str = ''.join(c for c in str(p_str) if c.isdigit() or c in '.,')
            if clean_str:
                # Falls Tausendertrennzeichen (1.000,99) existieren, den Punkt entfernen
                if clean_str.count('.') > 0 and clean_str.count(',') > 0:
                    clean_str = clean_str.replace('.', '')
                # Deutsches Komma in englischen Dezimalpunkt umwandeln und addieren
                val = float(clean_str.replace(',', '.'))
                total_sum += val
        except: 
            pass # Silent Fail: Wir ignorieren fehlerhafte Einträge, anstatt die App abstürzen zu lassen
        
        # HTML für die Darstellung im Warenkorb-Fenster
        list_items.append(html.Div([
            html.Div([
                html.B(i['name']), 
                dbc.Badge(i['slot'], color="info", className="ms-2"),
                html.Span(p_str, className="float-end text-success fw-bold")
            ]),
            html.Small(f"Marke: {i.get('brand', '-')} | Regal: {i['node']}", className="text-muted")
        ], className="border-bottom py-1 px-1"))
        
    if not list_items: 
        list_items = [html.Div("Warenkorb leer.", className="text-muted small p-2 text-center")]
    else:
        list_items.append(html.Div([
            html.B("Gesamtsumme: ", className="fs-6 text-dark"),
            html.B(f"{total_sum:.2f} €", className="float-end text-success fs-5")
        ], className="pt-2 mt-2 border-top text-end"))

    # ---------------------------------------------------------
    # VISUALISIERUNG: Supermarkt-Grundriss (Plotly) aufbauen
    # ---------------------------------------------------------
    fig = go.Figure()
    edge_x, edge_y = [], []
    
    # 1. Layer (Hintergrund): Die Laufwege (Gänge) aus dem Graphen zeichnen.
    for u, v, w in edges_config:
        if u in nodes_config and v in nodes_config:
            p1, p2 = nodes_config[u]['pos'], nodes_config[v]['pos']
            # Der 'None' Wert ist ein Trick für Plotly, um die Linie hier abzubrechen.
            # Ansonsten würde Plotly alle Gänge wie mit einem durchgehenden Stift verbinden.
            edge_x += [p1[0], p2[0], None] 
            edge_y += [p1[1], p2[1], None]
            
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='#e0e0e0', width=2), hoverinfo='skip', name='Gänge'))
    
    # 2. Layer: Die physischen Regale als Flächen (Hindernisse) darüberlegen.
    for s in shelves: 
        fig.add_shape(type="rect", x0=s['x0'], y0=s['y0'], x1=s['x1'], y1=s['y1'], line=dict(color=COLOR_SHELF_BORDER, width=2), fillcolor=COLOR_SHELF, opacity=0.8, layer="above")

    # Standard-Platzhalter für die Info-Kästen definieren
    info_box = html.Div("Fügen Sie Produkte hinzu und klicken Sie auf 'Route optimieren'.", className="text-center text-muted mt-4")
    path_box = html.Div()
    optimized_order = []

    # Basis-Parameter für unsere Laufzeit-Simulation
    WALKING_SPEED = 1.0  # Annahme: Der Kunde geht 1 Meter pro Sekunde.
    TRAFFIC_DELAY = 7.5  # Strafzeit in Sekunden für jede störende Person in einem Gang (Ausweichen kostet Zeit).

    # Wenn wir Produkte im Korb haben und der Button geklickt wurde:
    if target_nodes and n_clicks:
        try:
            if current_mode == 'static':
                # =========================================================
                # 1. STATISCHER MODUS (Baseline - Keine KI)
                # =========================================================
                # Wir berechnen die Route rein nach Distanz, ohne Rücksicht auf Stau.
                G_static = G_base.copy()
                _, _, dist_s, seq_s, _ = calculate_hybrid_route(G_static, s_node, target_nodes, e_node)
                
                # Wir füllen den Pfad mit Zwischenschritten auf, damit die Linie nicht durch Regale schneidet.
                px_s, py_s = build_true_plot_path(seq_s, G_static)
                optimized_order = seq_s
                
                fig.add_trace(go.Scatter(x=px_s, y=py_s, mode='lines', name='Statische Route', line=dict(color='#0d6efd', width=4)))
                
                # Wenn der Algorithmus eine Kasse empfiehlt, extrahieren wir diese (vorletzter Knoten).
                kb = seq_s[-2] if len(seq_s) > 1 and seq_s[-2] in ['vK1', 'vK2', 'vK3'] else "vK1"
                est_time = round(dist_s / WALKING_SPEED)
                
                info_box = dbc.Card([
                    dbc.CardHeader([html.I(className="fa-solid fa-bullseye me-2"), "Operations Research: Statisches Modell"], className="bg-primary text-white fw-bold p-2"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([html.H5(f"{dist_s:.1f} m", className="text-primary mb-0"), html.Small("Laufstrecke", className="text-muted")], width=4, className="border-end text-center"),
                            dbc.Col([html.H5(f"{est_time}s", className="text-primary mb-0"), html.Small("Laufzeit", className="text-muted")], width=4, className="border-end text-center"),
                            dbc.Col([html.H5("2-OPT", className="text-primary mb-0"), html.Small("Optimierung", className="text-muted")], width=4, className="text-center"),
                        ]),
                        html.Hr(className="my-2"),
                        html.Div([html.I(className="fa-solid fa-cash-register me-2 text-muted"), f"Empfohlene Kasse: {kb}"], className="text-center text-dark fw-bold small"),
                    ])
                ], className="shadow-sm border-0")

                path_box = dbc.Accordion([
                    dbc.AccordionItem([html.Div(format_path_steps(seq_s, cart_data))], title="🧭 Turn-by-Turn Navigation starten")
                ], start_collapsed=False, flush=True)

            else:
                # =========================================================
                # 2. DYNAMISCHER KI-MODUS (Stau-Vermeidung & M/M/1/K)
                # =========================================================
                if not date_value: 
                    sim_dt = datetime.datetime.now()
                else: 
                    sim_dt = datetime.datetime.fromisoformat(date_value)
                    
                # Vorhersage vom ML-Modell abfragen (Wie voll ist es an jedem Regal?)
                loads_dict, total_agents = TrafficSimulationEngine.get_traffic_state(sim_dt)
                
                # Wir legen zwei separate Graphen an, um KI und Baseline gegeneinander antreten zu lassen.
                G_legal = G_base.copy() # Der Standard-Graph für den dummen Kunden
                G_smart = G_base.copy() # Der veränderte Graph für unseren smarten Agenten

                for u, v, d in G_base.edges(data=True):
                    orig_w = d.get('weight', 1.0)
                    # Auslastung für diese Kante (Gang) holen
                    real_load = loads_dict.get(f"{u}-{v}", loads_dict.get(f"{v}-{u}", 0.0))
                    
                    if orig_w < 999999:
                        # WICHTIG: Stufenweise Penalty-Funktion (BPR-Kurve)
                        # Verkehrsstau verhält sich nicht linear. 2 Leute behindern sich kaum, 
                        # aber ab einer kritischen Masse (z.B. 4 Personen) explodiert die Verzögerung.
                        # Deshalb wird das Kanten-Gewicht hier exponentiell (** 2.5) bestraft.
                        if real_load >= heat_thresh:
                            penalty = (real_load ** 2.5) * 200.0 
                        elif real_load >= 1.5:  
                            penalty = real_load * 5.0
                        else:
                            penalty = 0.0
                        
                        G_smart[u][v]['weight'] = orig_w + penalty
                    else:
                        G_smart[u][v]['weight'] = orig_w

                # Dynamisches Kapazitäts-Management der Kassen simulieren.
                # Ist die Hauptkasse zu voll, entscheidet das System, weitere Kassen zu öffnen.
                raw_k1 = loads_dict.get('vW1-vK1', loads_dict.get('vK1-vW1', 0))
                is_k3_open = (raw_k1 >= THRESHOLD_OPEN_K3)
                is_k2_open = (raw_k1 >= THRESHOLD_OPEN_K2)

                # Wenn eine Kasse geschlossen ist, sperren wir den Weg dorthin mathematisch ab (Gewicht auf Unendlich setzen).
                if not is_k2_open and G_legal.has_edge('vK2', 'vAusgang'): 
                    G_legal['vK2']['vAusgang']['weight'] = 999999
                    G_smart['vK2']['vAusgang']['weight'] = 999999
                if not is_k3_open and G_legal.has_edge('vK3', 'vAusgang'): 
                    G_legal['vK3']['vAusgang']['weight'] = 999999
                    G_smart['vK3']['vAusgang']['weight'] = 999999

                # A. Dumme Route berechnen (Baseline, läuft unbeeindruckt durch Stau)
                _, _, _, seq_b, _ = calculate_hybrid_route(G_legal, s_node, target_nodes, e_node)
                px_b, py_b = build_true_plot_path(seq_b, G_legal)

                # B. Smarte Route berechnen (KI weicht verstopften Gängen automatisch aus)
                _, _, _, seq_m, _ = calculate_hybrid_route(G_smart, s_node, target_nodes, e_node)
                px_m, py_m = build_true_plot_path(seq_m, G_smart)
                optimized_order = seq_m

                # Hilfsfunktion zur fairen Zeitberechnung (Realitäts-Simulation)
                # Beide Agenten laufen ihre berechneten Routen virtuell ab, und wir 
                # berechnen für jeden Schritt die tatsächlichen Zeitverluste.
                def calc_time(seq, G_sim):
                    w_sec, q_sec = 0.0, 0.0
                    for i in range(len(seq)-1):
                        u, v = seq[i], seq[i+1]
                        try:
                            # Der tatsächlich zurückgelegte Pfad in Metern
                            path = nx.dijkstra_path(G_sim, u, v, weight='weight')
                            for k in range(len(path)-1):
                                n1, n2 = path[k], path[k+1]
                                dist = G_base[n1][n2]['weight']
                                w_sec += (dist / WALKING_SPEED)
                                
                                # Zeitverlust durch Stau am jeweiligen Wegpunkt ermitteln
                                r_load = loads_dict.get(f"{n1}-{n2}", loads_dict.get(f"{n2}-{n1}", 0.0))
                                
                                # Kassenstau kostet extrem viel Zeit (Warteschlange)
                                if n2 in ['vK1', 'vK2', 'vK3'] and str(n1).startswith('vW'):
                                    q_sec += (r_load * 45.0) 
                                else:
                                    # Normaler Gang-Stau kostet nur Ausweich-Zeit
                                    w_sec += (r_load * TRAFFIC_DELAY)
                        except: pass
                    return w_sec, q_sec

                # Evaluierung der beiden Kontrahenten
                walk_b, queue_b = calc_time(seq_b, G_legal)
                walk_m, queue_m = calc_time(seq_m, G_smart)
                
                tb, tm = int(walk_b + queue_b), int(walk_m + queue_m)

                # ---------------------------------------------------------
                # SANITY CHECK: Fallback-Schutz für die KI
                # ---------------------------------------------------------
                # Ein absolut kritischer Aspekt bei KI-Systemen im Produktivbetrieb:
                # Was passiert, wenn die KI halluziniert und einen absurden, kilometerweiten 
                # Umweg vorschlägt, nur um 1 wartenden Kunden zu umgehen?
                # Hier verifizieren wir die Entscheidung. Ist die KI langsamer (oder exakt 
                # gleich schnell) wie die Baseline, überschreiben wir den KI-Vorschlag 
                # und nutzen die bekannte, direkte Route. Das schafft Vertrauen ins System.
                if tm > tb or (px_b == px_m and py_b == py_m):
                    walk_m, queue_m = walk_b, queue_b
                    tm = tb
                    seq_m = seq_b
                    px_m, py_m = px_b, py_b
                    optimized_order = seq_b
                
                # Gesamt-Zeitersparnis (in Sekunden)
                ts = tb - tm

                # Heatmap-Modus: Färbt die stark belasteten Gänge rot ein
                if view_mode in ['all', 'heat']:
                    for u, v in G_base.edges():
                        real_load = loads_dict.get(f"{u}-{v}", loads_dict.get(f"{v}-{u}", 0.0))
                        if real_load >= heat_thresh:
                            # Skaliert die Farbe sanft von Gelb zu intensivem Rot
                            intensity = min(1.0, (real_load - heat_thresh) / 15.0)
                            r = 255
                            g = int(200 * (1 - intensity)) 
                            b = 0
                            color = f'rgba({r},{g},{b},0.6)'
                            
                            p1, p2 = nodes_config[u]['pos'], nodes_config[v]['pos']
                            fig.add_trace(go.Scatter(
                                x=[p1[0], p2[0]], y=[p1[1], p2[1]], 
                                mode='lines', 
                                line=dict(color=color, width=14 + (intensity*6)), 
                                hoverinfo='text',
                                hovertext=f"Stau: {int(real_load)} Pers.",
                                showlegend=False
                            ))

                # Linien im Graphen einzeichnen
                if view_mode in ['all', 'base'] and px_b:
                    fig.add_trace(go.Scatter(x=px_b, y=py_b, mode='lines', name='Baseline Route', line=dict(color='#e74c3c', dash='dot', width=3)))
                if view_mode in ['all', 'smart'] and px_m:
                    fig.add_trace(go.Scatter(x=px_m, y=py_m, mode='lines', name='KI Route', line=dict(color='#2ecc71', width=5)))

                # Empfohlene Kassen identifizieren
                kb = seq_b[-2] if len(seq_b)>1 and seq_b[-2] in ['vK1', 'vK2', 'vK3'] else "N/A"
                km = seq_m[-2] if len(seq_m)>1 and seq_m[-2] in ['vK1', 'vK2', 'vK3'] else "N/A"

                if ts > 0:
                    rating_text = f"KI Rating: {ts} Sekunden gespart!"
                    rating_class = "alert alert-success py-3 mt-3 mb-0 text-center fw-bold fs-5 shadow-sm"
                else:
                    rating_text = "KI Rating: Routen sind identisch. Markt ist frei."
                    rating_class = "alert alert-secondary py-3 mt-3 mb-0 text-center fs-5 shadow-sm"

                # Aufbau des Statistik-Panels (Vergleichstabelle)
                info_box = html.Div([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fa-solid fa-users me-2"), 
                            f"Markt-Auslastung: ~{total_agents} Personen | 📊 M/M/1 Queuing"
                        ], className="fw-bold bg-dark text-white p-2"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Div("Dummer Kunde (Baseline)", className="text-danger fw-bold", style={"fontSize": "16px"}),
                                    html.H2(f"{tb} s", className="text-danger fw-bold mb-0"),
                                    html.Div(f"Laufweg: {int(walk_b)}s  +  Stau: {int(queue_b)}s", className="small text-muted mb-2"),
                                    dbc.Badge(f"Kasse {kb}", color="danger", className="p-2 fs-6")
                                ], width=6, className="border-end text-center p-3"),
                                
                                dbc.Col([
                                    html.Div("Smart KI Agent", className="text-success fw-bold", style={"fontSize": "16px"}),
                                    html.H2(f"{tm} s", className="text-success fw-bold mb-0"),
                                    html.Div(f"Laufweg: {int(walk_m)}s  +  Stau: {int(queue_m)}s", className="small text-muted mb-2"),
                                    dbc.Badge(f"Kasse {km}", color="success", className="p-2 fs-6")
                                ], width=6, className="text-center p-3")
                            ]),
                            html.Div(
                                [html.I(className="fa-solid fa-bolt me-2"), rating_text], 
                                className=rating_class
                            )
                        ], className="p-0")
                    ], className="shadow-lg border-0")
                ])
                
                path_box = dbc.Accordion([
                    dbc.AccordionItem([html.Div(format_path_steps(seq_m, cart_data, True))], title="🟢 KI Turn-by-Turn Navigation"),
                    dbc.AccordionItem([html.Div(format_path_steps(seq_b, cart_data, False))], title="🔴 Baseline Wegbeschreibung"),
                ], start_collapsed=True)

        except Exception as e:
            # Fehler robust abfangen und in der UI anzeigen, statt komplett abzustürzen
            info_box = dbc.Alert([html.H4("🚨 Fehler bei der Routenberechnung"), html.Hr(), html.P(str(e))], color="danger", className="shadow-lg")
            traceback.print_exc()

    # ---------------------------------------------------------
    # KNOTEN (REGALE) AUF DIE KARTE ZEICHNEN
    # ---------------------------------------------------------
    for nid, d in nodes_config.items():
        category_name = node_to_category.get(nid, "")
        category_label = f"<b>{category_name}</b><br>" if category_name else ""
        items_text = inv_manager.get_items_text(nid)
        full_hover = f"{category_label}{items_text}"
        
        # Gehört der Knoten zum Einkaufszettel des Users? Dann markant hervorheben.
        if nid in target_nodes or nid in [s_node, e_node]:
            col, lbl = COLOR_HIGHLIGHT, "!"
            if nid == s_node: col, lbl = "#2ecc71", "S"
            elif nid == e_node: col, lbl = "#e67e22", "Z"
            # Fügt eine Nummerierung (1, 2, 3...) in den Kreis ein, basierend auf der optimalen Route
            elif optimized_order and nid in optimized_order: 
                lbl = str(optimized_order.index(nid))
                
            fig.add_trace(go.Scatter(x=[d['pos'][0]], y=[d['pos'][1]], mode='markers+text', text=lbl, marker=dict(size=16, color=col, line=dict(width=2, color="black")), textfont=dict(color="white"), hovertext=full_hover, hoverinfo='text', showlegend=False))
        
        # Normale, nicht besuchte Knotenpunkte (werden dezent im Hintergrund gezeichnet)
        elif nid != 'vInCorner':
            fig.add_trace(go.Scatter(x=[d['pos'][0]], y=[d['pos'][1]], mode='markers', marker=dict(color=d['col'], size=6, opacity=0.3), hovertext=full_hover, hoverinfo='text', showlegend=False))

    # Optisches Feintuning des Plotly-Graphen (Rahmen und Achsen ausblenden)
    fig.update_layout(plot_bgcolor='white', xaxis=dict(visible=False, fixedrange=True, range=[-5, 40]), yaxis=dict(visible=False, fixedrange=True, range=[-3, 17]), margin=dict(l=0, r=0, t=5, b=0), showlegend=True, legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.9)", bordercolor="#e0e0e0", borderwidth=1))
    
    return html.Div(list_items), fig, info_box, path_box

# =============================================================================
# 7. ADMIN CALLBACKS (Datenbank-Management / CRUD)
# =============================================================================

@app.callback(
    [Output("admin-add-node", "value"), Output("admin-needs-review", "value"), Output("ai-suggest-result", "children"), Output("admin-add-cat", "value")], 
    Input("btn-ai-suggest", "n_clicks"), State("admin-add-name", "value"), prevent_initial_call=True
)
def ai_suggest_slot(n_clicks, product_name):
    """
    Fragt das NLP-Modell (TF-IDF + Log. Regression), wo ein neu angelegtes Produkt
    einsortiert werden soll. Setzt zudem ein 'Human-in-the-Loop' Flag: 
    Falls die KI sich zu unsicher ist (Konfidenz unter 80%), muss ein Mensch 
    die Entscheidung überprüfen.
    """
    if not product_name: 
        return no_update, no_update, dbc.Alert("Name eingeben.", color="warning", className="py-1 small"), no_update
    if not ml_predictor or not ml_predictor.is_loaded: 
        return no_update, no_update, dbc.Alert("ML Modell inaktiv.", color="danger", className="py-1 small"), no_update
    
    try:
        cat, node, conf = ml_predictor.predict(product_name) 
        alert = dbc.Alert(f"KI: {cat} ({conf:.1%})", color="success" if conf>=0.8 else "warning", className="py-2 small fw-bold")
        # Review-Flag ist True, wenn KI-Sicherheit unter 80% (0.8) liegt.
        return node, conf<0.8, alert, cat
    except Exception as e:
        alert = dbc.Alert(f"Fehler bei Vorhersage: {str(e)}", color="danger", className="py-2 small fw-bold")
        return no_update, no_update, alert, no_update

@app.callback(
    [Output("modal-admin-login", "is_open"), Output("admin-panel-collapse", "is_open"), Output("admin-login-error", "children")],
    [Input("btn-admin-modal-open", "n_clicks"), Input("btn-admin-login", "n_clicks")],
    [State("modal-admin-login", "is_open"), State("admin-password-input", "value")]
)
def toggle_admin(n_open, n_login, is_open, pw):
    """
    Authentifizierung für das Admin-Panel via MD5-Hashing.
    Hinweis: MD5 ist kryptografisch veraltet und dient hier lediglich der Demonstration.
    In einem produktiven System müsste zwingend SHA-256 mit Salt verwendet werden, 
    um Rainbow-Table-Angriffe auszuschließen.
    """
    ctx = dash.callback_context
    tid = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # Popup öffnen/schließen
    if tid == "btn-admin-modal-open": return not is_open, False, ""
    
    # Login prüfen
    if tid == "btn-admin-login":
        # Vergleicht den Hash, niemals das Klartext-Passwort!
        system_pw = os.environ.get("ADMIN_PW_HASH", "e19e075037d046fccbb4a17961d9a0cd")
        if hashlib.md5(pw.encode('utf-8')).hexdigest() == system_pw:
            return False, True, "" 
        else:
            return True, False, "Falsch!"
    return is_open, False, ""

@app.callback(
    [Output("admin-add-feedback", "children"), Output("admin-delete-feedback", "children"), Output("admin-delete-select", "options"), Output("floor-plan", "figure", allow_duplicate=True)],
    [Input("btn-admin-save", "n_clicks"), Input("btn-admin-delete", "n_clicks")],
    [State("admin-add-node", "value"), State("admin-add-name", "value"), State("admin-add-brand", "value"), State("admin-add-cat", "value"), State("admin-needs-review", "value"), State("admin-delete-select", "value")],
    prevent_initial_call=True
)
def admin_manage(n_save, n_del, a_node, a_name, a_brand, a_cat, n_rev, del_val):
    """
    CRUD (Create/Delete) Operationen für die Supermarkt-Produktdatenbank.
    Nutzt zwingend Threading-Locks (with inv_manager._lock), um Race-Conditions 
    (Abstürze) zu vermeiden, falls ein Nutzer zeitgleich im Dashboard nach 
    einem Produkt sucht, während der Admin speichert.
    """
    ctx = dash.callback_context
    tid = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    m_add, m_del = "" , ""
    
    # PRODUKT HINZUFÜGEN
    if tid == "btn-admin-save":
        if a_node and a_name and a_brand:
            try:
                with inv_manager._lock:
                    if a_node not in inv_manager.stock: 
                        inv_manager.stock[a_node] = []
                    inv_manager.stock[a_node].append({'name': a_name, 'brand': a_brand, 'category': a_cat or "Sonstiges"})
                    
                    # Speichern und Index sofort aktualisieren, damit es für die Live-Suche verfügbar ist
                    inv_manager.save_to_json()
                    inv_manager._build_search_index()
                m_add = dbc.Alert("Gespeichert!", color="success", className="py-1 px-2")
            except Exception as e: 
                m_add = dbc.Alert(f"Fehler: {e}", color="danger", className="py-1 px-2")
        else: 
            m_add = dbc.Alert("Alle Felder füllen.", color="warning", className="py-1 px-2")

    # PRODUKT LÖSCHEN
    if tid == "btn-admin-delete" and del_val:
        try:
            with inv_manager._lock:
                node, name = del_val.split("|", 1) 
                # List-Comprehension filtert das zu löschende Produkt sauber heraus
                inv_manager.stock[node] = [i for i in inv_manager.stock.get(node, []) if i['name'] != name]
                inv_manager.save_to_json()
                inv_manager._build_search_index()
            m_del = dbc.Alert("Gelöscht.", color="success", className="py-1 px-2")
        except Exception as e: 
            m_del = dbc.Alert(f"Fehler: {e}", color="danger", className="py-1 px-2")

    # Aktualisiert die Dropdown-Liste für das Löschen-Fenster dynamisch
    d_opts = [{'label': f"{i['name']} ({i['brand']}) - [{n}]", 'value': f"{n}|{i['name']}"} for n, items in inv_manager.stock.items() for i in items]
    return m_add, m_del, d_opts, dash.no_update

if __name__ == '__main__':
    # Startet den internen Entwicklungsserver
    app.run(debug=True, port=8050)