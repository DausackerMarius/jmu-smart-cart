Frontend-Architektur: Reaktive UI, Edge-Computing & Hardware-Integration
========================================================================

Die architektonische Konzeption dieses Systems erfordert eine strikte Trennung der Verantwortlichkeiten (Separation of Concerns). Während das zentrale Management- und Data-Science-Backend (wie in den vorherigen Kapiteln beschrieben) auf dem Framework *Plotly Dash* operiert, verlangt die Hardware am Einkaufswagen eine völlig andere Herangehensweise. Dash generiert sein eigenes Frontend und ist für native Hardware-Zugriffe (Bluetooth, Laser-Scanner) ungeeignet. 

Die Client-Applikation des JMU Smart Carts operiert stattdessen als entkoppelte **Progressive Web App (PWA)** – implementiert in nativem React 18 und striktem TypeScript, gewrappt im Capacitor-Framework für direkten Hardware-Zugriff. Sie kommuniziert über asynchrone APIs mit dem zentralen Dash-Backend.

Dieses Tablet am Einkaufswagen fungiert als dedizierter Edge-Computing-Knoten und unterliegt extremen Hardware-Restriktionen: Limitierter LPDDR-Arbeitsspeicher, ein leistungsschwacher ARM-SoC (System-on-a-Chip) und das permanente Risiko von Thermal Throttling (Hitze-Drosselung der CPU) bei kontinuierlicher Grafikausgabe. 

Zudem ist die zugrundeliegende JavaScript V8-Engine des Browsers inhärent Single-Threaded. Die Architektur muss zwingend vier hochfrequente I/O- und CPU-Tasks parallel orchestrieren, ohne den V8-Call-Stack jemals zu blockieren:

1. Kontinuierliche Sensor-Fusion zur Indoor-Lokalisierung via Bluetooth (BLE).
2. Interrupt-gesteuerte Verarbeitung von Barcode-Laserscans.
3. Das latenzfreie, hardware-adaptive Rendering komplexer Supermarkt-Topologien.
4. Die asynchrone Verarbeitung hochfrequenter WebSocket-Datenströme (Echtzeit-Stauupdates).

Ein naives, synchrones Architektur-Design würde hier unweigerlich den Event-Loop blockieren, was zu Layout Thrashing und Frame-Drops (dem Einfrieren der Benutzeroberfläche). Dieses Kapitel dekonstruiert die Architekturpatterns, die angewandt werden, um das Frontend als hochperformantes, resilientes Echtzeitsystem zu etablieren.

1. State Hydration, BFF & Cache Invalidation
--------------------------------------------
Beim Systemstart (Bootstrapping) ist der In-Memory-Zustand des Tablets volatil und leer. Um das gefürchtete N+1-Query-Problem (Dutzende isolierte REST-Calls, die das Netzwerk überlasten) zu vermeiden, nutzt das System ein Backend-for-Frontend (BFF) Pattern. 

Das State-Management wird über "Zustand" (eine atomare, Flux-basierte State-Engine) realisiert. Diese Engine umgeht den React-Context-Tree vollständig und verhindert somit globale $\mathcal{O}(N)$ Re-Renders der gesamten Applikation. Um das System vor veralteten Offline-Daten (Stale Data) zu schützen, implementiert das Bootstrapping zudem eine kryptografische Cache Invalidation Strategie via ETag/Hash-Abgleich.

.. code-block:: typescript

    import { useEffect } from 'react';
    import { useCartStore } from './store/cartStore';
    import { apiGateway } from './network/apiGateway';
    import { trafficSocketManager } from './network/websockets';
    import { EdgeCache } from './storage/indexedDB';

    export function useSystemBootstrap() {
        useEffect(() => {
            async function hydrateSystemState() {
                useCartStore.setState({ systemStatus: 'BOOTING' });
                try {
                    // 1. Hole den kleinen Versions-Hash vom Server (Netzwerk-Latenz < 20ms)
                    const currentServerVersion = await apiGateway.fetchTopologyVersion();
                    const cachedData = await EdgeCache.loadTopology();
                    
                    let activeTopology;
                    
                    // 2. Cache Invalidation: Nur neu laden, wenn der Supermarkt umgebaut wurde
                    if (!cachedData || cachedData.versionHash !== currentServerVersion) {
                        activeTopology = await apiGateway.fetchStaticMapTopology();
                        await EdgeCache.saveTopology(activeTopology); // Lokales Update
                    } else {
                        activeTopology = cachedData.topology; // Zero-Latency Boot aus dem Cache
                    }
                    
                    const sessionPayload = await apiGateway.restoreSession();
                    
                    // 3. Atomare Mutation des Global States in O(1)
                    useCartStore.setState({ 
                        mapData: activeTopology, 
                        cart: sessionPayload.items,
                        systemStatus: 'READY' 
                    });
                    
                    trafficSocketManager.connectAndSubscribe();
                } catch (error) {
                    // Fallback-Strategie bei komplettem Netzwerkausfall
                    EdgeCache.fallbackToLocalRouting();
                }
            }
            hydrateSystemState();
        }, []); 
    }

2. Hardware Abstraction Layer (HAL) & Kalman-Filter
---------------------------------------------------
Klassisches GPS durchdringt die Stahlbetonarchitektur eines Supermarkts nicht. Das System nutzt stattdessen Bluetooth Low Energy (BLE) Beacons. JavaScript im Web-Browser hat aus strikten Sandbox-Sicherheitsgründen jedoch keinen Zugriff auf das passive Scannen von BLE-Signalen. 

Die React-Applikation wird daher über das Capacitor-Framework in einen nativen Hardware-Container gewrappt. Ein Hardware Abstraction Layer (HAL) agiert als Foreign Function Interface (FFI) Bridge und streamt die rohen Antennen-Daten asynchron in die V8-Engine. Das rohe Received Signal Strength Indicator (RSSI) Signal unterliegt im Supermarkt jedoch extremem stochastischem Rauschen durch Multipath-Fading (Signal-Reflexionen an Regalen). Die Architektur implementiert daher einen eindimensionalen Kalman-Filter zur iterativen Sensor-Fusion. 

Mathematische Fundierung: Die Signalstärke wird zunächst in eine physikalische Distanz ($d$) übersetzt, wobei der Path Loss Exponent ($n$) ein kalibrierter Umgebungsfaktor für die Signalabsorption ist (typischerweise 2.5 für Innenräume):

$$d=10^{\frac{P_{Tx}-RSSI}{10\cdot n}}$$

Der Kalman-Filter errechnet daraufhin den Kalman-Gain ($K_t$), der dynamisch abwägt, ob der verrauschten Messung ($R$) oder der bisherigen Prädiktion ($P$) mehr vertraut werden soll:

$$K_t=\frac{P_{vorher}}{P_{vorher}+R}$$

.. code-block:: typescript

    import { useEffect, useRef } from 'react';
    import { KalmanFilter } from './math/KalmanFilter';
    import { calculateTrilateration } from './math/geometry';
    import { NativeBLEBridge } from '@capacitor/bluetooth-le';

    interface BeaconSignal { rssi: number; txPower: number; beaconId: string; }

    export function useIndoorPositioning() {
        // R: Messrauschen des Sensors, Q: Prozessrauschen (Bewegungsdynamik des Kunden)
        const filter = useRef(new KalmanFilter({ R: 0.008, Q: 2.5 })); 

        useEffect(() => {
            // Callback wird asynchron aus dem nativen C++/Java-Kernel getriggert
            const processBeaconSignal = ({ rssi, txPower }: BeaconSignal) => {
                // 1. Logarithmische Distanzapproximation
                const rawDistance = Math.pow(10, (txPower - rssi) / (10 * 2.5));
                
                // 2. Stochastische Glättung (Update-Step der Kalman-Matrix)
                const smoothedDistance = filter.current.filter(rawDistance);
                
                // 3. Geometrische Lokalisierung (Schnittpunkt der Radien)
                const newPosition = calculateTrilateration(smoothedDistance, knownBeacons);
                useCartStore.setState({ currentPosition: newPosition });
            };

            NativeBLEBridge.startScanning(processBeaconSignal);
            return () => NativeBLEBridge.stopScanning();
        }, []);
    }

3. Hardware-Interrupts & HID Injection Defense
----------------------------------------------
Der physische Barcode-Laserscanner am Wagen operiert als sogenanntes Keyboard Wedge. Er emuliert ein Human Interface Device (HID) und injiziert die gescannten Ziffern als hochfrequente Tastatur-Events in den Browser.

Dies öffnet einen gefährlichen Angriffsvektor für HID Injection Attacks (z.B. absichtlich kompromittierte Barcodes, die bösartigen Schadcode tippen). Das Frontend implementiert als Gegenmaßnahme eine strikte Debouncing- und Sanitization-Pipeline. Da ein physischer Laser-Scanner die Ziffern in Abständen von unter 30 Millisekunden (< 30ms) emuliert, blockiert eine Timing-Heuristik manuelle (und damit langsamere) Touchscreen-Tastatureingaben rigoros.

.. code-block:: typescript

    export function useHardwareScannerIntegration() {
        const buffer = useRef<string>('');
        const lastKeyTime = useRef<number>(Date.now());

        useEffect(() => {
            const handleHardwareScan = (e: KeyboardEvent) => {
                const currentTime = Date.now();
                
                // Timing-Schutz: Blockiert menschliche Tastatureingaben (> 50ms Latenz).
                if (currentTime - lastKeyTime.current > 50) buffer.current = ''; 
                lastKeyTime.current = currentTime;

                // Auslöser: Der Barcodescanner sendet ein abschließendes 'Enter'
                if (e.key === 'Enter' && buffer.current.length >= 8) {
                    e.preventDefault();
                    
                    // Security Sanitization: Erlaubt zwingend nur numerische EAN/GTIN-Codes
                    const sanitizedCode = buffer.current.replace(/[^0-9]/g, '').slice(0, 14);
                    buffer.current = ''; 
                    
                    if (sanitizedCode.length >= 8) {
                        useCartStore.getState().processScannedItem(sanitizedCode);
                    }
                    return;
                }
                
                if (e.key.length === 1) buffer.current += e.key;
            };

            window.addEventListener('keydown', handleHardwareScan);
            return () => window.removeEventListener('keydown', handleHardwareScan);
        }, []);
    }

4. Spatial Rendering: Die Affine Koordinaten-Transformation
-----------------------------------------------------------
Das Backend rechnet (wie im Topologie-Kapitel definiert) in einem lokalen, kartesischen System in absoluten "Metern". Das Frontend-Tablet zeichnet jedoch auf einem physischen Bildschirm in "Pixeln". 

Der Edge-Client muss daher eine ständige affine Transformation (Skalierung und Translation) durchführen. Da Tablets im Supermarkt unterschiedliche Bildschirmauflösungen besitzen können, darf diese Skalierung nicht im Code hartcodiert sein. Das Frontend berechnet den Skalierungsfaktor dynamisch beim Boot-Vorgang basierend auf den exakten Ausmaßen des DOM-Containers.

Mathematisch erfolgt dies durch eine **Affine Transformation** über eine homogene Matrix (Translation vom Backend-Meter-Raum in den Frontend-Pixel-Raum):

$$\begin{pmatrix}X_{pixel}\\Y_{pixel}\\1\end{pmatrix}=\begin{pmatrix}S_x&0&T_x\\0&S_y&T_y\\0&0&1\end{pmatrix}\begin{pmatrix}X_{meter}\\Y_{meter}\\1\end{pmatrix}$$

.. code-block:: typescript

    class MapRenderer {
        constructor(canvasElement, backendTopology) {
            this.canvas = canvasElement;
            this.ctx = this.canvas.getContext('2d');
            this.topology = backendTopology;
            
            // Stack-Safe Max-Berechnung: Verhindert "Maximum call stack size exceeded" bei großen Graphen
            this.maxX = Object.values(this.topology.nodes).reduce((max, n) => Math.max(max, n.coordinates[0]), 0);
            this.maxY = Object.values(this.topology.nodes).reduce((max, n) => Math.max(max, n.coordinates[1]), 0);
            
            this.calculateScale();
            window.addEventListener('resize', () => this.calculateScale());
        }

        calculateScale() {
            // Ein dynamischer Skalierungsfaktor (Pixel pro Meter) 
            this.scaleX = this.canvas.width / (this.maxX * 1.1); // 10% Padding
            this.scaleY = this.canvas.height / (this.maxY * 1.1);
            
            // Isotropische Skalierung: Nutzt den kleineren Faktor für beide Achsen,
            // damit die Supermarkt-Geometrie auf dem Display nicht verzerrt wird.
            this.scale = Math.min(this.scaleX, this.scaleY);
        }

        transformCoordinates(x_meter, y_meter) {
            // Mathematische Translation vom Backend-Meter-Raum in den Frontend-Pixel-Raum
            const paddingOffset = 20; 
            const x_pixel = (x_meter * this.scale) + paddingOffset;
            const y_pixel = (y_meter * this.scale) + paddingOffset;
            return { x: x_pixel, y: y_pixel };
        }
    }

5. Echtzeit-Stauheatmap (Dynamic Traffic Rendering)
---------------------------------------------------
Ein Kernfeature der Architektur ist es, dem Kunden die von der KI prädizierte Stausituation visuell aufzubereiten. Das Tablet empfängt asynchron über WebSockets sekündliche JSON-Pakete, die Verkehrsstrafen für Graphen-Kanten enthalten. Das Frontend übersetzt diese abstrakten Metriken in eine kognitive Heatmap.

Ein klassischer Performance-Bug (der "rAF-Tsunami") entsteht, wenn das Netzwerk kurzzeitig laggt und der Server daraufhin plötzlich 10 Stau-Updates in derselben Millisekunde pusht. Ein naiver Event-Listener würde 10 redundante Render-Aufträge für denselben Frame in die Warteschlange packen, was zum Einfrieren des Tablets führt. Die Architektur implementiert daher ein strenges **Render-Lock (`isRenderPending`)**. Neue Daten-Updates aktualisieren lediglich den internen Zustand; überschüssige Render-Aufträge werden rigoros blockiert, solange der vorherige Frame noch nicht auf den Canvas gezeichnet wurde.

Die Browser-API ``requestAnimationFrame()`` synchronisiert die Zeichenaufrufe exakt mit der Bildwiederholrate des Tablets (V-Sync). Die Canvas-Eigenschaft ``globalCompositeOperation`` erlaubt das fließende, transluzente Überblenden der Stau-Gradients über den statischen Gebäudeplan.

.. code-block:: typescript

    class TrafficHeatmapLayer {
        constructor(renderer) {
            this.renderer = renderer;
            this.currentTrafficData = {}; 
            this.isRenderPending = false; // Render-Lock gegen rAF-Tsunamis
        }

        updateTraffic(wsPayload) {
            // Daten-Update erfolgt synchron, Rendering wird entkoppelt
            this.currentTrafficData = wsPayload.edge_penalties;
            
            // Senior-Pattern: Verhindert Frame-Drops durch gebündelte Netzwerklags
            if (!this.isRenderPending) {
                this.isRenderPending = true;
                requestAnimationFrame(() => this.drawHeatmap());
            }
        }

        getColorForPenalty(penaltyValue) {
            // Interpoliert den XGBoost-Strafwert in eine visuelle Signalfarbe
            if (penaltyValue < 2.0) return 'rgba(46, 204, 113, 0.6)';  // Grün (Frei)
            if (penaltyValue < 6.0) return 'rgba(241, 196, 15, 0.7)';  // Gelb (Warnung)
            return 'rgba(231, 76, 60, 0.9)';                           // Rot (Stau)
        }

        drawHeatmap() {
            this.isRenderPending = false; // Render-Lock sofort für den nächsten Cycle freigeben
            
            const ctx = this.renderer.ctx;
            ctx.clearRect(0, 0, this.renderer.canvas.width, this.renderer.canvas.height);
            
            // Die Heatmap wird als transparenter Layer über die Basis-Karte gezeichnet
            ctx.globalCompositeOperation = 'source-over';
            ctx.lineCap = 'round';
            ctx.lineWidth = 14; 

            for (const [edge_id, penalty] of Object.entries(this.currentTrafficData)) {
                const [nodeU, nodeV] = edge_id.split('_');
                const coordsU = this.renderer.topology.nodes[nodeU].coordinates;
                const coordsV = this.renderer.topology.nodes[nodeV].coordinates;

                const pU = this.renderer.transformCoordinates(coordsU[0], coordsU[1]);
                const pV = this.renderer.transformCoordinates(coordsV[0], coordsV[1]);

                ctx.beginPath();
                ctx.moveTo(pU.x, pU.y);
                ctx.lineTo(pV.x, pV.y);
                ctx.strokeStyle = this.getColorForPenalty(penalty);
                ctx.stroke();
            }
        }
    }

6. Adaptive Render Engine: Web Workers & Thermal Protection
-----------------------------------------------------------
Das permanente Rendering einer Route mit Tausenden topologischen Vektoren würde die V8-Engine unter Last überfordern und den Akku massiv belasten. Die Architektur delegiert die schwere Grafik-Berechnung daher an einen isolierten Hintergrund-Thread (Web Worker) via OffscreenCanvas API und Zero-Copy Memory Transfer (Übergabe des Canvas-Pointers in $\mathcal{O}(1)$).

Architektonischer Schutz vor Thermal Throttling: Wenn das Edge-Device überhitzt, drosselt das Betriebssystem die CPU-Taktrate. Ein starrer 60-FPS-Loop würde dann zu einem unkontrollierten Memory-Overflow der Worker-Queue führen. Die Applikation implementiert daher Adaptive Framerate Control. Der Worker misst die exakte Render-Latenz, meldet die maximale Hardware-Kapazität an den Main-Thread zurück, und der Main-Thread drosselt seinen Rendering-Loop dynamisch entsprechend.

.. code-block:: typescript

    // --- MapWorker.ts (Hintergrund-Thread) ---
    self.onmessage = (evt: MessageEvent) => {
        const { routeCoords, scale, offsetX, offsetY, offscreenCanvas } = evt.data;
        const startTime = performance.now();
        
        const ctx = offscreenCanvas.getContext('2d', { alpha: false }); 
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, offscreenCanvas.width, offscreenCanvas.height);
        
        ctx.beginPath();
        ctx.strokeStyle = '#0066cc'; 
        ctx.lineWidth = 4;
        
        routeCoords.forEach((p, index) => {
            const pixelX = (p.x * scale) + offsetX;
            const pixelY = (p.y * scale) + offsetY;
            if (index === 0) ctx.moveTo(pixelX, pixelY);
            else ctx.lineTo(pixelX, pixelY);
        });
        ctx.stroke(); 
        
        const renderDuration = performance.now() - startTime;
        
        // Thermal Protection: Berechnung der maximalen Hardware-Kapazität
        let optimalFPS = 60;
        if (renderDuration > 16.6) optimalFPS = 30; // 30 FPS bei leichter Drosselung
        if (renderDuration > 33.3) optimalFPS = 15; // 15 FPS bei aktivem Thermal Throttling
        
        const bitmap = offscreenCanvas.transferToImageBitmap();
        self.postMessage({ bitmap, optimalFPS }, [bitmap]);
    };

    // --- MainThread.tsx (React Komponente) ---
    export function useAdaptiveRenderLoop(worker: Worker) {
        const targetFPS = useRef<number>(60);
        const lastFrameTime = useRef<number>(0);

        useEffect(() => {
            // Listener für das Feedback des Workers
            worker.onmessage = (e) => { targetFPS.current = e.data.optimalFPS; };

            const renderLoop = (timestamp: number) => {
                const frameInterval = 1000 / targetFPS.current;
                
                // Throttling-Sync: Main-Thread feuert nur, wenn der Worker Kapazitäten hat
                if (timestamp - lastFrameTime.current >= frameInterval) {
                    const state = useCartStore.getState();
                    worker.postMessage({ routeCoords: state.route, /* ... */ });
                    lastFrameTime.current = timestamp;
                }
                requestAnimationFrame(renderLoop);
            };
            requestAnimationFrame(renderLoop);
        }, [worker]);
    }

7. Asynchrone Topologie-Mutation: Das Pre-Halftime Protokoll
------------------------------------------------------------
Wie in der Backend-Architektur formalisiert, darf der Zielknoten (die Kassenzone) dem TSP-Routing-Solver nicht erst am Ende des Einkaufs übergeben werden, um Deadlocks zu vermeiden. Das Frontend implementiert hierfür das Pre-Halftime Prädiktions-Protokoll. Bei exakt 40 Prozent Fortschritt triggert das Client-System asynchron die Stochastik-Engine des Backends.

.. code-block:: typescript

    import { useEffect } from 'react';
    import { useCartStore } from './store/cartStore';
    import { apiGateway } from './network/apiGateway';
    import { shallow } from 'zustand/shallow';

    export function useHalftimeCheckoutPrediction() {
        useEffect(() => {
            // O(1) Selectors & Shallow Equality blockieren unnötige React-Re-Renders
            const unsubscribe = useCartStore.subscribe(
                (state) => ({ 
                    collected: state.cart.length, 
                    total: state.shoppingList.length,
                    isPredicted: state.checkoutPredicted
                }),
                (current) => {
                    if (current.total === 0 || current.isPredicted) return;
                    
                    // Architektur-Gesetz: Pre-Halftime Trigger bei 40%
                    const progressRatio = current.collected / current.total;
                    if (progressRatio >= 0.4) {
                        useCartStore.setState({ checkoutPredicted: true }); 
                        
                        apiGateway.triggerStochasticCheckoutPrediction().then(optimalExitNode => {
                            // Asynchrone Mutation der Graphen-Topologie
                            useCartStore.getState().appendExitNodeToRoute(optimalExitNode);
                        });
                    }
                },
                { equalityFn: shallow }
            );
            return unsubscribe;
        }, []);
    }

8. Das Hidden Admin-Dashboard & MLOps-Telemetrie
------------------------------------------------
Für den produktiven Betrieb und die wissenschaftliche Auswertung benötigt das System ein geschütztes Backend-Kontrollzentrum direkt auf dem Tablet. Die Architektur implementiert ein Hidden Admin-Dashboard. Dieses wird nicht über einen Button aktiviert, sondern über ein unsichtbares Zonen-Tapping (Security via Obscurity). 

Dieses Dashboard ist die mobile Leitstelle des MLOps-Lifecycles. Es fragt System-Telemetrie ab und ermöglicht den manuellen Eingriff in die serverseitige Machine-Learning-Pipeline. Da es sich um eine React-Anwendung handelt, wird die komplexe Logik in einem isolierten Custom Hook gekapselt, der den Lifecycle-State sauber orchestriert.

*Die Security-Architektur (Just-in-Time Auth):* Ein öffentliches Kiosk-Terminal (wie der Einkaufswagen) darf niemals persistente JWT-Tokens im ``localStorage`` oder ``sessionStorage`` speichern, da dies den Weg für Cross-Site-Scripting (XSS) und weitreichende Token-Diebstähle ebnen würde. Die Architektur erzwingt daher eine Just-in-Time-Autorisierung. Die Filialleitung authentifiziert sich über eine dedizierte React-Modal-Komponente (nicht-blockierend). Der zurückgegebene Token wird streng in einer flüchtigen JavaScript-Variable (In-Memory) gehalten. Auf persistente Speicherung wird bewusst verzichtet.

.. code-block:: typescript

    import { useEffect, useState } from 'react';

    export function useAdminDashboard() {
        const [tapSequence, setTapSequence] = useState<number[]>([]);
        const [isAdminActive, setIsAdminActive] = useState(false);
        const [telemetry, setTelemetry] = useState<any>(null);

        useEffect(() => {
            /*
             * Security via Obscurity: Der Admin-Modus wird aktiviert, wenn 
             * exakt dreimal schnell in die obere linke Ecke getippt wird.
             */
            const handleSecretTap = (e: MouseEvent) => {
                if (e.offsetX < 50 && e.offsetY < 50) {
                    setTapSequence(prev => {
                        const newSeq = [...prev, Date.now()];
                        if (newSeq.length === 3) {
                            const timeDiff = newSeq[2] - newSeq[0];
                            if (timeDiff < 1500) setIsAdminActive(!isAdminActive);
                            return []; // Reset nach erfolgreichem/verfehltem Versuch
                        }
                        return newSeq;
                    });
                } else {
                    setTapSequence([]); // Sofortiger Reset bei falschem Klick
                }
            };
            
            const canvas = document.getElementById('mapCanvas');
            if (canvas) canvas.addEventListener('click', handleSecretTap as any);
            return () => {
                if (canvas) canvas.removeEventListener('click', handleSecretTap as any);
            };
        }, [isAdminActive]);

        useEffect(() => {
            if (isAdminActive) {
                // API-Aufruf an das Backend zur Abfrage der System-Gesundheit
                fetch('/api/v1/admin/telemetry')
                    .then(res => res.json())
                    .then(data => setTelemetry(data));
            }
        }, [isAdminActive]);

        // Wird aus der asynchronen React-Modal-Komponente aufgerufen
        const forceRetraining = async (adminPin: string) => {
            if (!adminPin) return;

            try {
                // Auth-Flow zur Generierung des flüchtigen JWT
                const authResponse = await fetch('/api/v1/auth/jit_admin', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ pin: adminPin })
                });
                
                if (!authResponse.ok) throw new Error("Unauthorized");
                const { transient_token } = await authResponse.json();
                
                await fetch('/api/v1/admin/force_ml_retrain', { 
                    method: 'POST',
                    headers: { 'Authorization': `Bearer ${transient_token}` }
                });
                
                // Erfolg per State-Trigger an die UI melden
                return { success: true, message: 'ML-Pipeline gestartet.' };
                
            } catch (error) {
                return { success: false, message: 'Autorisierung fehlgeschlagen.' };
            } finally {
                // Security-Best-Practice: Lokale Referenz auflösen.
                // Anmerkung: Strings sind in JS immutable, ein echtes Memory-Wiping 
                // ist systemseitig nicht möglich, aber die GC-Bindung wird gelöst.
                adminPin = '';
            }
        };

        return { isAdminActive, telemetry, forceRetraining };
    }

9. Resilienz im Netzwerk: Exponential Backoff mit Full Jitter
-------------------------------------------------------------
Wenn das WLAN im Supermarkt für 10 Sekunden ausfällt, trennen sich hunderte Tablets gleichzeitig vom WebSocket-Server. Kommt das WLAN zurück, würden alle Tablets in derselben Millisekunde versuchen, sich neu zu verbinden. Dieser Tsunami an synchronen Requests (das sogenannte *Thundering Herd Problem*) würde den Backend-Server sofort durch Überlastung in die Knie zwingen. Das Phänomen gleicht dem Verkaufsstart von beliebten Konzertkarten, bei dem Tausende Nutzer in derselben Millisekunde die Website aktualisieren und die Server zum Absturz bringen.

Das System verhindert dies durch den architektonischen Industriestandard des Exponential Backoff mit Full Jitter. Schlägt ein Verbindungsversuch fehl, wartet das Tablet nicht stur eine feste Sekunde. Die Wartezeit verdoppelt sich bei jedem Versuch exponentiell (1, 2, 4, 8 Sekunden). 

Der mathematische Jitter-Faktor: Würden alle Wagen exakt 4 Sekunden warten, käme die Überlastung nur verzögert an. Der Code addiert daher eine pseudozufällige Zeitspanne (den Jitter). Das bewirkt, dass sich die 100 Einkaufswagen nicht gleichzeitig, sondern leicht asynchron, wie bei einem Reißverschlussverfahren, sanft wieder mit dem Server verbinden. Ein harter unterer Schwellenwert (100 ms) schützt das System dabei vor fatalen 0-Millisekunden-Berechnungen durch den Zufallsgenerator.

.. code-block:: typescript

    class ResilientSocketManager {
        private retryCount = 0;
        private readonly BASE_DELAY_MS = 1000;
        private readonly MAX_DELAY_MS = 30000;

        public connectAndSubscribe() {
            const ws = new WebSocket('wss://api.smartcart.jmu/traffic');

            ws.onclose = () => {
                // Exponential Backoff: Skaliert Basis-Verzögerung mathematisch mit 2^n.
                // Die Nutzung von Math.pow schützt das System vor Syntax-Parsing-Fehlern.
                const exponentialDelay = Math.min(
                    this.MAX_DELAY_MS, 
                    this.BASE_DELAY_MS * Math.pow(2, this.retryCount)
                );
                
                // Full Jitter (Desynchronisiert die Reconnects der Flotte physikalisch)
                // Math.max(100) verhindert einen DDoS-Angriff auf sich selbst durch t=0
                const jitteredWaitTime = Math.max(100, Math.random() * exponentialDelay);
                
                this.retryCount++;
                setTimeout(() => this.connectAndSubscribe(), jitteredWaitTime);
            };

            ws.onopen = () => { this.retryCount = 0; };
            
            ws.onmessage = (event) => {
                const trafficUpdate = JSON.parse(event.data);
                useCartStore.getState().applyTrafficPenalties(trafficUpdate);
            };
        }
    }

10. Offline-First & Graceful Degradation (IndexedDB)
----------------------------------------------------
Ein Einkauf dauert im Durchschnitt 40 Minuten. In dieser Zeit durchläuft der Einkaufswagen Zonen mit hoher Metalldichte (Tiefkühlschränke), die als faradayscher Käfig wirken und das WLAN blockieren. Verliert das Tablet die Verbindung, darf die bisherige Einkaufsliste auf keinen Fall gelöscht werden. 

Das Frontend implementiert das State-Management daher nicht nur im RAM. Fällt das Backend vollständig aus, greift das Prinzip der Graceful Degradation (fehlertolerante System-Reduktion). Das System verzichtet auf die synchrone ``localStorage`` API, welche den V8 Event-Loop blockiert und ein strenges Speicherlimit von ca. 5 MB besitzt. 

Stattdessen wird die asynchrone IndexedDB über den Wrapper ``localforage`` angesteuert. Fällt das WebSocket aus, liest das System die Topologie sicher von der SSD des Tablets und wechselt autark am Edge von dynamischem ML-Routing auf deterministisches, lokales TSP-Routing um.

.. code-block:: typescript

    import localforage from 'localforage';
    import { GraphTopology } from '../types/graph';

    localforage.config({
        driver: localforage.INDEXEDDB,
        name: 'SmartCartEdgeDB',
        storeName: 'topology_store',
        description: 'Persistiert Graphen für Offline-Graceful-Degradation'
    });

    export const EdgeCache = {
        async saveTopology(graph: { versionHash: string, topology: GraphTopology }): Promise<void> {
            // Asynchroner I/O blockiert den Main-Thread nicht
            await localforage.setItem('static_graph', graph);
        },
        
        async loadTopology(): Promise<{ versionHash: string, topology: GraphTopology } | null> {
            return await localforage.getItem('static_graph');
        },
        
        fallbackToLocalRouting(): void {
            useCartStore.setState({ isOfflineMode: true, routingStrategy: 'DETERMINISTIC_LOCAL' });
            console.warn("System degraded to offline local routing.");
        }
    };