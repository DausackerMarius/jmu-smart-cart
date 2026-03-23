Frontend-Architektur: Reaktive UI, Edge-Computing & Hardware-Integration
========================================================================

Die größte Herausforderung des JMU Smart Cart Frontends (geschrieben in React und TypeScript) besteht darin, eine riesige Menge an Datenbergen in Echtzeit auf den Bildschirm zu zeichnen, ohne dass das Tablet abstürzt. Das Tablet am Einkaufswagen operiert als sogenanntes **Edge-Device** (ein Endgerät am äußersten Rand des Netzwerks). Das bedeutet: Es ist kein leistungsstarker Server, sondern ein kleiner Computer mit schwachem Prozessor, der vom Akku lebt und bei zu viel Rechenlast heiß wird und sich selbst drosselt (**Thermal Throttling**).

Das Tablet muss vier schwere Aufgaben gleichzeitig erledigen:
1. Den Wagen im Supermarkt lokalisieren (Wo bin ich?).
2. Physische Barcodescans des Lasers verarbeiten.
3. Die optimale Route als blaue Linie durch hunderte Gänge flüssig auf den Bildschirm zeichnen.
4. Minütliche Stau-Warnungen (Traffic Updates) aus dem Internet empfangen und verarbeiten.

Wenn der Browser versucht, all das gleichzeitig im Hauptprozessor zu erledigen, entsteht **Layout Thrashing**. 
*Verständnis-Exkurs:* Stellen Sie sich einen Maler vor, der ein Bild malt, während ihm jemand ständig die Leinwand wegzieht und eine neue hinstellt. Der Maler kommt nicht voran und das Bild ruckelt. In der Informatik bedeutet das: Die App friert ein. 

Um flüssige 60 Bilder pro Sekunde (FPS) zu garantieren, müssen Datenempfang, Zustandsverwaltung und das reine Zeichnen der Grafik strikt voneinander getrennt werden. Wie das genau funktioniert, erklärt dieses Kapitel.

1. System-Bootstrapping & Backend-for-Frontend (BFF)
----------------------------------------------------
Wenn das Tablet morgens eingeschaltet wird, ist sein Arbeitsspeicher komplett leer. Es weiß nicht, wie der Supermarkt aussieht oder welche Produkte es gibt. Das anfängliche Laden dieser Daten nennt man **Bootstrapping** (das System zieht sich an den eigenen Schnürsenkeln hoch).

Hierfür nutzt die Architektur ein **Backend-for-Frontend (BFF)** Muster.
*Verständnis-Exkurs:* Normalerweise müsste das Tablet beim Start 20 verschiedene Abteilungen im Backend anrufen: "Gib mir die Wände", "Gib mir die Regale", "Gib mir die Preise". Das dauert lange und kostet Akkuleistung. Das BFF-Muster funktioniert wie ein persönlicher Kellner: Das Tablet ruft nur ein einziges Mal den BFF-Server an, dieser sammelt im Hintergrund alle Daten aus den verschiedenen Datenbanken zusammen und serviert dem Tablet ein einziges, fertiges "Menü" (Payload).

.. code-block:: typescript

   import { useEffect } from 'react';
   import { useCartStore } from './store';
   import { apiGateway } from './api';
   import { trafficSocketManager } from './websockets';

   export function useSystemBootstrap() {
       // useEffect mit leeren Klammern [] bedeutet: Führe dies exakt 1x beim Start aus
       useEffect(() => {
           async function initializeCart() {
               useCartStore.setState({ systemStatus: 'BOOTING' });
               try {
                   // 1. Der BFF-Kellner bringt die fertige Karte und alte Sitzungsdaten
                   const [topology, session] = await Promise.all([
                       apiGateway.fetchStaticMap(),
                       apiGateway.restoreSession() // Falls das Tablet abgestürzt war, laden wir den alten Warenkorb
                   ]);
                   
                   // 2. Wir speichern die starren Wände und Regale im Arbeitsspeicher
                   useCartStore.setState({ 
                       mapData: topology, 
                       cart: session.items,
                       systemStatus: 'READY' 
                   });
                   
                   // 3. Erst wenn die starre Karte da ist, öffnen wir den Funkkanal (WebSocket) 
                   // für die sich ständig ändernden Stau-Meldungen.
                   trafficSocketManager.connectAndSubscribe();
               } catch (error) {
                   handleFatalBootError(error);
               }
           }
           initializeCart();
       }, []); 
   }

2. Indoor-Lokalisierung & Sensor Fusion (Kalman-Filter)
-------------------------------------------------------
Woher weiß der blaue Punkt auf der Karte, wo der Einkaufswagen gerade steht? GPS funktioniert in einem Supermarkt aus Stahlbeton nicht. Das System nutzt daher **Bluetooth Low Energy (BLE)** Beacons. Das sind kleine Funkfeuer, die an der Supermarktdecke hängen.

Das Problem: Bluetooth-Signale springen und schwanken extrem, wenn z. B. ein Mensch durch die Funkstrecke läuft. Würde das Tablet die rohen Signale nutzen, würde der Standort-Punkt auf der Karte wild hin und her springen.
Die Lösung ist ein mathematischer **Kalman-Filter**.
*Verständnis-Exkurs:* Wenn Sie im Dunkeln laufen, sehen Sie schlecht (das schwankende Bluetooth-Signal). Aber Sie wissen ungefähr, wie schnell Sie laufen und in welche Richtung (Ihre eigene Bewegungsschätzung). Der Kalman-Filter ist ein Algorithmus, der beides kombiniert: Er glättet die Ausreißer des Funksignals auf Basis der wahrscheinlichen Physik. Aus dem Chaos wird eine ruhige, präzise Linie.

$$ d = 10^{\frac{P_{Tx} - RSSI}{10 \cdot n}} $$
*(Diese Formel rechnet die Bluetooth-Signalstärke $RSSI$ in Meter $d$ um).*

.. code-block:: typescript

   import { useEffect, useRef } from 'react';
   import { KalmanFilter } from './math/kalman';

   export function useIndoorPositioning() {
       // Initialisierung des Filters (R = Rauschen des Bluetooth, Q = Ungenauigkeit unserer Bewegung)
       const filter = useRef(new KalmanFilter({ R: 0.01, Q: 3 })); 

       useEffect(() => {
           const processBeaconSignal = (rssi: number, txPower: number) => {
               // 1. Wie weit ist das Beacon weg? (Formel von oben)
               const rawDistance = Math.pow(10, (txPower - rssi) / (10 * 2.5));
               
               // 2. Glättung des zitternden Signals durch den Kalman-Filter
               const smoothedDistance = filter.current.filter(rawDistance);
               
               // 3. Berechnung des genauen X/Y Punktes auf der Karte und Speicherung
               const newPosition = calculateTrilateration(smoothedDistance, knownBeacons);
               useCartStore.setState({ currentPosition: newPosition });
           };

           bluetoothService.subscribeToRSSI(processBeaconSignal);
           return () => bluetoothService.unsubscribe();
       }, []);
   }

3. Barcode-Scanner & Schutz vor Hacker-Eingaben
-----------------------------------------------
Wenn der Kunde einen Artikel einscannt, nutzt die Hardware das **Keyboard Wedge Pattern**. 
*Verständnis-Exkurs:* Der rote Laser-Scanner tut so, als wäre er eine Computertastatur. Er liest den Strichcode "12345" und "tippt" diese Zahlen in Millisekunden unsichtbar in das Tablet ein, gefolgt von der Enter-Taste.

Dies birgt jedoch ein Sicherheitsrisiko (**HID Injection Attack**). Ein böswilliger Kunde könnte einen eigenen Strichcode drucken, der statt einer Zahl einen Programmierbefehl enthält (z. B. "Lösche die Datenbank"). Das Frontend wehrt dies durch strenge **Sanitisierung** (Desinfektion) ab:

.. code-block:: typescript

   export function useHardwareScannerIntegration() {
       const buffer = useRef<string>(''); // Hier sammeln wir die getippten Zahlen
       const lastKeyTime = useRef<number>(Date.now());

       useEffect(() => {
           const handleHardwareScan = (e: KeyboardEvent) => {
               const currentTime = Date.now();
               
               // Hardware-Filter: Ein echter Scanner tippt Zeichen in unter 50 Millisekunden.
               // Tippt jemand von Hand auf den Touchscreen, ist das langsamer und wird hier blockiert.
               if (currentTime - lastKeyTime.current > 50) buffer.current = ''; 
               lastKeyTime.current = currentTime;

               // "Enter" bedeutet: Der Scan ist fertig.
               if (e.key === 'Enter' && buffer.current.length >= 8) {
                   e.preventDefault();
                   
                   // SECURITY (Sanitisierung): Wir filtern alles heraus, was keine Zahl von 0-9 ist.
                   // Hacker-Befehle werden hier einfach weggeschnitten.
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

4. React-Optimierung: Shallow Equality (Oberflächlicher Vergleich)
------------------------------------------------------------------
Sobald das Produkt gescannt ist, fragt das Tablet den Backend-Server: "Was ist meine neue, beste Route zur Kasse?". 
Wenn der Server eine neue Route schickt, würde eine normale App den gesamten Bildschirm komplett neu aufbauen. Das verbraucht extrem viel Akkuleistung.

Das System nutzt daher **Shallow Equality Checks** (oberflächliche Gleichheitsprüfung). 
*Verständnis-Exkurs:* Wenn Sie prüfen wollen, ob zwei dicke Bücher identisch sind, können Sie entweder jede einzelne Seite lesen (Deep Check = sehr langsam) oder einfach nur schauen, ob die ISBN-Nummer auf der Rückseite gleich ist (Shallow Check = blitzschnell). 
Das Tablet prüft nur die "ID" der Route im Arbeitsspeicher. Wenn die Route trotz des neuen Produkts geometrisch gleich geblieben ist, bricht das Tablet das Neuzeichnen der Karte ab und fügt das Produkt nur still in den Warenkorb ein.

5. Grafik-Beschleunigung: Web Worker & Zero-Copy
------------------------------------------------
Wenn sich die Route aber ändert, muss die Karte neu gezeichnet werden. Tausende Wegpunkte zu malen, würde das Tablet komplett auslasten; der Kunde könnte in dieser Zeit keine Buttons mehr drücken.

Die Lösung ist ein **Web Worker**. 
*Verständnis-Exkurs:* Stellen Sie sich vor, der Hauptprozessor des Tablets ist der Chefkoch. Wenn er selbst Kartoffeln schälen muss (die Karte zeichnen), kann er keine Bestellungen (Kundenklicks) mehr annehmen. Ein Web Worker ist ein Sous-Chef (ein Hilfsarbeiter im Hintergrund). Der Chefkoch ruft dem Sous-Chef zu: "Zeichne mir mal diese Route!". 

Damit die Pixel auf den Bildschirm passen, nutzt der Sous-Chef eine **Affine Transformation**: Er rechnet die Meter aus dem Supermarkt über eine Formel in Bildschirm-Pixel um:
$$ P_{pixel} = \begin{pmatrix} s & 0 \\ 0 & s \end{pmatrix} \cdot \vec{P}_{meter} + \vec{t} $$

.. code-block:: typescript

   // --- MapWorker.ts (Der Sous-Chef, der im Hintergrund arbeitet) ---
   self.onmessage = (evt: MessageEvent) => {
       const { routeCoords, scale, offsetX, offsetY, offscreenCanvas } = evt.data;
       
       // Wir bereiten die unsichtbare Leinwand vor
       const ctx = offscreenCanvas.getContext('2d', { alpha: false }); 
       ctx.fillStyle = '#ffffff';
       ctx.fillRect(0, 0, offscreenCanvas.width, offscreenCanvas.height);
       
       // Wir zeichnen die Route als blaue Linie
       ctx.beginPath();
       ctx.strokeStyle = '#0066cc'; 
       ctx.lineWidth = 4;
       
       routeCoords.forEach((p, i) => {
           // Umrechnung von Supermarkt-Metern in Tablet-Pixel
           const pixelX = (p.x * scale) + offsetX;
           const pixelY = (p.y * scale) + offsetY;
           if (i === 0) ctx.moveTo(pixelX, pixelY);
           else ctx.lineTo(pixelX, pixelY);
       });
       ctx.stroke(); 
       
       // Übergabe des fertigen Bildes an den Chefkoch (Main-Thread)
       const bitmap = offscreenCanvas.transferToImageBitmap();
       self.postMessage({ bitmap }, [bitmap]);
   };

**Der absolute Performance-Trick (Zero-Copy):**
Der Befehl ``transferToImageBitmap()`` ist Magie. Anstatt dass der Sous-Chef das fertige Bild mühsam kopiert und dem Chefkoch zuschickt (was den Arbeitsspeicher füllt), übergibt er einfach nur die Besitzurkunde (den Speicher-Pointer) für das Bild. Der Chefkoch darf sofort auf das Bild zugreifen. Das nennt sich **Zero-Copy** (Null-Kopie) und garantiert butterweiche Grafiken.

6. Schutz vor Datenfluten: Backpressure & Ringpuffer
----------------------------------------------------
Während der Fahrt sendet das Internet sekündlich Stau-Warnungen auf das Tablet. Wenn 50 Einkaufswagen gleichzeitig einen Stau melden, entsteht ein Daten-Sturm. Würde das Tablet versuchen, alle Meldungen sofort zu verarbeiten, würde es abstürzen.

Das System nutzt **Backpressure Handling** (Gegendruck).
*Verständnis-Exkurs:* Stellen Sie sich das Tablet wie einen Türsteher vor einem Club vor. Wenn plötzlich 50 Gäste (Datenpakete) gleichzeitig durch die Tür stürmen wollen, gibt es Chaos. Der Türsteher (unser Code) sperrt die Tür ab, lässt die Daten in einem **Ringpuffer** (einem digitalen Warteraum) sammeln, und lässt sie dann geordnet in kleinen, gut verdaulichen Gruppen exakt passend zur Bildwiederholrate des Bildschirms (60 Mal pro Sekunde) ein.

7. Die Vorhersage der Kasse: Halftime-Prädiktion
------------------------------------------------
Ein riesiges Problem bei Navigationssystemen ist es, wenn sie zu spät Bescheid geben. Sagt das Tablet dem Kunden erst drei Meter vor den Kassen, dass Kasse 4 die kürzeste Schlange hat, muss er abrupt umdrehen und sich durch die Menschenmassen zwängen.

Die Lösung ist ein **Architektonischer Halftime-Trigger**.
Das Tablet wartet nicht bis zum Schluss. Sobald exakt die Hälfte der Einkaufsliste abgehakt ist (Halftime / Halbzeit), rechnet das Tablet stochastisch aus, welche Kasse in 10 Minuten frei sein wird, und integriert diesen Zielpunkt sofort heimlich in die laufende Route. So wird der Kunde völlig unauffällig und staufrei zur richtigen Kasse manövriert.

.. code-block:: typescript

   import { useEffect } from 'react';
   import { useCartStore } from './store';
   import { api } from './api';

   export function useHalftimeCheckoutPrediction() {
       useEffect(() => {
           // Das Tablet lauscht lautlos, wie viele Artikel gescannt wurden
           const unsubscribe = useCartStore.subscribe(
               (state) => ({ cartLen: state.cart.length, total: state.shoppingList.length }),
               (current) => {
                   if (current.total === 0) return;
                   
                   // Berechnung des Fortschritts (z.B. 5 von 10 Artikeln = 0.5)
                   const progress = current.cartLen / current.total;
                   
                   // Sobald 50% erreicht sind, blockieren wir diesen Trigger für die Zukunft (checkoutPredicted)
                   if (progress >= 0.5 && !useCartStore.getState().checkoutPredicted) {
                       useCartStore.setState({ checkoutPredicted: true }); 
                       
                       // Wir fragen das Backend: "Welche Kasse wird später frei sein?"
                       api.triggerCheckoutPrediction().then(optimalExitNode => {
                           // Wir hängen die optimale Kasse still und heimlich ans Ende der Route an
                           useCartStore.getState().appendExitNodeToRoute(optimalExitNode);
                       });
                   }
               },
               { equalityFn: shallow }
           );
           return unsubscribe;
       }, []);
   }

8. Netzwerkausfälle: Graceful Degradation & Thundering Herd
-----------------------------------------------------------
Im Supermarkt gibt es viele Metallregale, die WLAN-Signale blockieren (Faraday-Käfig). Das WLAN wird also ständig abbrechen. Die App darf dann nicht einfach eine Fehlermeldung zeigen. 

Das nennt sich **Graceful Degradation** (elegantes Herabstufen): Ohne WLAN greift das Tablet einfach auf die lokal gespeicherte Offline-Karte zurück und navigiert normal weiter, nur eben ohne Live-Stauwarnungen.

Das eigentliche Problem entsteht, wenn der Wagen aus dem Funkloch herausfährt und das WLAN zurückkehrt. Würden 100 Einkaufswagen exakt im selben Moment versuchen, sich wieder mit dem Server zu verbinden, würde das den Server durch Überlastung in die Knie zwingen (Das **Thundering Herd Problem** / Die donnernde Herde).

Die Lösung ist ein **Exponential Backoff mit Jitter**:
Das Tablet wartet nach einem Fehlversuch erst 1 Sekunde, dann 2, dann 4, dann 8 ($2^n$). Der **Jitter** addiert eine winzige Zufallszahl auf diese Wartezeit (z. B. 0.3 Sekunden). Das bewirkt, dass sich die 100 Einkaufswagen nicht gleichzeitig, sondern leicht versetzt, wie bei einem Reißverschlussverfahren, sanft wieder mit dem Server verbinden.