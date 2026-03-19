Data Engineering & Supermarkt-Ontologie (ETL)
=============================================

Ein intelligenter Supermarkt-Graph benötigt eine valide, rauscharme Datenbasis. Da das System nicht auf statische, händisch gepflegte Dummydaten vertraut, implementiert es eine vollständig autonome **ETL-Pipeline** (Extract, Transform, Load). 

Diese Pipeline, orchestriert durch die Klasse ``MasterOrchestrator``, generiert bei jedem Systemstart einen konsistenten "digitalen Zwilling" des Marktes, indem sie Rohdaten des Bundeslebensmittelschlüssels (BLS) extrahiert, linguistisch normalisiert und topologisch zuweist.

1. Extract: Asynchrone Datenbeschaffung (BLSDownloader)
-------------------------------------------------------
Die Grundlage der Ontologie bildet der externe BLS-Datensatz. Der initiale Zugriff auf diese Massendaten birgt das Risiko von "Out-of-Memory"-Fehlern (OOM) im Arbeitsspeicher. 

Die Klasse ``BLSDownloader`` löst dieses Problem durch ein speichereffizientes Streaming-Verfahren:
* **Chunking-Paradigma:** Anstatt die Excel-Datei monolithisch in den RAM zu laden, wird der HTTP-Response-Stream in iterativen 8-Kilobyte-Blöcken (Chunks) gelesen und direkt auf die Festplatte geschrieben.
* **Target Column Detection:** Da externe Datensätze variierende Schemata aufweisen können, iteriert der ``CSVBuilder`` heuristisch über die Spalten und identifiziert die korrekten Features (BLS-Code und Originalname) autonom anhand von RegEx-Mustern und Vokaldichten.

2. Transform: Semantische Bereinigung (TextSanitizer)
-----------------------------------------------------
Rohe Ernährungsdatenbanken sind durchdrungen von B2B-Terminologie, die das spätere NLP-Klassifikationsmodell durch Rauschen (Noise) korrumpieren würde. Die Klasse ``TextSanitizer`` operiert als deterministischer linguistischer Filter:

* **Noise Reduction:** Eine vordefinierte Ontologie an Störwörtern (``B2B_NOISE_WORDS``) eliminiert industrielle Fachbegriffe (z. B. "Schlachtkörper", "Kutterhilfsmittel"). Parallel werden englische Fremdwörter (``EN_KILLER_WORDS``) entfernt, um den semantischen Raum des TF-IDF-Vektorisierers strikt auf die deutsche Sprache zu begrenzen.
* **RegEx-Sanitization:** Komplexe reguläre Ausdrücke entfernen irrelevante Metadaten. So löscht der Befehl ``re.sub(r'\s*\(.*?\)', '', name)`` sämtliche Klammerzusätze, während signalwortbasierte Cuts (z. B. vor dem Wort "mindestens") den String auf seine absolute Kernidentität reduzieren.
* **Zero-Shot Mapping:** Der ``SupermarketMapper`` übersetzt die abstrakten alphanumerischen BLS-Codes (z. B. "M" für Milchprodukte) über deterministische Dictionaries in unsere logischen Supermarkt-Zonen ("Kühlregal Molkerei").

3. Load: Mathematische Allokation (StoreBuilder)
------------------------------------------------
Nach der Generierung der Ground-Truth-CSV müssen die bereinigten Produkte physisch auf die leeren Knoten (Flexible Zones) des Routing-Graphen verteilt werden. Da die Anzahl der Knoten stark limitiert ist, löst der ``StoreBuilder`` ein Ressourcenallokationsproblem.

**3.1 Algorithmische Sitzverteilung (Sainte-Laguë-Verfahren)**
Um "Geister-Regale" zu verhindern und eine proportionale Auslastung zu garantieren, adaptiert das System das aus der Wahlarithmetik bekannte Höchstzahlverfahren nach Sainte-Laguë. Die Zuweisung berechnet sich iterativ durch den Quotienten:

.. math::

   Q = \frac{V}{s + 0.5}

Hierbei ist :math:`V` die absolute Anzahl der extrahierten Artikel einer Kategorie und :math:`s` die Anzahl der bereits zugewiesenen Regal-Knoten. Der Divisor +0.5 verhindert die algorithmische Benachteiligung kleinerer Sortimente.

**3.2 Stochastische Preisgenerierung (PricingEngine)**
Da eine statische Preiszuweisung die Immersion brechen würde, generiert die ``PricingEngine`` dynamische Kaufpreise auf Basis warengruppenspezifischer Normalverteilungen. Die Preisfunktion :math:`P(x)` zieht für jeden Artikel einen Zufallswert basierend auf Erwartungswert :math:`\mu` und Standardabweichung :math:`\sigma`:

.. math::

   P(x) = \max(\mathcal{N}(\mu, \sigma), P_{min})

4. Architektonischer Output
---------------------------
Der finale Export der Pipeline serialisiert den Graphen-Zustand. Das Inventar, angereichert mit AI-Confidence-Scores und stochastischen Preisen, wird in die ``products.json`` geschrieben, während die topologische Zuordnung in der ``routing_config.json`` fixiert wird.