GitHub Codespaces Schnellstart & Daten-Pipeline
===============================================

Um das Projekt ohne lokale Installation direkt im Browser zu evaluieren, unterstützt die Architektur die Bereitstellung via GitHub Codespaces. Dieses Tutorial führt durch den Prozess von der Rohdaten-Konvertierung bis zum Start des prädiktiven Systems.

1. Instanziierung des Codespace
-------------------------------
1. Navigieren Sie zum GitHub-Repository des Projekts.
2. Klicken Sie auf die Schaltfläche **"Code"**, wählen Sie den Reiter **"Codespaces"** und klicken Sie auf **"Create codespace on main"**.
3. Der Container wird automatisch vorbereitet und installiert alle Abhängigkeiten aus der ``requirements.txt``.

2. Die End-to-End Daten-Pipeline (Initialisierung)
--------------------------------------------------
Bevor der primäre Server gestartet werden kann, müssen die Daten-Artefakte und ML-Modelle in einer strikten logischen Reihenfolge generiert werden. Führen Sie folgende Befehle nacheinander im Terminal aus:

**Phase A: Data Engineering & Topologie**
In dieser Phase werden die Basisdaten für den Warenkatalog und die physische Struktur des Marktes erzeugt.

.. code-block:: bash

    # 1. Konvertierung des Bundeslebensmittelschlüssels (BLS) in ein flaches Format
    python bls_to_csv.py

    # 2. Erzeugung der datengetriebenen Topologie (JSON-Graphen & Regallayouts)
    python generate_data_driven_store.py

**Phase B: Simulation & Machine Learning**
Hier wird das stochastische Kundenverhalten simuliert, um die notwendigen Trainingsdaten für die prädiktiven Modelle zu generieren.

.. code-block:: bash

    # 3. Start der agentenbasierten Simulation (Generiert traffic_data.parquet)
    # Hinweis: Dieser Schritt ist die Voraussetzung für das Training der Stau-KI.
    python simulation.py

    # 4. Training der NLP-Klassifikatoren (Modelle für die Suchfunktion)
    python model.py

    # 5. Hyperparameter-Tuning & Training der Traffic-KI (XGBoost) via Optuna
    # Dieser Schritt nutzt die Daten aus der zuvor ausgeführten Simulation.
    python train_model_optuna.py

3. Start der Applikation
------------------------
Sobald alle Modelle (``.pkl`` / ``.joblib``) und Artefakte im Verzeichnis vorliegen, kann das Gesamtsystem gestartet werden:

.. code-block:: bash

    python app.py

**Zugriff auf die Benutzeroberfläche:**
GitHub Codespaces erkennt den Port **8050** automatisch. Klicken Sie auf das erscheinende Pop-up **"Open in Browser"** oder nutzen Sie den Reiter **"Ports"** in der unteren Leiste, um das Dashboard und die interaktive Karte zu öffnen.

4. Kompilierung & Anzeige der Dokumentation
-------------------------------------------
Um diese Dokumentation (Sphinx) innerhalb des Codespace zu rendern und als Webseite zu betrachten:

.. code-block:: bash

    # 1. HTML-Build erstellen
    cd docs
    make html

    # 2. Dokumentation via Python-Webserver auf einem separaten Port bereitstellen
    python -m http.server 8000 --directory _build/html

Klicken Sie im Port-Reiter auf den neu erschienenen Port **8000**, um die formatierte Dokumentation im Browser zu lesen.

5. Wichtige Hinweise zur Cloud-Umgebung
---------------------------------------
* **Rechenleistung:** Das Training via ``train_model_optuna.py`` ist CPU-intensiv. Sollte der Prozess im Standard-Codespace langsam sein, empfiehlt es sich, in den Einstellungen einen Maschinentyp mit mehr Kernen zu wählen.
* **Persistenz:** Einmal generierte Daten (Parquet) und trainierte Modelle bleiben im Speicher des Codespace erhalten. Ein erneutes Durchlaufen der gesamten Pipeline ist bei einem bloßen Neustart der ``app.py`` nicht erforderlich.