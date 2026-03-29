# JMU Smart Cart: Predictive Routing & Digital Twin Engine

Das JMU Smart Cart Projekt ist ein Prototyp für einen vollständig datengetriebenen, intelligenten Supermarkt. Statt einer statischen Karte bietet das System auf einem Tablet am Einkaufswagen eine dynamische Navigation. Es kombiniert automatisiertes Data Engineering, Natural Language Processing für eine fehlerverzeihende Produktsuche und agentenbasierte Simulationen. Letztere fungieren als digitaler Zwilling und erzeugen die nötigen Trainingsdaten für ein Machine-Learning-Modell (XGBoost), welches aufkommende Staus in den Gängen prädiziert, um die Einkaufsroute des Kunden in Echtzeit zu optimieren.

## Schnellstart & Daten-Pipeline

Um das System schnell und ohne lokale Installationskonflikte zu testen, ist das Repository für GitHub Codespaces optimiert. Klicke auf GitHub einfach auf "Code" -> "Codespaces" -> "Create codespace on main", um die Umgebung zu starten. Bevor die Applikation laufen kann, müssen zunächst die Datenbasis und die Modelle generiert werden. Führe dazu nacheinander folgende Befehle im Terminal aus:

    # Phase 1: Topologie und Warenkatalog generieren
    python bls_to_csv.py
    python generate_data_driven_store.py

    # Phase 2: Simulation starten und KI-Modelle trainieren
    python simulation.py
    python model.py
    python train_model_optuna.py

## Applikation starten & bedienen

Sobald die Vorbereitungs-Pipeline abgeschlossen ist, startest du das interaktive Dashboard mit folgendem Befehl:

    python app.py

In GitHub Codespaces erscheint nun unten rechts ein Pop-up. Klicke auf "Open in Browser", um die Nutzeroberfläche zu öffnen. Die Bedienung simuliert den realen Einkaufsprozess: Suche zunächst nach Produkten (das System korrigiert Tippfehler dabei automatisch) und füge sie deiner Liste hinzu. Auf dem 2D-Grundriss erscheint die optimale Route als blaue Leitlinie, die sich dynamisch anpasst, um rot markierte Stauzonen proaktiv zu umfahren. 

Klicke auf die Produkte in deiner Liste, um das Einscannen am Regal zu simulieren. Zwingend vor der Halbzeit des Einkaufs greift das integrierte Warteschlangenmodell ein und weist dir die Kasse mit der geringsten Wartezeit zu. Für die Filialleitung existiert zudem ein verstecktes Admin-Dashboard: Klicke dreimal schnell in die äußerste obere linke Ecke des Grundrisses und gib die PIN ein, um die Live-Telemetrie einzusehen.

## Wissenschaftliche Dokumentation

Die tiefergehende architektonische Dokumentation (Sphinx) liegt im Verzeichnis "docs" bereit. Du kannst sie kompilieren und direkt über einen lokalen Webserver betrachten, indem du diese Befehle ausführst:

    cd docs
    make html
    python -m http.server 8000 --directory _build/html

Öffne anschließend einfach den angezeigten Port 8000 im Browser (in Codespaces über den "Ports"-Reiter abrufbar), um die formatierte Dokumentation inklusive aller Architektur-Plots zu lesen.
