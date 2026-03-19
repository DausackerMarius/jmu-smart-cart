Einleitung & Business Case
==========================

Das Projekt "JMU Smart Supermarket" stellt einen Prototyp für den intelligenten Einzelhandel der Zukunft dar. In einem Marktumfeld, das zunehmend durch Zeitdruck und Effizienzsteigerung geprägt ist, bildet die Verknüpfung von **Data Engineering**, **Machine Learning** und **Operations Research** den entscheidenden Wettbewerbsvorteil.

Problemstellung und Motivation
------------------------------
Der stationäre Einzelhandel steht vor einer technologischen Zäsur. Während E-Commerce-Plattformen durch personalisierte Empfehlungen und effiziente Suchalgorithmen glänzen, bleibt der physische Einkauf oft ineffizient. 

Drei Kernprobleme wurden als primäre Hemmschuhe für das Kundenerlebnis identifiziert:

1. **Informationsasymmetrie:** Kunden finden Produkte aufgrund unklarer Regalstrukturen oder falscher Artikelbezeichnungen nicht unmittelbar.
2. **Statische Navigation:** Klassische Einkaufslisten berücksichtigen weder die physische Anordnung der Warengruppen noch die aktuelle Kundenstromdichte (Stausituation).
3. **Queueing-Ineffizienz:** Wartezeiten an Kassenbereichen werden stochastisch nicht erfasst, was zu einer suboptimalen Auslastungsverteilung der Ressourcen führt.

Wissenschaftlicher Ansatz: Der Smart Cart
-----------------------------------------
Die vorliegende Systemarchitektur löst diese Probleme durch einen **hybriden Technologie-Stack**. Dabei werden nicht nur triviale Algorithmen implementiert, sondern komplexe mathematische Herausforderungen adressiert:

* **NLP-basierte Inferenz:** Mittels einer Machine-Learning-Pipeline werden semantische Unschärfen in Nutzeranfragen aufgelöst. Hierbei kommt eine Kombination aus phonetischen Algorithmen und Logistischer Regression zum Einsatz, um eine Robustheit gegenüber Tippfehlern zu garantieren.
* **Dynamische Graphen-Optimierung:** Das Routing wird als **Traveling Salesperson Problem (TSP)** modelliert. Da dieses Problem in der Komplexitätstheorie als NP-schwer gilt, implementiert das System ein Strategy-Pattern, das dynamisch zwischen exakten Algorithmen (Held-Karp) und physikalisch motivierten Metaheuristiken (Simulated Annealing, ACO) skaliert.
* **Prädiktives Traffic-Management:** Durch die Integration eines Traffic-Modells werden Kanten im Supermarkt-Graphen gewichtet, um den "kürzesten Weg" als Funktion der Zeit und nicht nur der Distanz zu definieren.
* **Stochastische Modellierung:** Die Kassenbelegung wird über ein **M/M/1/K-Warteschlangenmodell** nach Kendall abgebildet, um eine präzise Prognose der Wartezeiten unter Berücksichtigung begrenzter Systemkapazitäten zu ermöglichen.

Zielsetzung der Dokumentation
-----------------------------
Dieses Dokument dient als technische Referenz und wissenschaftliche Einordnung des Gesamtsystems. Es dokumentiert die End-to-End Pipeline – von der ETL-Verarbeitung der Rohdaten bis hin zur Echtzeit-Visualisierung im Frontend – und evaluiert die Leistungsfähigkeit der eingesetzten Modelle unter realnahen Bedingungen.