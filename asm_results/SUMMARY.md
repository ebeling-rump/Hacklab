# Active Spatial Mapping: TabPFN vs GP

Stand: 2026-11-21

TabPFN bringt bei der Vorhersage unbeobachteter Belegung keinen Vorteil: Ein Gaussian Process ist in allen 10 Läufen besser, auch gegen TabPFN mit neu gewichteten Trainingsdaten oder Kontext-Merkmalen.

## Aufbau

Der Unitree A1 scannt in der bestehenden MuJoCo-Simulation mit seinem LiDAR aus 5 Posen. Daraus entsteht ein Belegungsgitter mit 80 × 80 Zellen zu je 0,1 m, das etwa 77 % der Fläche abdeckt. Alle Modelle lernen aus denselben 2000 beobachteten Zellen und werden nur auf den Zellen bewertet, die der LiDAR nie gesehen hat. Gemessen wurde in `env_A` und `env_B` mit je 5 Seeds.

## Ergebnis

| Modell | Brier ↓ | Precision | Recall | AUROC | Laufzeit pro Lauf | Besser als GP |
| --- | --- | --- | --- | --- | --- | --- |
| GP | 0,028 | 0,90 | 0,77 | 0,967 | 23 s | – |
| TabPFN mit Kontext-Merkmalen | 0,051 | 0,85 | 0,58 | 0,941 | 3 s | 0/10 |
| TabPFN | 0,050 | 0,88 | 0,48 | 0,925 | 3 s | 0/10 |
| TabPFN neu gewichtet (1 : 3) | 0,059 | 0,71 | 0,66 | 0,914 | 3 s | 0/10 |
| Nächste beobachtete Zelle | 0,063 | 0,75 | 0,70 | 0,833 | 0,2 s | – |
| GP neu gewichtet (1 : 3) | 0,074 | 0,67 | 0,88 | 0,947 | 0,4 s | – |

Mittelwerte über 10 Läufe. TabPFN läuft über die Cloud-API, seine Laufzeit enthält die Netzwerk-Latenz.

## Einordnung

- Nur 11 % der unbeobachteten Zellen sind belegt, deshalb erreicht schon „immer frei“ 89 % Accuracy. Verglichen wird daher über Brier-Score und AUROC.
- Der GP ist auch bei diesen beiden schwellwertunabhängigen Maßen vorn. Der Abstand liegt also nicht nur an der Schwelle 0,5.
- Ein Ground-Truth-Leakage gibt es nicht: Trainingslabels stammen nur aus dem LiDAR, die Ground Truth dient nur zur Bewertung.
- Ein GP mit neu gewichteten Daten braucht nur 0,4 s und hat den höchsten Recall. Für schnelle Exploration ist er eine praktikable Option.

Details mit Standardabweichungen, paarweisen Vergleichen und Karten: [sweep_tabpfn/summary.md](sweep_tabpfn/summary.md) und [sweep/summary.md](sweep/summary.md). Reproduzieren mit `python asm_sweep.py` (siehe Kopf der Datei). Navigation ist noch nicht angebunden.
