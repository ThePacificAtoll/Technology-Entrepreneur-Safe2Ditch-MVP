# Safe2Ditch-style landing simulation

A one-page guide to the generated neighborhood, emergency landing logic, trials, and results.

## Map and suburban layout

Each seeded design is a **1,000 × 1,000 grid of 1 m × 1 m cells**, representing 1 km². It is a procedural neighborhood, not a real geographic map. Roads are 10 or 15 m wide. Residential blocks contain two back-to-back rows of 20 × 30 m lots; each lot has a 10 × 15 m house, a street-facing setback, and a rear yard adjoining another rear yard. Every lot faces one road or two adjacent roads.

| Cell type | Assigned risk | Meaning |
|---|---:|---|
| Open | 0.05 | Yards and multi-block undeveloped swaths |
| Park | 0.20 | Road-accessible block used as a park |
| Road | 0.70 | Street surface |
| House | 0.90 | Building footprint |
| School | 0.95 | Road-accessible block used as a school |
| Person | 1.00 | Dynamic 5 × 5 m danger zone overriding the static cell |

Each map has **four parks** (one per quadrant, at least 300 m apart) and **two schools** (at least 500 m apart). Three undeveloped swaths merge 2–4 former housing blocks apiece, removing interior road segments; their cells remain **Open**. Parks and schools occupy selected road-bounded blocks, typically rectangular rather than square.

## Three landing strategies

| Strategy | How its final landing cell is selected |
|---|---|
| **Drop in place** | Land on the cell directly below the failure point; no travel or site search. |
| **Map only** | Within the remaining range, choose the site minimizing mapped risk + 0.0004 × distance in meters. The map does not know where people are. |
| **Map + verification** | Rank the same sites, inspect actual conditions, and take the first site with risk below 0.60. Try up to five sites at least 5 m apart; if none clears, use the last inspected site. |

## Trials

The script generates **10 maps** and runs **100 failures per map** (1,000 total). Each trial redraws 1,500 person centers at allowed park or road/open-border positions; each center marks a 5 × 5 m danger zone. It also draws a new drone position and a remaining range uniformly between **25 and 150 m**. All three strategies face the same map, people, position, and range in that trial. A new map is generated after its 100 trials. Fixed seeds make reruns reproducible.

## Reading the outputs

`simulation_report.txt` gives one section per map and an overall section. The map header identifies its image and lists parcel counts, spacing, and residential lots.

- **Land coverage** is the percentage of map cells by type. *Static* is the original terrain; *Effective* is averaged over trials after person zones override covered cells.
- **Landings** is the percentage of final landing cells of each actual type, using 100 landings **per strategy** on each map or 1,000 per strategy overall.
- **High risk** is the share landing on a cell with score at least 0.60 (road, house, school, or person).
- **Mean risk** averages assigned scores; it is not a probability of injury.

The `maps/` folder contains each static map. The `charts/` folder contains one four-panel PNG per map and one overall chart. CSV files contain the same quantitative summaries.
