HARD_USER_INPUT = """Design a 5-story office building in Shenzhen with a central atrium.
Overall dimensions: 40m x 30m footprint.
Ground floor height: 5.0m; standard floors (2F-5F): 4.0m each.
Central atrium: 16m x 12m, open from ground to roof with a skylight (U=2.0, SHGC=0.35).
Exterior walls: 200mm reinforced concrete + 80mm rock wool insulation + 15mm gypsum board.
Interior partition walls: double 13mm gypsum board with air gap.
Roof: 200mm concrete + 100mm XPS insulation + waterproof membrane.
Windows: Low-E double glazing (U=1.6, SHGC=0.35), 40% WWR on south/east facades, 25% on north/west.
Floor layout per typical floor (2F-5F):
  - South wing: 4 open-plan offices (~80 m² each), 10 people per office, 120 W/person.
  - North wing: 4 meeting rooms (~30 m² each), 12 people capacity, 100 W/person.
  - East wing: server room (~40 m²), 500 W/m² equipment load, 24/7 cooling to 22°C.
  - West wing: break room (~35 m²) and restrooms (~25 m²).
  - Core: elevator lobby and stairwell (traffic box, ~85 m², unconditioned).
Ground floor (1F):
  - Lobby: 200 m², 5m ceiling, 15 W/m² lighting, conditioned 20-26°C.
  - Retail/café space: 120 m², 20 people, 150 W/person, 12 W/m² lighting.
  - Mechanical/electrical room: 80 m², unconditioned, 200 W/m² equipment.
  - Loading dock: 60 m², unconditioned.
Schedules:
  - Office & meeting rooms: occupied 8am-7pm weekdays.
  - Lobby & retail: occupied 7am-9pm daily.
  - Server room: 24/7 operation.
  - Lighting: 12 W/m² offices, 15 W/m² lobby, 8 W/m² corridors (dusk-to-dawn + occupancy).
HVAC:
  - Offices/meeting rooms: VAV with reheat, cooling 24°C / heating 21°C occupied, setback 28°C / 15°C.
  - Server room: dedicated split DX system, 22°C year-round.
  - Lobby: fan coil units, cooling 26°C / heating 20°C.
  - Atrium: natural ventilation when outdoor temp 18-28°C, otherwise mechanical.
Infiltration: 0.5 ACH for all conditioned zones.
Ventilation: per ASHRAE 62.1 (office 8.5 L/s/person, meeting 5 L/s/person)."""

SIMPLE_USER_INPUT = """Design a Large 5-zone office in Shenzhen.
10m x 8m x 3m (one floor). Brick 100mm + EPS 50mm + gypsum 15mm exterior walls,
double-glazed window (U=1.8, SHGC=0.4) covering 30% of the south facade.
Occupancy: 8 people, 8am-6pm weekdays, office activity (120 W/person).
Lighting: 10 W/m^2, on 8am-6pm weekdays.
HVAC: ideal loads, heating 20C / cooling 24C during occupied hours,
setback 15C / 28C otherwise."""
