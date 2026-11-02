# Fastest run against the rest: production and development rates

Every combined-system (AOB + NOB + CMX) run of the 100x100 heatmap sweep
with the finer O2 grid (`../heatmap/results_all_N99_T150/`) is compared
with the run that reaches the boundary first: NH4_BC = 1000 uM,
O2_BC = 250 uM, stopping at T* = 47.5 solver units = 50.1 days. All values
are read at T*, when no run has reached the boundary. Preset `eloi`, setup
as in `../README.md`.

## Production rates

Volume-integrated reaction rates over the whole domain, stored for every
step of every run; in umol per mm depth per day (totals in umol per mm
depth). NH4 consumption is shown positive; NO2 is net production (AOB
production minus NOB consumption). NO3 is the combined output of NOB and
CMX; the stored rates do not split it by species.

| at T* | fastest run | rest of the map |
|---|---|---|
| NH4 consumption rate | 0.84, the highest | 0.82 at NH4 387 uM; 0.44 at O2 175, 0.19 at O2 100, 0.05 at O2 40 (high NH4); 0.0006 at the lowest NH4 |
| NH4 consumed up to T* | 17.6, the highest | 11 at O2 175, 5.4 at O2 100, 1.7 at O2 40 |
| NO3 production rate | 0 (stops at 25-32 days) | up to 0.14 (NH4 150, O2 250) |
| NO3 produced up to T* | 1.4 | up to 5.8 (NH4 150, O2 250) |
| NO3 share of oxidised N up to T* | 0.08 | 1.00 at NH4 <= 8.6 uM; 0.11-0.23 at high NH4 with O2 >= 70 uM |

- The fastest run turns over the most nitrogen but nitrifies only
  partially: ~92 % of the oxidised N leaves as NO2 through the boundary.
  The AOB shell takes all the O2 before it reaches NOB and CMX in the
  colony interior (see `../heatmap/comparison_at_stop.png`).
- The same NO3 collapse runs through the whole high-NH4, high-O2 corner
  (NH4 >= 150 uM, O2 >= 70 uM), earlier at higher O2.
- Complete nitrification only at low to moderate NH4 (<= ~22 uM), at much
  lower rates. The most NO3 by T* comes from NH4 150 uM, O2 250 uM.

## Development rates

Net specific growth rate d ln M / dt of each species [1/day], the mean rate
ln(M(T*) / M(0)) / T*, the total biomass growth rate [ug per mm depth per
day], and the colony footprint (area where the total volume fraction
exceeds 0.001) at the last snapshot before T* (45 units, 47.4 days)
relative to t = 0.

| at T* | fastest run | rest of the map |
|---|---|---|
| AOB rate at T* | +0.043, the highest | +0.034 at O2 175, +0.027 at O2 100, +0.016 at O2 40; -0.017 to -0.023 at low NH4 or O2 = 5 uM |
| AOB mean rate 0..T* | +0.023, the highest | ~0 at NH4 58 uM; -0.04 at the lowest NH4 |
| NOB rate at T* | -0.016 | grows only at NH4 22-150 uM with high O2 (up to +0.011) |
| CMX rate at T* | -0.003 | -0.002 to -0.003 everywhere |
| total biomass growth rate | +0.61, the highest | +0.30 at O2 175, +0.11 at O2 100; negative at O2 <= 15 uM or NH4 <= 8.6 uM |
| footprint / initial | 4.7, the largest | 3.6 at O2 100, 2.8 at O2 40; 1.55 at the lowest corner |

- AOB drives the fastest run: after a lag of ~8 days it grows at
  ~0.035 per day and is still accelerating when the run stops.
- NOB in the fastest run recovers briefly, then declines from ~15 days,
  when the AOB shell cuts off its O2 (the same timing as the NO3 collapse).
- CMX develops the same way in every run: decay only, slowing from about
  -0.007 to -0.0015 per day, never positive.
- Footprint growth is not biomass growth: the bottom-left run spreads to
  1.55x its initial area while every species shrinks, through the
  pressure-driven motility.

### Development over time

`development_over_time.png` shows the same quantities at 10.5, 21.1, 31.6,
42.2 and 50.1 days (footprint at the snapshot times, the last at 47.4 days).

- AOB in the fastest run is the first to grow (+0.015 per day at 10.5 days,
  when every other run still declines) and has the highest rate on the map
  at every time.
- The growing-AOB region spreads from the top-right corner downwards and to
  lower NH4 with time: at 21 days it covers O2 >= 100 uM with NH4 >= 58 uM,
  at 50 days O2 >= 40 uM with NH4 >= 22 uM. At O2 <= 15 uM AOB never grows
  before T*.
- NOB growth appears only from ~32 days and only at NH4 22-150 uM with
  O2 >= 70 uM; the fastest run's NOB declines throughout.
- CMX declines at the same, slowly easing rate everywhere (-0.005 at 10.5
  days to -0.003 at 50 days).
- The footprint of the fastest run grows 1.68 -> 2.35 -> 3.17 -> 4.14 ->
  4.73; the low-NH4 or low-O2 runs spread only to 1.3-1.7 by motility.

## Files

```
render_rates.py            production rates: python render_rates.py [T_max of the sweep, default 150]
render_development.py      development rates: python render_development.py [T_max, default 150]
rates_timeseries.png       NH4 consumption, net NO2 and NO3 production over time, all runs
rates_at_Tstar.png         the same at T* and accumulated to T*, NO3 share of oxidised N
rates_summary.json         all numbers behind rates_at_Tstar.png
development_timeseries.png biomass and net specific growth rate per species over time, all runs
development_at_Tstar.png   growth rates at T* and averaged to T*, footprint, total growth rate
development_summary.json   all numbers behind development_at_Tstar.png
render_development_times.py  python render_development_times.py [T_max, default 150]
development_over_time.png  growth rate per species and footprint at 10.5-50.1 days
development_over_time.json all numbers behind development_over_time.png
```

Both scripts read the stored sweep results (not committed; produced by
`../heatmap/run_heatmap.py`, see `../README.md`) and need no new runs. In
the plots the fastest run is the black line or the red box; thin lines are
coloured by O2_BC.
