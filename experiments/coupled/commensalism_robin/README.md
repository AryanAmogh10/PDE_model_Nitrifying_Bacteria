# Coupled (time-dependent substrate) -- commensalism, robin regime

Same solver, physics, IC, `eps = 25` (`Lambda = 2.5`) as
`../commensalism_dirichlet/` (see its README); only the regime on
c1 differs. Robin on c1 with `Bi = 5`, `c_inf = 5`, the same pair used in
the slow-fast and coupled-competition Robin folders; c2, c3 stay sealed.

```
python run_coupled_commensalism.py
python render_coupled_commensalism.py
```

## Files

Identical to the dirichlet folder's apart from `REGIME`; outputs are named
`coupled_commensalism_robin_*`.
