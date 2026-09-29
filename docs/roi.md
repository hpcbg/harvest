# ROI & investment analysis

The `roi/` package adds a dedicated **long-term return-on-investment** layer on top
of the one-day operational simulator. Where the Operations view compares operational
scenarios for a single simulated day, the **ROI & Investment** view answers a
different question: *over a 5–20 year horizon, do the main HARVEST investments pay
back?* It is available as a dashboard tab, an HTTP endpoint (`POST /api/roi`) and a
headless CLI (`python -m roi`).

> **All ROI outputs are estimates based on user-supplied assumptions.** The financial
> figures shipped in `config.yaml` are **demonstration assumptions** — illustrative
> examples, **not** commercial quotations or verified market prices. Replace them with
> supplier quotations and local operating data before making an investment decision.

### Connected Operations → ROI workflow

ROI is **not a separate simulator**. It is based strictly on the latest successful
Operations run:

```
Configure Operations  →  Run Operations  →  Open ROI  →  Select period & financial
assumptions  →  Run long-term ROI analysis
```

- The ROI **Run** button is disabled until Operations has been run successfully. Until
  then the ROI page shows: *"Run the Operations simulation first. ROI uses the fleet,
  energy system, tasks, seed, and scenarios from the latest successful Operations run."*
- A successful `/simulate` returns an **`operations_run_id`**; the server keeps that
  run's normalised request + merged config in memory. The `/api/roi` request must carry
  the identifier, and the endpoint rejects a missing / unknown / expired / mismatched id
  with **HTTP 409** and the message *"Run the Operations simulation before running ROI
  analysis."* Operational parameters are always taken from the saved run, never from
  independently editable ROI fields.
- **All scenarios** selected in Operations are propagated automatically and shown
  read-only under *"Scenarios from Operations"*. ROI cannot introduce a scenario that
  was not selected on the Operations page.
- Operational controls (fleet, PV, chargers, tasks, seed, tariffs, scenarios,
  prediction backend, MARL/shedding/dynamic-event config…) come from Operations and are
  displayed read-only in an **Operational basis** card with a *Return to Operations*
  button. They are never duplicated as editable ROI inputs.
- Changing **any** Operations input or scenario selection marks the result **stale**,
  invalidates the ROI source and disables ROI: *"Operations settings have changed. Run
  the Operations simulation again before calculating ROI."*
- The ROI page produces **two separate result types**: a **long-term operational
  comparison** (how every scenario performs over the period) and the **investment
  analysis** (financial return from paired/counterfactual simulations). They are shown
  in distinct sections and never merged into one unexplained result.

### Operational period vs financial horizon

Two independent time concepts, kept deliberately separate in the UI:

| Concept | Meaning | Example |
|---|---|---|
| **Operational data period** | The calendar range actually simulated to measure annual energy, fuel, distance and hours | 2026-01-01 → 2026-12-31 |
| **Financial investment horizon** | The number of years the cash-flow model projects (payback, NPV, IRR) | 10 years |

The operational period is simulated once and **annualised to a 365-day year**; the
annual totals then drive every year of the financial horizon (with escalation,
degradation and scheduled replacements applied per year).

### Period calculation modes

| Mode | Behaviour |
|---|---|
| `exact` | One deterministic simulation per calendar day in the range |
| `representative_month` | N representative days per covered month (default 1), weighted by calendar days — captures seasonal PV variation without simulating all 365 days |
| `auto` | `exact` for periods ≤ `auto_exact_max_days` (default 90), otherwise `representative_month` |

A deterministic seed rule keeps runs reproducible: **`daily_seed = base_seed + YYYYMMDD`**.

Because the default `static` PV backend has no seasonal variation, the period runner
applies a documented **monthly seasonal factor** (peak in June ≈ 1.0, trough in
December ≈ 0.25) to PV *generation* only when that backend is active — so a one-year
analysis reflects real seasonal PV differences instead of repeating the June day
unchanged. Installed capacity (and therefore CAPEX) is unaffected. When a genuinely
seasonal predictor (`stub` / `openmeteo` / `nn`) is active the factor is 1.0 to avoid
double-counting. The result metadata reports which method, how many simulations ran,
how many calendar days are represented, and the seasonal model used.

### Supported investments

| Investment | Baseline it is compared against |
|---|---|
| **Electric fleet vs diesel** | Same workload performed by an equivalent diesel fleet (matched-service) |
| **Fixed farm PV** | An identical simulation with `farm_fixed_peak_kw = 0` |
| **Tractor-roof PV** | An identical simulation with roof `panel_peak_w = 0` |
| **Charging infrastructure** | Included in the electric-fleet principal (CAPEX shown separately) |
| **Backup / islanding** | Grid-tied system with no backup (reliability benefit only) |
| **Combined HARVEST portfolio** | Sequential incremental stages — see below |

Each PV investment uses **paired simulations**: two otherwise-identical runs that
differ only in the asset under study, so savings come from the real simulator
behaviour, priced at **time-step tariffs** (never a single blended average price).

Every investment is computed **per source scenario** and labelled accordingly
(e.g. *"Fixed farm PV — Full smart"*, *"Electric fleet — Smart"*), so ROI is never
silently based on a single hard-coded scenario. Roof PV is only computed for scenarios
that actually include roof panels. Installed farm-PV kWp, roof-panel W and the equipped
tractor count all come from the Operations run — the ROI page supplies only unit costs.
Missing required inputs (e.g. a cleared CAPEX field) yield **"Input required"** rather
than a misleading zero-year payback, and such incomplete investments are excluded from
the portfolio.

### Electric-vs-diesel calculation

A diesel counterfactual performs exactly the electric fleet's workload:

```
travel_litres = travel_distance_km × litres_per_100km / 100
work_litres   = PTO_work_hours     × pto_litres_per_hour
idle_litres   = idle_hours         × idle_litres_per_hour
diesel_fuel_cost = (travel + work + idle) litres × diesel_price_per_litre
```

The electric alternative uses the **actual charging electricity cost from the
simulation** (charger input energy priced at time-step tariffs — not estimated from
total farm energy), plus charger CAPEX/install, electric maintenance and an optional
battery replacement. The comparison is **matched-service** (same generated task set,
compared on cost per completed task); a warning is raised if service levels differ.

### Farm PV & roof PV calculation

```
avoided_grid_cost = baseline_grid_cost − candidate_grid_cost   (paired sims, time-step priced)
```

Farm PV and roof PV are evaluated **independently** with separate ROI results.
Exported energy has zero value unless `export_surplus_enabled` is set (then it uses the
configured feed-in tariff — never net metering unless explicitly configured); PV
yield **degrades** each year while the value of each kWh **escalates** with the
electricity price. Farm PV and roof PV savings are never counted twice.

### Outage & reliability assumptions

Reliability is an **expected-value** model over the simulated 15-minute power
profile. For each step it compares critical demand against available backup supply:

```
expected_outage_hours_per_year = frequency_per_year × average_duration_hours
expected_unserved_energy       = Σ unsupported critical load over expected outages
expected_outage_cost           = unserved_energy × value_of_lost_load
                               + unsupported_hours × downtime_cost
                               + expected task-disruption cost
reliability_benefit            = baseline_outage_cost − candidate_outage_cost
```

> **Physical rule:** Grid-connected PV is assumed to disconnect during an outage and
> therefore provides **no** backup-power benefit unless an islanding-capable inverter
> is enabled. The simulator does **not** model a stationary backup battery, so its
> contribution is clearly labelled an **analytical** backup model (PV availability and
> critical load come from the simulated profile; the fixed battery and V2L headroom
> are analytical). Sources are never mixed silently.

### Combined portfolio — no double-counting

The portfolio uses a **sequential incremental** comparison so overlapping savings are
counted once:

1. Diesel fleet baseline
2. **+** Electric tractors & charging infrastructure (marginal vs diesel)
3. **+** Fixed farm PV (marginal vs stage 2)
4. **+** Tractor-roof PV (marginal vs stage 3)
5. **+** Islanding / backup (reliability benefit)

Each stage counts only its marginal benefit over the previous stage. The combined
portfolio benefit therefore does **not** equal the sum of the independently-calculated
standalone savings, and a warning states so explicitly.

### Financial formulas

`net_capex = equipment + installation + other − grants − subsidies`;
`annual_net_benefit = avoided_operating_cost + revenue + reliability_benefit −
maintenance − recurring_costs`. Year 0 holds the CAPEX. **NPV** discounts every year
plus residual value; **IRR** is found by numerically-safe bisection (returns *not
available* when cash flows don't admit a valid IRR); **simple** and **discounted
payback** interpolate the fractional year the cumulative cash flow turns non-negative;
**ROI** is `(cumulative_undiscounted_net_benefit − net_capex) / net_capex × 100`
(*N/A* when CAPEX ≤ 0). A cash-flow row is returned for every year with baseline /
candidate operating cost, fuel, electricity, maintenance, outage loss, revenue,
replacement, net, discount factor, discounted and cumulative values.

### ROI metrics

| Metric | Meaning |
|---|---|
| **Net CAPEX** | Equipment + installation − grants/subsidies |
| **Annual net benefit** | Operating savings + revenue + reliability − recurring costs (year 1) |
| **Simple ROI** | `(cumulative net benefit − CAPEX) / CAPEX × 100` over the horizon |
| **Simple payback** | First fractional year cumulative undiscounted cash flow ≥ 0 |
| **Discounted payback** | First fractional year cumulative discounted cash flow ≥ 0 |
| **NPV** | Discounted net present value incl. residual value |
| **IRR** | Internal rate of return (bisection; *N/A* if undefined) |
| **Expected outage cost** | Annual expected cost of unserved critical load + downtime |
| **Avoided outage cost** | Reliability benefit = baseline − candidate outage cost |

### Sensitivity analysis

A one-way sensitivity varies diesel price, electricity-price escalation, farm-PV CAPEX,
electric-tractor CAPEX, discount rate, outage frequency and value-of-lost-load by
±`variation_pct` (default 20%), recomputing portfolio NPV / payback / ROI for each.
It **reuses the operational energy totals** (no simulation re-runs) and is shown as a
tornado-style bar chart.

### Dashboard usage

Use the header tabs to switch between **Operations** and **ROI & Investment**. First
configure the farm on the Operations page and **RUN SIMULATION**. That successful run
becomes the source of truth for ROI. Then open the ROI tab:

1. An **Operational basis** card shows the run's read-only configuration (fleet, PV,
   chargers, battery, grid cap, tasks/day, seed, scenarios, one-day results) with a
   *Return to Operations* button. *Scenarios from Operations* are shown as read-only
   badges — there is no independent scenario selector.
2. Set the **analysis period**, period method and **financial horizon** (these are the
   only period controls ROI adds), and adjust the collapsible **financial / investment
   assumption** panels (each field shows its unit; per-tractor/per-charger amounts are
   labelled as such). *Reload from config* re-reads `config.yaml` + `config.local.yaml`;
   *Reset to demo* restores the shipped demonstration values.
3. **RUN ROI ANALYSIS** (disabled until Operations has run). Results show, in separate
   sections: the **long-term operational comparison** (every scenario over the period,
   with charts), the **investment analysis** (per source scenario, with *Input required*
   for cleared fields), the **combined portfolio** (with a scenario selector defaulting
   to the best one-day task completion), a **reliability** panel and a **sensitivity**
   chart. Warnings (demonstration note, distances derived, exact vs representative,
   outage analytical vs simulated, service-level differences, standalone-overlap) are
   always visible. Export **CSV / JSON** — both carry the `operations_run_id` and run
   metadata so a result can be traced back to its exact Operations run.

Changing any Operations control marks ROI **stale** and disables it until Operations is
re-run — the header shows `ROI unavailable` / `ROI stale` / `ROI ready` / `ROI running`
/ `ROI complete` / `ROI error`.

### CLI usage

The headless CLI evaluates every scenario in the config (or a subset via `--scenarios`);
it does not require the dashboard or a saved run:

```bash
python -m roi \
    --config config.yaml \
    --start 2026-01-01 \
    --end 2026-12-31 \
    --period-mode auto \
    --horizon 10 \
    --scenarios smart,full_smart      # optional; default = all config scenarios
```

Generated files (default `outputs/roi/`):

- `roi_summary.csv` — one row per investment (per scenario) plus each portfolio; carries `operations_run_id` + scenario
- `roi_cashflows.csv` — one row per investment per year
- `roi_assumptions.json` — resolved assumptions + `export_meta` (run traceability)
- `roi_report.json` — the full response (meta, operational basis, long-term, investments, portfolios, sensitivity)

### Configuration

The `roi:` section of `config.yaml` holds **demonstration assumptions** and documents
every field (`analysis`, `financial`, `service_value`, `electric_fleet`, `diesel`,
`farm_pv`, `tractor_roof_pv`, `outages`, `sensitivity`). Per-unit costs use explicit key
names (`electric_tractor_purchase_eur_each`, `charger_capex_eur_each`,
`residual_value_eur_each`, `grant_eur_total`, …). Economic assumptions are **never**
hard-coded in Python or JavaScript — the dashboard loads them from the server
(`GET /api/config`), and `config.local.yaml` can override any of them without a git
commit. **Operational** values (fleet, PV, chargers, tasks, seed, tariffs, scenarios)
are **not** in the `roi:` section — ROI always takes them from the Operations run.
`roi.demonstration_assumptions: true` makes the dashboard label the figures as demo.

### Warnings & limitations

- ROI is based strictly on the **latest successful Operations run**; it cannot run
  before Operations, and any Operations change invalidates a prior ROI result.
- ROI outputs are **estimates** based on user-supplied assumptions; the shipped
  demonstration prices are **not** commercial quotations.
- Missing required assumptions (e.g. a cleared CAPEX) show **"Input required"** — not a
  zero-year payback — and are excluded from the portfolio.
- Grid-tied PV provides **no** outage backup unless islanding is enabled.
- Standalone investment results **cannot always be added together** — use the portfolio.
- The stationary backup battery is an **analytical** model (not simulated); its results
  are labelled as such and never mixed with simulated values.
- Work (in-field) distances are **derived** from execution time × configured working
  speed; the dashboard states when distances are derived rather than measured.

### Future work — energy sharing (out of scope)

Energy exchange with neighbouring farms or households — peer-to-peer trading, energy
communities, export coordination — is **explicitly out of scope** for this module and
is noted here only as possible future work. Nothing in `roi/` implements it.

