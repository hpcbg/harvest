"""Unit tests for roi.investments (diesel, farm PV, roof PV, portfolio)."""

import unittest

from roi import investments as inv
from roi.models import OperationalTotals, ROIAssumptions


def make_assumptions(**overrides):
    roi = {
        "analysis": {"financial_horizon_years": 10},
        "financial": {"discount_rate_pct": 6.0, "electricity_price_escalation_pct": 0.0,
                      "diesel_price_escalation_pct": 0.0, "general_cost_escalation_pct": 0.0,
                      "maintenance_escalation_pct": 0.0},
        "diesel": {"fuel_price_eur_per_litre": 1.5, "travel_litres_per_100km": 25.0,
                   "pto_litres_per_hour": 8.0, "idle_litres_per_hour": 2.0},
        "electric_fleet": {"electric_tractor_purchase_eur": 90000, "diesel_equivalent_purchase_eur": 60000,
                           "charger_capex_eur_each": 4000, "charger_installation_eur_each": 1000},
        "farm_pv": {"capex_eur_per_kwp": 900, "annual_degradation_pct": 0.5},
        "tractor_roof_pv": {"capex_eur_per_tractor": 1200},
    }
    for grp, vals in overrides.items():
        roi.setdefault(grp, {}).update(vals)
    return ROIAssumptions.from_config({"roi": roi})


class TestDieselModel(unittest.TestCase):
    def test_diesel_litres_from_distance_pto_idle(self):
        a = make_assumptions()
        litres = inv.diesel_fuel_litres(transit_distance_km=100.0, pto_hours=10.0,
                                        idle_hours=5.0, a=a)
        # travel = 100 * 25/100 = 25 ; work = 10*8 = 80 ; idle = 5*2 = 10
        self.assertAlmostEqual(litres["travel"], 25.0, places=6)
        self.assertAlmostEqual(litres["work"], 80.0, places=6)
        self.assertAlmostEqual(litres["idle"], 10.0, places=6)
        self.assertAlmostEqual(litres["total"], 115.0, places=6)


class TestFarmPV(unittest.TestCase):
    def _totals(self, grid_cost, pv_gen=1000.0, pv_used=800.0):
        return OperationalTotals(grid_cost_eur=grid_cost,
                                 farm_pv_generated_kwh=pv_gen, farm_pv_used_kwh=pv_used)

    def test_avoided_cost_from_time_step_paired_totals(self):
        # Avoided grid cost = baseline grid cost - candidate grid cost (paired sim diff,
        # each already accumulated at time-step tariffs).
        a = make_assumptions()
        baseline = self._totals(2000.0, pv_gen=0.0, pv_used=0.0)
        candidate = self._totals(1600.0)
        built = inv.farm_pv(baseline, candidate, a, farm_pv_kwp=5.0)
        self.assertAlmostEqual(built.result.metrics["annual_avoided_grid_cost_eur"], 400.0, places=6)

    def test_degradation_reduces_benefit_over_years(self):
        a = make_assumptions()
        built = inv.farm_pv(self._totals(2000.0), self._totals(1600.0), a, farm_pv_kwp=5.0)
        rows = [r for r in built.result.annual_cashflows if r.year in (1, 5)]
        y1 = next(r for r in rows if r.year == 1)
        y5 = next(r for r in rows if r.year == 5)
        # electricity_cost stores -benefit; degraded benefit in year 5 is smaller.
        self.assertLess(abs(y5.electricity_cost), abs(y1.electricity_cost))

    def test_inverter_replacement_in_correct_year(self):
        a = make_assumptions(farm_pv={"capex_eur_per_kwp": 900,
                                      "inverter_replacement_year": 3,
                                      "inverter_replacement_eur": 1500})
        built = inv.farm_pv(self._totals(2000.0), self._totals(1600.0), a, farm_pv_kwp=5.0)
        by_year = {r.year: r for r in built.result.annual_cashflows}
        self.assertAlmostEqual(by_year[3].replacement_cost, 1500.0, places=6)
        self.assertAlmostEqual(by_year[2].replacement_cost, 0.0, places=6)
        self.assertAlmostEqual(by_year[4].replacement_cost, 0.0, places=6)


class TestRoofPV(unittest.TestCase):
    def test_roof_pv_paired_avoided_cost(self):
        a = make_assumptions()
        baseline = OperationalTotals(grid_cost_eur=1000.0, tractor_charge_cost_eur=800.0)
        candidate = OperationalTotals(grid_cost_eur=850.0, tractor_charge_cost_eur=650.0,
                                      tractor_pv_generated_kwh=300.0)
        built = inv.roof_pv(baseline, candidate, a, n_equipped=3, roof_panel_w=650.0)
        self.assertAlmostEqual(built.result.metrics["annual_electricity_cost_avoided_eur"], 150.0, places=6)


class TestNoDoubleCounting(unittest.TestCase):
    def test_baseline_choice_changes_marginal_benefit(self):
        # Farm PV measured over two different baselines yields different benefits;
        # this is exactly why standalone results are not additive and the portfolio
        # must use sequential baselines.
        a = make_assumptions()
        candidate = OperationalTotals(grid_cost_eur=1500.0, farm_pv_generated_kwh=1000.0,
                                      farm_pv_used_kwh=800.0)
        base_high = OperationalTotals(grid_cost_eur=2000.0)   # e.g. no other PV present
        base_low = OperationalTotals(grid_cost_eur=1700.0)    # e.g. roof PV already present
        b1 = inv.farm_pv(base_high, candidate, a, farm_pv_kwp=5.0)
        b2 = inv.farm_pv(base_low, candidate, a, farm_pv_kwp=5.0)
        self.assertNotAlmostEqual(
            b1.result.metrics["annual_avoided_grid_cost_eur"],
            b2.result.metrics["annual_avoided_grid_cost_eur"], places=3)

    def test_portfolio_capex_is_sum_but_benefit_marginal(self):
        a = make_assumptions()
        c = OperationalTotals(grid_cost_eur=1500.0, farm_pv_generated_kwh=1000.0, farm_pv_used_kwh=800.0)
        s_elec = inv.electric_vs_diesel(
            OperationalTotals(grid_cost_eur=1800, tractor_charge_cost_eur=1800,
                              transit_distance_km=1000, pto_hours=100, idle_hours=50,
                              operating_hours=300),
            a, n_tractors=3, n_chargers=2)
        s_farm = inv.farm_pv(OperationalTotals(grid_cost_eur=1800), c, a, farm_pv_kwp=5.0)
        pf = inv.portfolio([s_elec, s_farm], a, ["diesel", "electric_fleet", "farm_pv"])
        self.assertAlmostEqual(pf.net_capex, s_elec.net_capex + s_farm.net_capex, places=6)
        # portfolio yearly net == sum of marginal stage nets (by construction)
        self.assertAlmostEqual(pf.year_nets[0], s_elec.year_nets[0] + s_farm.year_nets[0], places=6)


if __name__ == "__main__":
    unittest.main()
