"""Unit tests for roi.reliability (expected-value outage model)."""

import unittest

from roi.models import OutageAssumptions
from roi.reliability import analyze_reliability


def flat_profile(critical_kw=1.0, pv_kw=2.0, n=96):
    return [{"critical_load_kw": critical_kw, "total_pv_kw": pv_kw} for _ in range(n)]


class TestReliability(unittest.TestCase):
    def test_pv_benefit_zero_when_islanding_disabled(self):
        # PV present, no battery/V2L, islanding OFF → candidate == baseline → 0 benefit.
        o = OutageAssumptions(enabled=True, frequency_per_year=10, average_duration_hours=2,
                              value_of_lost_load_eur_per_kwh=5, islanding_enabled=False)
        res = analyze_reliability(flat_profile(1.0, 5.0), o, n_tractors=3,
                                  tractor_battery_kwh=44.8, v2l_max_discharge_kw=6.6, has_pv=True)
        self.assertAlmostEqual(res["avoided_outage_cost_eur"], 0.0, places=6)

    def test_pv_benefit_positive_when_islanding_enabled(self):
        o = OutageAssumptions(enabled=True, frequency_per_year=10, average_duration_hours=2,
                              value_of_lost_load_eur_per_kwh=5, islanding_enabled=True)
        res = analyze_reliability(flat_profile(1.0, 5.0), o, n_tractors=3,
                                  tractor_battery_kwh=44.8, v2l_max_discharge_kw=6.6, has_pv=True)
        self.assertGreater(res["avoided_outage_cost_eur"], 0.0)

    def test_expected_value_no_backup_full_unserved(self):
        # No PV, no islanding, no backup: every kWh of critical load is unserved.
        o = OutageAssumptions(enabled=True, frequency_per_year=8, average_duration_hours=3,
                              value_of_lost_load_eur_per_kwh=6, islanding_enabled=False)
        res = analyze_reliability(flat_profile(2.0, 0.0), o, n_tractors=0,
                                  tractor_battery_kwh=0.0, v2l_max_discharge_kw=0.0, has_pv=False)
        # expected unserved energy = crit(2 kW) * duration(3h) * freq(8) = 48 kWh
        self.assertAlmostEqual(res["expected_unserved_energy_kwh"], 48.0, places=6)
        self.assertAlmostEqual(res["expected_outage_hours_per_year"], 24.0, places=6)
        # cost = 48 kWh * 6 €/kWh = 288 (no downtime/disruption cost configured)
        self.assertAlmostEqual(res["expected_annual_outage_cost_eur"], 288.0, places=6)

    def test_battery_labelled_analytical(self):
        o = OutageAssumptions(enabled=True, frequency_per_year=5, average_duration_hours=2,
                              islanding_enabled=True, backup_battery_kwh=20.0, backup_max_power_kw=5.0)
        res = analyze_reliability(flat_profile(1.0, 0.0), o, 3, 44.8, 6.6, has_pv=False)
        self.assertEqual(res["sources"]["stationary_battery"], "analytical")


if __name__ == "__main__":
    unittest.main()
