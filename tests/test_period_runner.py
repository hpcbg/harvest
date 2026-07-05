"""Unit tests for roi.period_runner (period resolution, seasonality, seeds)."""

import unittest
from datetime import date

import main
from roi import period_runner as pr
from roi.models import AnalysisConfig


class TestPeriodResolution(unittest.TestCase):
    def test_representative_month_weighting_sums_to_year(self):
        days = pr._representative_days(date(2026, 1, 1), date(2026, 12, 31),
                                       "representative_month", 1)
        self.assertEqual(len(days), 12)
        self.assertAlmostEqual(sum(w for _, w in days), 365.0, places=6)

    def test_representative_days_per_month_configurable(self):
        days = pr._representative_days(date(2026, 6, 1), date(2026, 6, 30),
                                       "representative_month", 3)
        self.assertEqual(len(days), 3)
        self.assertAlmostEqual(sum(w for _, w in days), 30.0, places=6)

    def test_exact_mode_one_entry_per_day(self):
        days = pr._representative_days(date(2026, 3, 1), date(2026, 3, 5), "exact", 1)
        self.assertEqual(len(days), 5)
        self.assertTrue(all(w == 1.0 for _, w in days))

    def test_auto_switches_on_threshold(self):
        a = AnalysisConfig(period_mode="auto", auto_exact_max_days=90)
        m_short, _ = pr.resolve_method(date(2026, 1, 1), date(2026, 2, 1), a)
        m_long, _ = pr.resolve_method(date(2026, 1, 1), date(2026, 12, 31), a)
        self.assertEqual(m_short, "exact")
        self.assertEqual(m_long, "representative_month")


class TestSeasonalFactor(unittest.TestCase):
    def test_summer_higher_than_winter(self):
        self.assertGreater(pr._seasonal_factor(6, True), pr._seasonal_factor(12, True))

    def test_disabled_returns_one(self):
        self.assertEqual(pr._seasonal_factor(6, False), 1.0)
        self.assertEqual(pr._seasonal_factor(1, False), 1.0)


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_result(self):
        cfg = main.load_yaml("config.yaml")
        cfg["task_generation"]["num_tasks"] = 8
        out1 = main.run_simulation(cfg, ["full_smart"], start_date="2026-06-01", seed=123)
        out2 = main.run_simulation(cfg, ["full_smart"], start_date="2026-06-01", seed=123)
        self.assertAlmostEqual(out1["scenarios"][0]["summary"]["total_cost_eur"],
                               out2["scenarios"][0]["summary"]["total_cost_eur"], places=9)


if __name__ == "__main__":
    unittest.main()
