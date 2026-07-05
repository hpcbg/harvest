"""Unit tests for the pure financial primitives (roi.calculator)."""

import unittest

from roi import calculator as calc


class TestNPV(unittest.TestCase):
    def test_npv_known_cash_flows(self):
        # -1000 now, +500 for three years @ 10 %.
        flows = [-1000, 500, 500, 500]
        expected = -1000 + 500 / 1.1 + 500 / 1.1**2 + 500 / 1.1**3
        self.assertAlmostEqual(calc.npv(10.0, flows), expected, places=6)

    def test_npv_zero_rate_is_sum(self):
        flows = [-1000, 400, 400, 400]
        self.assertAlmostEqual(calc.npv(0.0, flows), 200.0, places=9)


class TestPayback(unittest.TestCase):
    def test_simple_payback_fractional(self):
        # cumulative: -1000, -600, -200, +200 → crosses at 2.5 years.
        flows = [-1000, 400, 400, 400, 400]
        self.assertAlmostEqual(calc.simple_payback(flows), 2.5, places=6)

    def test_discounted_payback_later_than_simple(self):
        flows = [-1000, 400, 400, 400, 400]
        simple = calc.simple_payback(flows)
        disc = calc.discounted_payback(flows, 10.0)
        self.assertIsNotNone(disc)
        self.assertGreater(disc, simple)

    def test_negative_savings_no_payback(self):
        # All cash flows negative → payback never happens → None.
        self.assertIsNone(calc.simple_payback([-1000, -100, -100]))
        self.assertIsNone(calc.discounted_payback([-1000, -100, -100], 5.0))


class TestIRR(unittest.TestCase):
    def test_irr_valid_sequence(self):
        # -1000, +600, +600 → IRR ≈ 13.07 %.
        r = calc.irr([-1000, 600, 600])
        self.assertIsNotNone(r)
        # Verify by construction: NPV at the returned IRR must be ~0.
        self.assertAlmostEqual(calc.npv(r, [-1000, 600, 600]), 0.0, places=2)
        self.assertAlmostEqual(r, 13.07, places=1)

    def test_irr_unavailable_for_all_positive(self):
        self.assertIsNone(calc.irr([1000, 500, 500]))

    def test_irr_unavailable_for_all_negative(self):
        self.assertIsNone(calc.irr([-1000, -500, -500]))


class TestROIAndSanitize(unittest.TestCase):
    def test_zero_capex_handled(self):
        self.assertIsNone(calc.roi_pct(0.0, 500.0))
        self.assertIsNone(calc.roi_pct(-10.0, 500.0))

    def test_roi_positive(self):
        # capex 1000, cumulative benefit 1500 → ROI 50 %.
        self.assertAlmostEqual(calc.roi_pct(1000.0, 1500.0), 50.0, places=6)

    def test_sanitize_nan_inf(self):
        self.assertIsNone(calc.sanitize(float("nan")))
        self.assertIsNone(calc.sanitize(float("inf")))
        self.assertEqual(calc.sanitize(3.14159, 2), 3.14)


class TestEscalationDegradation(unittest.TestCase):
    def test_escalation_year1_unchanged(self):
        # Year 1 is the base year → no escalation.
        self.assertAlmostEqual(calc.escalate(100.0, 5.0, 1), 100.0, places=9)
        self.assertAlmostEqual(calc.escalate(100.0, 5.0, 2), 105.0, places=9)

    def test_degradation_reduces_over_years(self):
        y1 = calc.degrade(100.0, 0.5, 1)
        y5 = calc.degrade(100.0, 0.5, 5)
        self.assertAlmostEqual(y1, 100.0, places=9)
        self.assertLess(y5, y1)


if __name__ == "__main__":
    unittest.main()
