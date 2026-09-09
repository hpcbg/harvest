"""ROS 2 package checks that run without ROS installed.

The bridge node itself needs rclpy (exercised in the Docker stack via
validate_harvest_stack.sh); here we verify the topic contract module and that
every Python file in the package at least compiles.
"""
import pathlib
import py_compile
import sys
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
PKG = REPO / "ros2_ws" / "src" / "harvest_ros"

sys.path.insert(0, str(PKG))
from harvest_ros import topics  # noqa: E402


class TestTopicContract(unittest.TestCase):
    def test_topic_names_are_absolute_and_namespaced(self):
        names = [
            topics.FLEET_SNAPSHOT, topics.FLEET_COMMAND, topics.FLEET_ACK,
            topics.GRID_DRAW_KW, topics.GRID_PV_KW, topics.GRID_PRICE,
            topics.GRID_TARIFF,
        ]
        self.assertEqual(len(names), len(set(names)), "duplicate topic names")
        for name in names:
            self.assertTrue(name.startswith("/harvest/"), name)
            self.assertNotIn(" ", name)

    def test_tractor_soc_topic(self):
        self.assertEqual(
            topics.tractor_soc("tractor_1"), "/harvest/tractors/tractor_1/soc_pct")

    def test_no_status_leaf_topics(self):
        # Orion-LD's DDS bridge silently drops topics whose final segment is
        # exactly 'status' (WISEPACK finding); keep the contract clean of them
        # in case the DDS northbound is ever enabled.
        for name in vars(topics).values():
            if isinstance(name, str) and name.startswith("/"):
                self.assertFalse(name.endswith("/status"), name)


class TestPackageCompiles(unittest.TestCase):
    def test_all_python_files_compile(self):
        py_files = list(PKG.rglob("*.py"))
        self.assertGreaterEqual(len(py_files), 5)
        for path in py_files:
            py_compile.compile(str(path), doraise=True)


if __name__ == "__main__":
    unittest.main()
