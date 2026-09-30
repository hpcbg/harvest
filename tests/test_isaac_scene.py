"""The Isaac spectator camera's framing and the scene's colour vocabulary.

Pure arithmetic: ``scene.frame_points`` needs no Isaac Sim, GPU or USD, so the
view the WebRTC stream opens on is checked here rather than by eye.
"""
from __future__ import annotations

import math
import unittest

from harvest_integrations.simulators.isaac import scene


def _ndc(point, eye, target, tan_half_h, tan_half_v):
    """Project a world point into the camera's [-1, 1] frame."""
    forward = [t - e for t, e in zip(target, eye)]
    norm = math.sqrt(sum(v * v for v in forward))
    forward = [v / norm for v in forward]
    right = [forward[1], -forward[0], 0.0]
    norm = math.hypot(right[0], right[1])
    right = [v / norm for v in right]
    up = [right[1] * forward[2] - right[2] * forward[1],
          right[2] * forward[0] - right[0] * forward[2],
          right[0] * forward[1] - right[1] * forward[0]]
    rel = [p - e for p, e in zip(point, eye)]
    depth = sum(a * b for a, b in zip(rel, forward))
    return (sum(a * b for a, b in zip(rel, right)) / depth / tan_half_h,
            sum(a * b for a, b in zip(rel, up)) / depth / tan_half_v,
            depth)


class SpectatorCameraTest(unittest.TestCase):
    TAN_H = scene.CAMERA_APERTURE_MM[0] / 2.0 / scene.CAMERA_FOCAL_MM
    TAN_V = scene.CAMERA_APERTURE_MM[1] / 2.0 / scene.CAMERA_FOCAL_MM

    #: The committed farm's depot and a spread of task sites (config.yaml).
    WIDE = ([(x, 50.0, z) for x in (50.0, 60.0, 70.0) for z in (0.0, 3.0)]
            + [(x, 40.0, z) for x in (40.0, 45.0) for z in (0.0, 3.0)]
            + [(x, y, z) for x, y in ((120.0, 80.0), (250.0, 115.0),
                                      (60.0, 150.0), (210.0, 50.0),
                                      (700.0, 420.0))
               for z in (0.0, scene.TASK_BEAM_SIZE[2])])

    def _assert_framed(self, points):
        eye, target = scene.spectator_view(points)
        self.assertGreater(eye[2], 0.0)
        self.assertEqual(target[2], 0.0)
        projected = [_ndc(p, eye, target, self.TAN_H, self.TAN_V)
                     for p in points]
        for x, y, depth in projected:
            self.assertGreater(depth, 0.0)
            self.assertLessEqual(abs(x), 1.0)
            self.assertLessEqual(abs(y), 1.0)
        return projected

    def test_every_entity_is_inside_the_frame(self):
        self._assert_framed(self.WIDE)

    def test_the_fit_is_tight_and_centred(self):
        projected = self._assert_framed(self.WIDE)
        xs = [p[0] for p in projected]
        ys = [p[1] for p in projected]
        limit = 1.0 / scene.CAMERA_MARGIN
        # Something touches the margin: the camera is as close as it can be.
        self.assertAlmostEqual(max(max(map(abs, xs)), max(map(abs, ys))),
                               limit, places=2)
        # Whichever axis has room to spare is centred, not shoved to one side.
        self.assertAlmostEqual(max(xs), -min(xs), places=2)
        self.assertAlmostEqual(max(ys), -min(ys), places=2)

    def test_a_tall_farm_is_viewed_across_its_short_axis(self):
        wide_eye, wide_target = scene.spectator_view(
            [(0.0, 0.0, 0.0), (400.0, 100.0, 0.0)])
        tall_eye, tall_target = scene.spectator_view(
            [(0.0, 0.0, 0.0), (100.0, 400.0, 0.0)])
        # Wide farm: the camera stands south of it and looks north.
        self.assertGreater(wide_target[1] - wide_eye[1],
                           abs(wide_target[0] - wide_eye[0]))
        # Tall farm: it stands west of it and looks east.
        self.assertGreater(tall_target[0] - tall_eye[0],
                           abs(tall_target[1] - tall_eye[1]))

    def test_the_view_is_deterministic(self):
        self.assertEqual(scene.spectator_view(self.WIDE),
                         scene.spectator_view(list(self.WIDE)))

    def test_no_points_still_gives_a_camera(self):
        eye, target = scene.spectator_view([])
        self.assertGreater(eye[2], 0.0)
        self.assertTrue(all(math.isfinite(v) for v in eye + target))


class TaskColourTest(unittest.TestCase):
    def test_every_task_state_has_a_distinct_colour(self):
        self.assertEqual(set(scene.TASK_COLORS),
                         {"pending", "assigned", "active", "completed",
                          "deferred", "missed"})
        self.assertEqual(len(set(scene.TASK_COLORS.values())),
                         len(scene.TASK_COLORS))


if __name__ == "__main__":
    unittest.main()
