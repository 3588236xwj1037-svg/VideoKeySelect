import math
import unittest

import numpy as np

from src.selectors.qatss_v2 import (
    compute_semantic_transition_scores,
    select_indices_from_features,
)


class QatssV2SelectionTests(unittest.TestCase):
    def test_constant_features_are_finite_and_selectable(self):
        features = np.ones((5, 3), dtype=np.float32)
        text = np.ones(3, dtype=np.float32)
        selected, details = select_indices_from_features(
            features,
            text,
            timestamps=np.arange(5, dtype=np.float32),
            top_k=4,
        )

        self.assertEqual(len(selected), 4)
        self.assertEqual(selected, sorted(set(selected)))
        self.assertTrue(all(math.isfinite(value) for value in details["relevance"]))
        self.assertTrue(
            all(math.isfinite(value) for value in details["semantic_transition"])
        )

    def test_top_k_larger_than_frame_count_returns_every_frame_once(self):
        features = np.eye(3, dtype=np.float32)
        selected, _ = select_indices_from_features(
            features,
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            timestamps=np.array([0.0, 1.0, 2.0], dtype=np.float32),
            top_k=10,
        )

        self.assertEqual(selected, [0, 1, 2])

    def test_non_temporal_question_does_not_add_local_context(self):
        features = np.array(
            [[1.0, 0.0], [0.99, 0.1], [0.0, 1.0], [-1.0, 0.0]],
            dtype=np.float32,
        )
        selected, details = select_indices_from_features(
            features,
            np.array([1.0, 0.0], dtype=np.float32),
            timestamps=np.array([0.0, 1.0, 3.0, 5.0], dtype=np.float32),
            top_k=4,
            question="why is the person holding the object",
            question_type="CW",
        )

        self.assertEqual(len(selected), 4)
        self.assertEqual(details["selected_local"], [])
        self.assertFalse(details["is_temporal_question"])

    def test_temporal_question_adds_at_most_one_local_context_frame(self):
        features = np.array(
            [[1.0, 0.0], [0.99, 0.1], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
            dtype=np.float32,
        )
        selected, details = select_indices_from_features(
            features,
            np.array([1.0, 0.0], dtype=np.float32),
            timestamps=np.array([0.0, 1.0, 3.0, 5.0, 7.0], dtype=np.float32),
            top_k=4,
            question="what happens after the person moves",
            question_type="TN",
            local_context_seconds=1.5,
        )

        self.assertEqual(len(selected), 4)
        self.assertLessEqual(len(details["selected_local"]), 1)
        self.assertEqual(details["selected_local"], [1])
        self.assertTrue(details["is_temporal_question"])

    def test_strict_gap_constraint_still_fills_requested_budget(self):
        features = np.eye(3, dtype=np.float32)
        selected, details = select_indices_from_features(
            features,
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            timestamps=np.array([0.0, 0.5, 1.0], dtype=np.float32),
            top_k=3,
            min_gap_seconds=10.0,
        )

        self.assertEqual(selected, [0, 1, 2])
        self.assertEqual(details["min_gap_seconds"], 10.0)

    def test_semantic_transition_shape_and_range(self):
        features = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        scores = compute_semantic_transition_scores(features)

        self.assertEqual(scores.shape, (3,))
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreaterEqual(float(scores.min()), 0.0)
        self.assertLessEqual(float(scores.max()), 1.0)

    def test_result_is_sorted_by_timestamp_not_input_index(self):
        features = np.eye(3, dtype=np.float32)
        selected, _ = select_indices_from_features(
            features,
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            timestamps=np.array([2.0, 0.0, 1.0], dtype=np.float32),
            top_k=3,
        )

        self.assertEqual(selected, [1, 2, 0])


if __name__ == "__main__":
    unittest.main()
