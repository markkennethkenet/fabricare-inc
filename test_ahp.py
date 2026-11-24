import json
import tempfile
import unittest
from pathlib import Path

import ahp


def judgments_from(weights):
    """Perfectly consistent answers for the given weights: a_ij = w_i / w_j."""
    return {pair: weights[pair[0]] / weights[pair[1]] for pair in ahp.PAIRS}


class AhpTests(unittest.TestCase):
    def test_parses_saaty_values_and_rejects_others(self):
        self.assertEqual(ahp.parse_judgment("3"), 3.0)
        self.assertAlmostEqual(ahp.parse_judgment(" 1/5 "), 0.2)
        for bad in ("10", "1/10", "0", "x"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ahp.parse_judgment(bad)

    def test_consistent_answers_give_back_their_weights_with_zero_cr(self):
        weights = {"hole": 0.4, "stain": 0.3, "pilling": 0.2, "needlemark": 0.1}
        matrix = ahp.comparison_matrix(judgments_from(weights))
        found, lambda_max = ahp.priorities(matrix)
        for name, value in zip(ahp.CRITERIA, found):
            self.assertAlmostEqual(value, weights[name], places=6)
        self.assertAlmostEqual(lambda_max, 4.0, places=6)
        self.assertAlmostEqual(ahp.consistency_ratio(matrix), 0.0, places=6)

    def test_worked_example_matches_a_direct_eigenvalue_solution(self):
        # Hole 3x stain, 5x pilling, 7x needle marks; stain 3x pilling, 5x needle marks; pilling 3x needle marks.
        # numpy.linalg.eig on the same matrix gives 0.565, 0.262, 0.118, 0.055 and lambda_max 4.117.
        answers = {("hole", "stain"): 3, ("hole", "pilling"): 5, ("hole", "needlemark"): 7,
                   ("stain", "pilling"): 3, ("stain", "needlemark"): 5, ("pilling", "needlemark"): 3}
        matrix = ahp.comparison_matrix(answers)
        weights, lambda_max = ahp.priorities(matrix)
        self.assertEqual([round(w, 3) for w in weights], [0.565, 0.262, 0.118, 0.055])
        self.assertAlmostEqual(lambda_max, 4.117, places=3)
        self.assertAlmostEqual(ahp.consistency_ratio(matrix), 0.0433, places=4)

    def test_inconsistent_expert_is_left_out_of_the_group(self):
        good = judgments_from({"hole": 0.5, "stain": 0.25, "pilling": 0.15, "needlemark": 0.1})
        # Says hole > stain > pilling but also pilling 9x hole: contradictory.
        bad = {**good, ("hole", "pilling"): 1 / 9}
        result = ahp.analyse([("A", good), ("B", bad)])
        self.assertEqual([e["accepted"] for e in result["experts"]], [True, False])
        self.assertEqual(result["accepted_count"], 1)
        self.assertAlmostEqual(sum(result["group_weights"].values()), 1.0)
        self.assertAlmostEqual(result["group_weights"]["hole"], 0.5, places=6)

    def test_group_uses_the_geometric_mean_of_judgments(self):
        first = {pair: 1.0 for pair in ahp.PAIRS}
        second = {pair: 1.0 for pair in ahp.PAIRS}
        first[("hole", "stain")], second[("hole", "stain")] = 4.0, 1.0
        combined = ahp.group_matrix([ahp.comparison_matrix(first), ahp.comparison_matrix(second)])
        self.assertAlmostEqual(combined[0][1], 2.0)
        self.assertAlmostEqual(combined[1][0], 0.5)

    def test_reads_a_csv_and_writes_the_weights_to_the_config(self):
        with tempfile.TemporaryDirectory() as folder:
            responses = Path(folder) / "responses.csv"
            header = "expert," + ",".join(ahp.column(pair) for pair in ahp.PAIRS)
            responses.write_text(header + "\nLaundry owner,3,5,7,3,5,3\nTailor,1/1,3,5,3,5,3\n", encoding="utf-8")
            config = Path(folder) / "config.json"
            config.write_text(json.dumps({"quality": {"minimum_blur_score": 60}}), encoding="utf-8")
            result = ahp.analyse(ahp.read_responses(responses))
            ahp.write_config(result, config)
            saved = json.loads(config.read_text(encoding="utf-8"))
        self.assertEqual(saved["quality"], {"minimum_blur_score": 60})
        self.assertIn("2 expert(s)", saved["health_weights"]["source"])
        self.assertAlmostEqual(sum(saved["health_weights"]["weights"].values()), 1.0, places=3)
        self.assertIn("Group", ahp.report(result))


if __name__ == "__main__":
    unittest.main()
