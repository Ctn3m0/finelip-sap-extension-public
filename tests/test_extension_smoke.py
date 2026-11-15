import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ExtensionSmokeTests(unittest.TestCase):
    def test_profiles_and_sap_sources_are_present(self):
        fair = json.loads(
            (ROOT / "profiles" / "fair-ablation" / "profile.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(fair["shared_training"]["scheduler_mode"], "optimizer_update")
        self.assertTrue((ROOT / "src" / "finelip_sap_ext" / "sap.py").is_file())
        self.assertTrue((ROOT / "src" / "finelip_sap_ext" / "subcaptions.py").is_file())


if __name__ == "__main__":
    unittest.main()

