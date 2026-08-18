from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

try:
    from methaneunion import MethaneUnionDataset
except ModuleNotFoundError as exc:  # Optional geospatial/ML dependencies may be absent.
    MethaneUnionDataset = None
    LOADER_IMPORT_ERROR = exc
else:
    LOADER_IMPORT_ERROR = None


FIXTURES = Path(__file__).parent / "fixtures" / "manifests"


@unittest.skipIf(
    MethaneUnionDataset is None,
    f"loader dependencies unavailable: {LOADER_IMPORT_ERROR}",
)
class MethaneUnionDatasetTests(unittest.TestCase):
    def test_metadata_only_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_dir = root / "datasets" / "temporal_split" / "480m_GSD"
            manifest_dir.mkdir(parents=True)
            shutil.copyfile(FIXTURES / "release.csv", manifest_dir / "test.csv")

            dataset = MethaneUnionDataset(
                root=root,
                split_scheme="temporal",
                split="test",
                scale_m=480,
                load_arrays=False,
            )
            sample = dataset[1]

        self.assertEqual(len(dataset), 4)
        self.assertEqual(sample["id"], 2)
        self.assertEqual(sample["available_sensors"], ["S2", "L89"])
        self.assertEqual(sample["loaded_sensors"], ["S2", "L89"])
        self.assertEqual(sample["observations"]["L89"]["paths"]["t0"], "data/e2_l89.tif")

    def test_unsupported_split_scheme_fails(self) -> None:
        with self.assertRaises(ValueError):
            MethaneUnionDataset(
                root="unused",
                split_scheme="random",
                split="test",
                scale_m=480,
                load_arrays=False,
            )


if __name__ == "__main__":
    unittest.main()
