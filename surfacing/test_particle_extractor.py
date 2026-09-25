import unittest
from unittest.mock import patch

import numpy as np
from scipy import ndimage

from surfacing.particle_extractor import ParticleExtractor


class ParticleExtractorConnectivityTests(unittest.TestCase):
    @staticmethod
    def make_extractor() -> ParticleExtractor:
        volume = np.zeros((5, 5, 5), dtype=np.uint8)
        volume[2, 2, 2] = 10
        volume[2, 2, 3] = 3
        volume[3, 3, 4] = 3
        extractor = ParticleExtractor(volume, neighborhood_size=5)
        extractor.set_voxel_location((2, 2, 2), search_window=1)
        extractor._noise_estimate = 1.0
        return extractor

    def test_particle_region_excludes_corner_only_attachment(self):
        region = self.make_extractor().particle_region

        _, component_count = ndimage.label(
            region,
            ndimage.generate_binary_structure(3, 1),
        )
        self.assertEqual(component_count, 1)
        self.assertTrue(region[2, 2, 2])
        self.assertTrue(region[2, 2, 3])
        self.assertFalse(region[3, 3, 4])

    def test_cleanup_cannot_add_disconnected_island(self):
        extractor = self.make_extractor()

        def add_island(mask, _footprint):
            result = mask.copy()
            result[0, 0, 0] = True
            return result

        with patch(
            "surfacing.particle_extractor.morphology.closing",
            side_effect=add_island,
        ):
            region = extractor.particle_region

        self.assertFalse(region[0, 0, 0])
        _, component_count = ndimage.label(
            region,
            ndimage.generate_binary_structure(3, 1),
        )
        self.assertEqual(component_count, 1)


if __name__ == "__main__":
    unittest.main()
