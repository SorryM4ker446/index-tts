"""A missing real weight must never be mistaken for an inference alias."""
import unittest

import torch

from indextts.utils.webui_models import validate_checkpoint


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Module()
        self.model.layer = torch.nn.Linear(3, 2)
        self.model.inference_alias = self.model.layer
        self.checkpoint = {"layer.weight": torch.ones(2, 3), "layer.bias": torch.ones(2)}

    def test_registered_aliases_load_the_same_parameters(self):
        aliases = validate_checkpoint(self.model, self.checkpoint)
        result = self.model.load_state_dict(self.checkpoint, strict=False)
        self.assertEqual(sorted(result.missing_keys), aliases)
        self.assertTrue(torch.equal(self.model.inference_alias.weight, self.checkpoint["layer.weight"]))

    def test_real_missing_weight_is_rejected_without_mutation(self):
        before = self.model.layer.bias.detach().clone()
        del self.checkpoint["layer.bias"]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validate_checkpoint(self.model, self.checkpoint)
        self.assertTrue(torch.equal(before, self.model.layer.bias))

    def test_unknown_key_and_incompatible_shape_are_rejected(self):
        for checkpoint in [dict(self.checkpoint, alien=torch.ones(1)),
                           {**self.checkpoint, "layer.weight": torch.ones(5, 3)}]:
            with self.subTest(keys=list(checkpoint)), self.assertRaisesRegex(ValueError, "Incompatible"):
                validate_checkpoint(self.model, checkpoint)


if __name__ == "__main__":
    unittest.main()
