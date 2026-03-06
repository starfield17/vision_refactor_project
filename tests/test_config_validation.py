from __future__ import annotations

import unittest

from share.config.schema import DEFAULT_CONFIG, deep_merge_dict, validate_config


class ConfigValidationTests(unittest.TestCase):
    def test_llm_mode_accepts_api_key_env_name(self) -> None:
        cfg = deep_merge_dict(
            DEFAULT_CONFIG,
            {
                "class_map": {"names": ["person"], "id_map": {"person": 0}},
                "data": {"yolo_dataset_dir": "/tmp/yolo-dataset"},
                "autolabel": {
                    "mode": "llm",
                    "llm": {
                        "base_url": "https://example.com/v1",
                        "model": "test-model",
                        "api_key_env_name": "VISION_LLM_API_KEY",
                        "prompt": "return json",
                    },
                    "model": {"onnx_model": "/tmp/model.onnx", "backend": "yolo"},
                },
            },
        )

        validated = validate_config(cfg)
        self.assertEqual(validated["autolabel"]["llm"]["api_key_env_name"], "VISION_LLM_API_KEY")
