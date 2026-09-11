import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Exercise the user's MLX-LM checkout when present, without modifying it.
MLX_LM = Path(os.environ.get("MLX_LM_PATH", ROOT.parent.parent / "mlx-lm"))
if MLX_LM.exists():
    sys.path.insert(0, str(MLX_LM))


@pytest.fixture(scope="session")
def reference():
    path = (
        Path(os.environ.get("TTT_TORCH_PATH", ROOT.parent / "ttt-lm-pytorch"))
        / "ttt.py"
    )
    if not path.exists():
        pytest.fail("Set TTT_TORCH_PATH to the supplied ttt-lm-pytorch checkout")
    spec = importlib.util.spec_from_file_location("ttt_torch_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Transformers 5 changed only the tied-weight metadata format. Keep the
    # reference equations and forward/backward code untouched.
    import transformers

    if int(transformers.__version__.split(".")[0]) >= 5:
        module.TTTForCausalLM._tied_weights_keys = {
            "lm_head.weight": "model.embed_tokens.weight"
        }
    return module
