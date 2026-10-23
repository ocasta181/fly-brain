import sys
from pathlib import Path

import pytest

pytest.register_assert_rewrite("state_assertions")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
