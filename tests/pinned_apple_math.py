"""Qualification for tests whose bitwise oracle is pinned Apple ARM64 math."""
import platform
import sys

import pytest


def require_apple_arm64_reference():
    if sys.platform!='darwin' or platform.machine().lower() not in ('arm64','aarch64'):
        pytest.skip('This bitwise reference requires Apple ARM64 libm and its NumPy dispatch')
