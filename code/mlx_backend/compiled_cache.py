"""Process-safe publication for preparation artifacts (never simulation state)."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import tempfile


@contextmanager
def cache_lock(path):
    """Keep the lock inode stable so waiting processes cannot split the lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@contextmanager
def cache_output(path):
    """Yield a private output path on a miss; publish only a successful build."""
    path = Path(path)
    if path.exists():
        yield None
        return
    with cache_lock(path):
        if path.exists():
            yield None
            return
        descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
        os.close(descriptor)
        temporary = Path(temporary)
        try:
            yield temporary
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def atomic_write(path, text):
    """Publish metadata after its dependencies, also safe for repeated writes."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        temporary.write_text(text)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
