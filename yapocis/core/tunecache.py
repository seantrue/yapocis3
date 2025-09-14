
from __future__ import annotations
import os, json
from pathlib import Path
from typing import Optional, Tuple, Any, Dict

DEFAULT_PATH = Path(os.environ.get("YAPOCIS_TUNECACHE", Path.home() / ".cache" / "yapocis" / "tuning.json"))

def _ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def load_cache(path: Path = DEFAULT_PATH) -> Dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}

def save_cache(data: Dict[str, Any], path: Path = DEFAULT_PATH) -> None:
    _ensure_dir(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)

def get_device_fingerprint(backend: str = "cuda") -> str:
    if backend == "cuda":
        try:
            import pycuda.driver as cuda  # type: ignore
            # Ensure context is initialized; if pycuda.autoinit not imported, try init
            try:
                _ = cuda.Context.get_current()
            except Exception:
                cuda.init()
            dev = cuda.Device(0)
            name = dev.name()
            cc = dev.compute_capability()
            try:
                mp = dev.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)  # type: ignore[attr-defined]
            except Exception:
                mp = 0
            return f"cuda:{name}:cc{cc[0]}{cc[1]}:mp{mp}"
        except Exception:
            return "cuda:unknown"
    # Fallback / other backends
    return f"{backend}:unknown"

def _key(program: str, kernel: str, backend: str, tag: Optional[str]) -> Tuple[str, str, str, str]:
    return (backend, get_device_fingerprint(backend), program, f"{kernel}:{tag or ''}")

def get_local_size(program: str, kernel: str, backend: str = "cuda", *, tag: Optional[str] = None, path: Path = DEFAULT_PATH) -> Optional[Tuple[int, ...]]:
    data = load_cache(path)
    b, dev, prog, kval = _key(program, kernel, backend, tag)
    return tuple(data.get(b, {}).get(dev, {}).get(prog, {}).get(kval, {}).get("local_size") or ()) or None

def set_local_size(program: str, kernel: str, local_size: Tuple[int, ...], backend: str = "cuda", *, tag: Optional[str] = None, path: Path = DEFAULT_PATH) -> None:
    data = load_cache(path)
    b, dev, prog, kval = _key(program, kernel, backend, tag)
    data.setdefault(b, {}).setdefault(dev, {}).setdefault(prog, {})[kval] = {"local_size": list(map(int, local_size))}
    save_cache(data, path)
