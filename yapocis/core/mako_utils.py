from pathlib import Path
from mako.template import Template
def render_template(path: str | Path, **vars):
    path = Path(path); src = path.read_text()
    return Template(src).render(**vars)
def resolve_template(template_dir: str | Path, name: str, backend: str = "ocl") -> Path:
    tdir = Path(template_dir)
    if backend == "cuda":
        for cand in (f"cuda_{name}.mako", f"{name}.mako"):
            p = tdir / cand
            if p.exists(): return p
        raise FileNotFoundError(f"No CUDA template for {name} in {tdir}")
    p = tdir / f"{name}.mako"
    if not p.exists(): raise FileNotFoundError(f"No OCL template for {name} in {tdir}")
    return p
