"""Compiled Mako kernel modules are keyed by a digest of their template, not its mtime.

The digest is commonkit's current algorithm (``HASH``/``COMMONKIT_HASH``: md5 or b3).

Mako alone reuses a compiled module unless the template's whole-second mtime is newer, so an
edit in the same second as the last compile ran the old kernel (seen after a git stash pop).
"""

import os

import pytest
from commonkit import digest
from commonkit.testing import assert_isolated
from mako.lookup import TemplateLookup

from yapocis.core.kmgr_base import kernel_template_lookup


def write(path, text, mtime):
    path.write_text(text)
    os.utime(path, (mtime, mtime))


def render(lookup, name="kernel.mako"):
    return lookup.get_template(name).render().strip()


def test_plain_mako_reuses_a_stale_module_when_mtimes_tie(tmp_path):
    # The bug this cache fixes, kept as a control: same mtime, new content, old output.
    templates, modules = tmp_path / "t", tmp_path / "m"
    templates.mkdir()
    write(templates / "kernel.mako", "version 1", 1_700_000_000)
    assert (
        render(TemplateLookup([str(templates)], module_directory=str(modules)))
        == "version 1"
    )
    write(templates / "kernel.mako", "version 2", 1_700_000_000)
    assert (
        render(TemplateLookup([str(templates)], module_directory=str(modules)))
        == "version 1"
    )


def test_edits_with_the_same_mtime_recompile(tmp_path):
    templates, modules = tmp_path / "t", tmp_path / "m"
    templates.mkdir()
    write(templates / "kernel.mako", "version 1", 1_700_000_000)
    assert render(kernel_template_lookup([str(templates)], str(modules))) == "version 1"
    write(templates / "kernel.mako", "version 2", 1_700_000_000)
    assert render(kernel_template_lookup([str(templates)], str(modules))) == "version 2"


def test_included_templates_are_keyed_too(tmp_path):
    templates, modules = tmp_path / "t", tmp_path / "m"
    templates.mkdir()
    write(templates / "kernel.mako", '<%include file="helpers.mako"/>', 1_700_000_000)
    write(templates / "helpers.mako", "helper 1", 1_700_000_000)
    assert render(kernel_template_lookup([str(templates)], str(modules))) == "helper 1"
    write(templates / "helpers.mako", "helper 2", 1_700_000_000)
    assert render(kernel_template_lookup([str(templates)], str(modules))) == "helper 2"


def test_old_compiled_modules_are_removed(tmp_path):
    templates, modules = tmp_path / "t", tmp_path / "m"
    templates.mkdir()
    for version in range(3):
        write(templates / "kernel.mako", f"version {version}", 1_700_000_000)
        render(kernel_template_lookup([str(templates)], str(modules)))
    compiled = sorted(p.name for p in modules.glob("kernel.mako*.py"))
    assert len(compiled) == 1 and compiled[0].startswith("kernel.mako.")


#: The known digests of the template text ``version 1``.
KNOWN = {
    "md5": "db3ec040e20dfc657dab510aeab74759",
    "b3": "42186bbdf69916229202c6dc3f416ba22704a235eb95b474c6bfb196664dcc98",
}


def test_module_name_is_the_current_algorithms_digest(tmp_path):
    templates, modules = tmp_path / "t", tmp_path / "m"
    templates.mkdir()
    write(templates / "kernel.mako", "version 1", 1_700_000_000)
    render(kernel_template_lookup([str(templates)], str(modules)))
    names = [p.name for p in modules.glob("kernel.mako*.py")]
    assert names == [f"kernel.mako.{KNOWN[digest.current()]}.py"]


def test_hash_is_what_was_asked():
    assert_isolated()
    asked = os.environ.get("COMMONKIT_HASH")
    if asked is None:
        pytest.fail("COMMONKIT_HASH is not set")
    assert digest.current() == asked.strip().lower()
