"""In the SocialScrape repository: the shipped catalogue is what the generator writes from the API reference today.

Outside it (the public copy of this package has neither the generator nor the reference) the test skips.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent
GENERATOR = PACKAGE / "tools" / "generate.py"


def _generator():
    if not GENERATOR.exists():
        pytest.skip("not in the SocialScrape repository: no generator")
    spec = importlib.util.spec_from_file_location("crewai_socialscrape_generate", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not module.SOURCE.exists():
        pytest.skip("not in the SocialScrape repository: no API reference")
    return module


def test_shipped_catalogue_is_fresh():
    gen = _generator()
    expected = gen.render(gen.build(json.loads(gen.SOURCE.read_text(encoding="utf-8"))))
    assert gen.TARGET.read_text(encoding="utf-8") == expected, "run sdks/crewai/tools/generate.py"


def test_catalogue_holds_only_plain_get_endpoints():
    gen = _generator()
    reference = {e["id"]: e for e in json.loads(gen.SOURCE.read_text(encoding="utf-8"))}
    shipped = json.loads(gen.TARGET.read_text(encoding="utf-8"))["endpoints"]
    assert shipped
    for row in shipped:
        source = reference[row["id"]]
        assert source["method"] == "GET" and source["group"] not in gen.SKIP_GROUPS
        assert all(p["location"] == "query" for p in source["params"])
        assert source["status"] != "stub"
