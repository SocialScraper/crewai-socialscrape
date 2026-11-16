"""The shipped catalogue and the routing table agree with each other."""

from __future__ import annotations

import pytest

pytest.importorskip("crewai")
pytest.importorskip("socialscrape")

from crewai_socialscrape import _catalog as catalog
from crewai_socialscrape import endpoints

RETURNS = {
    "profile": lambda r: r == "Profile",
    "posts": lambda r: r == "ListPage[Post]",
    "post": lambda r: r == "Post",
    "comments": lambda r: r == "ListPage[Comment]",
    "search": lambda r: r.startswith("ListPage["),
}


def test_catalogue_is_loaded_and_read_only_endpoints():
    eps = endpoints()
    assert len(eps) > 150
    assert all(ep.path.startswith("/v1/") and "{" not in ep.path for ep in eps.values())
    assert not any(ep.platform in ("monitors", "jobs", "batch") for ep in eps.values())
    assert [p["name"] for p in catalog.common_params()] == ["trim", "raw", "max_age_days"]


@pytest.mark.parametrize("role", sorted(catalog.ROLES))
def test_every_routed_endpoint_exists_and_returns_what_the_tool_says(role):
    for platform, endpoint_id in catalog.ROLES[role].items():
        ep = endpoints()[endpoint_id]
        assert ep.platform == platform, endpoint_id
        assert RETURNS[role](ep.response), f"{endpoint_id} returns {ep.response}"
        assert ep.credits == "1 credit", f"{endpoint_id}: the tool descriptions say 1 credit"
        if role == "search":
            assert "query" in ep.required, endpoint_id
        else:
            assert any(name in ep.names for name in catalog.IDENTIFIERS), endpoint_id
        if role in ("post", "comments"):
            assert ep.hosts, f"{endpoint_id} has no url hosts to recognise a link by"


def test_transcript_endpoint_takes_a_url():
    ep = endpoints()[catalog.TRANSCRIPT]
    assert ep.required == ("url",) and ep.response == "Transcript"


@pytest.mark.parametrize("url, platform", [
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "youtube"),
    ("https://youtu.be/dQw4w9WgXcQ", "youtube"),
    ("https://vm.tiktok.com/ZM123/", "tiktok"),
    ("https://x.com/nasa/status/1", "twitter"),
    ("https://twitter.com/nasa/status/1", "twitter"),
    ("www.instagram.com/p/abc/", "instagram"),
    ("https://old.reddit.com/r/python/comments/abc/x/", "reddit"),
    ("https://example.com/post/1", None),
    ("https://notyoutube.com/watch?v=1", None),
])
def test_platform_is_read_from_the_link(url, platform):
    assert catalog.platform_of_url("post", url) == platform


def test_route_picks_the_identifier_the_endpoint_takes():
    ep, ident = catalog.route("profile", "tiktok", handle="nasa", url="https://www.tiktok.com/@nasa")
    assert (ep.id, ident) == ("tiktok.profile", {"handle": "nasa"})
    ep, ident = catalog.route("profile", "X", handle="nasa")
    assert ep.id == "twitter.profile"
    ep, ident = catalog.route("profile", "apple-music", url="https://music.apple.com/us/artist/x/1")
    assert ep.id == "apple_music.artist"
    with pytest.raises(ValueError, match=r"facebook\.profile looks things up by url; got handle"):
        catalog.route("profile", "facebook", handle="nasa")
    with pytest.raises(ValueError, match="no profile lookup for 'myspace'"):
        catalog.route("profile", "myspace", handle="tom")
    with pytest.raises(ValueError, match="pass a platform"):
        catalog.route("post", None, id="123")
    with pytest.raises(ValueError, match="no platform recognised"):
        catalog.route("post", None, url="https://example.com/1")


def test_params_are_checked_before_any_request():
    ep = endpoints()["youtube.channel.videos"]
    sent = catalog.check_params(ep, {"handle": "NASA", "cursor": None, "trim": True})
    assert sent == {"handle": "NASA", "trim": True}
    with pytest.raises(ValueError, match="does not take user"):
        catalog.check_params(ep, {"user": "NASA"})
    with pytest.raises(ValueError, match="needs one of handle, id, url"):
        catalog.check_params(ep, {"cursor": "abc"})
    with pytest.raises(ValueError, match="needs query"):
        catalog.check_params(endpoints()["reddit.search"], {})
    with pytest.raises(ValueError, match="does not take trim"):
        catalog.check_params(endpoints()["account.credits"], {"trim": True})
    with pytest.raises(ValueError, match=r"unknown endpoint id 'tiktok\.nope'.*tiktok\.profile"):
        catalog.get_endpoint("tiktok.nope")


def test_signature_reads_in_one_line():
    assert endpoints()["tiktok.profile"].signature() == "handle | id (one needed)"
    assert endpoints()["reddit.search"].signature().startswith("query (required), ")
    assert endpoints()["account.credits"].signature() == "none"
