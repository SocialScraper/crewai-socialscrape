"""The tools, against a stand-in API that answers with recorded sample responses."""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("crewai")
pytest.importorskip("socialscrape")

from socialscrape import MissingApiKeyError

from crewai_socialscrape import (
    SocialScrapeCallEndpointTool,
    SocialScrapeCommentsTool,
    SocialScrapeEndpointsTool,
    SocialScrapePostsTool,
    SocialScrapePostTool,
    SocialScrapeProfileTool,
    SocialScrapeSearchTool,
    SocialScrapeTranscriptTool,
    endpoints,
)


def test_profile_calls_the_platform_endpoint_with_the_key(api, client):
    api.add("/v1/tiktok/profile", "tiktok.profile")
    out = json.loads(SocialScrapeProfileTool(client=client).run(platform="tiktok", handle="khaby.lame"))

    assert out["data"]["handle"] == "khaby.lame"
    assert out["data"]["stats"]["followers"] == 162967647
    assert (out["credits_charged"], out["credits_remaining"]) == (1, 4999)
    request = api.requests[0]
    assert request.method == "GET" and request.url.path == "/v1/tiktok/profile"
    assert api.params() == {"handle": "khaby.lame", "trim": "true"}
    assert request.headers["x-api-key"] == "ss_test_key"


def test_tool_made_from_a_key_sends_it_and_names_itself(api, monkeypatch):
    import httpx
    from socialscrape import SocialScrape

    made = {}
    real = SocialScrape.__init__

    def init(self, **kwargs):
        made.update(kwargs)
        real(self, **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(api)))

    monkeypatch.setattr(SocialScrape, "__init__", init)
    api.add("/v1/tiktok/profile", "tiktok.profile")
    tool = SocialScrapeProfileTool(api_key="ss_test_other", base_url="https://api.example.test")
    tool.run(platform="tiktok", handle="nasa")

    assert made["api_key"] == "ss_test_other" and made["base_url"] == "https://api.example.test"
    assert api.requests[0].headers["x-api-key"] == "ss_test_other"
    assert api.requests[0].headers["user-agent"].startswith("crewai-socialscrape/")
    assert api.requests[0].url.host == "api.example.test"
    assert "ss_test_other" not in repr(tool)
    assert "ss_test_other" not in tool.model_dump_json()


def test_key_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("SOCIALSCRAPE_API_KEY", "ss_env_key")
    assert SocialScrapeProfileTool().api_key.get_secret_value() == "ss_env_key"


def test_no_key_is_an_error_at_the_first_call_not_at_construction():
    tool = SocialScrapeProfileTool()
    with pytest.raises(MissingApiKeyError):
        tool.run(platform="tiktok", handle="nasa")


def test_an_identifier_the_platform_does_not_take_is_answered_without_a_request(api, client):
    out = SocialScrapeProfileTool(client=client).run(platform="facebook", handle="nasa")
    assert out == "Error: facebook.profile looks things up by url; got handle"
    assert api.requests == []


def test_posts_follow_the_cursor(api, client):
    api.add("/v1/youtube/channel/videos", "youtube.channel.videos")
    tool = SocialScrapePostsTool(client=client)
    first = json.loads(tool.run(platform="youtube", handle="@NASA"))
    assert first["data"]["has_more"] is True and len(first["data"]["items"]) == 2

    tool.run(platform="youtube", handle="@NASA", cursor=first["data"]["cursor"])
    assert api.params() == {"handle": "@NASA", "cursor": first["data"]["cursor"], "trim": "true"}


def test_post_reads_the_platform_from_the_link(api, client):
    api.add("/v1/youtube/video", "youtube.video")
    out = json.loads(SocialScrapePostTool(client=client).run(url="https://youtu.be/dQw4w9WgXcQ"))
    assert out["data"]["object"] == "post"
    assert api.requests[0].url.path == "/v1/youtube/video"
    assert api.params()["url"] == "https://youtu.be/dQw4w9WgXcQ"

    out = SocialScrapePostTool(client=client).run(url="https://example.com/p/1")
    assert out.startswith("Error: no platform recognised in 'https://example.com/p/1'")
    assert len(api.requests) == 1


def test_post_by_id_needs_the_platform(api, client):
    api.add("/v1/tiktok/video", "youtube.video")
    SocialScrapePostTool(client=client).run(platform="tiktok", id="7300000000000000000")
    assert api.params() == {"id": "7300000000000000000", "trim": "true"}


def test_comments(api, client):
    api.add("/v1/youtube/video/comments", "youtube.video.comments")
    out = json.loads(SocialScrapeCommentsTool(client=client).run(url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"))
    assert [c["object"] for c in out["data"]["items"]] == ["comment", "comment"]
    assert api.requests[0].url.path == "/v1/youtube/video/comments"


def test_transcript_leaves_segments_out_unless_asked(api, client):
    api.add("/v1/transcript", "transcript")
    tool = SocialScrapeTranscriptTool(client=client)
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    out = json.loads(tool.run(url=url))
    assert out["data"]["text"].startswith("Example transcript text.")
    assert "segments" not in out["data"] and out["data"]["source"] == "captions"
    assert api.params() == {"url": url, "trim": "true"}

    out = json.loads(tool.run(url=url, include_segments=True, allow_long_transcription=True, language="en"))
    assert len(out["data"]["segments"]) == 2
    assert api.params() == {"url": url, "language": "en", "allow_long_transcription": "true", "trim": "true"}


def test_search(api, client):
    api.add("/v1/reddit/search", "reddit.search")
    out = json.loads(SocialScrapeSearchTool(client=client).run(platform="reddit", query="keyboards"))
    assert len(out["data"]["items"]) == 2
    assert api.params() == {"query": "keyboards", "trim": "true"}
    with pytest.raises(ValueError, match="arguments validation failed"):  # outside the tool's list: no request
        SocialScrapeSearchTool(client=client).run(platform="github", query="x")
    assert len(api.requests) == 1


def test_an_api_error_is_the_answer(api, client):
    api.fail("/v1/tiktok/profile", "not_found", "No TikTok account with that handle.", 404)
    out = SocialScrapeProfileTool(client=client).run(platform="tiktok", handle="no-such-account")
    assert out == "Error: not_found: No TikTok account with that handle. | HTTP 404 | request req_test"


def test_cached_results_and_tool_settings(api, client, sample):
    cached = sample("tiktok.profile")
    cached.update(cached=True, credits_charged=0)
    api.add("/v1/tiktok/profile", cached)
    tool = SocialScrapeProfileTool(client=client, max_age_days=7, trim=False)
    out = json.loads(tool.run(platform="tiktok", handle="khaby.lame"))
    assert out["cached"] is True and out["credits_charged"] == 0
    assert api.params() == {"handle": "khaby.lame", "max_age_days": "7.0"}


def test_call_endpoint_runs_any_endpoint_by_id(api, client):
    api.add("/v1/github/user/repositories", "github.user.repositories")
    tool = SocialScrapeCallEndpointTool(client=client)
    out = json.loads(tool.run(endpoint="github.user.repositories", params={"handle": "torvalds"}))
    assert out["data"]["items"][0]["object"] == "repo"
    assert api.requests[0].url.path == "/v1/github/user/repositories"
    assert api.params() == {"handle": "torvalds", "trim": "true"}

    tool.run(endpoint="github.user.repositories", params={"handle": "torvalds", "trim": False})
    assert api.params()["trim"] == "false"


def test_call_endpoint_checks_before_it_spends_a_request(api, client):
    tool = SocialScrapeCallEndpointTool(client=client)
    assert tool.run(endpoint="github.nope", params={}).startswith("Error: unknown endpoint id 'github.nope'")
    out = tool.run(endpoint="github.user.repositories", params={"user": "torvalds"})
    assert out.startswith("Error: github.user.repositories does not take user. It takes: handle | url (one needed)")
    assert tool.run(endpoint="reddit.search").startswith("Error: reddit.search needs query")
    assert api.requests == []


def test_call_endpoint_only_reaches_endpoints_in_the_catalogue(api, client):
    out = SocialScrapeCallEndpointTool(client=client).run(endpoint="../../v1/account/keys", params={})
    assert out.startswith("Error: unknown endpoint id")
    assert api.requests == []


def test_account_endpoints_get_no_data_parameters(api, client):
    api.add("/v1/account/credits", "account.credits")
    out = json.loads(SocialScrapeCallEndpointTool(client=client).run(endpoint="account.credits"))
    assert out["data"]["credits_remaining"] == 5000
    assert api.params() == {}


def test_list_endpoints_makes_no_request_and_needs_no_key():
    tool = SocialScrapeEndpointsTool()

    everything = tool.run()
    assert everything.startswith(f"{len(endpoints())} endpoints, listed by platform key.")
    assert "tiktok (TikTok): tiktok.profile, " in everything

    platform = tool.run(platform="tiktok")
    assert "tiktok.profile.videos: " in platform and "Takes: handle | id (one needed)" in platform
    assert tool.run(platform="x").startswith("twitter.")
    assert tool.run(platform="myspace").startswith("Error: no platform 'myspace'")

    detail = tool.run(endpoint="youtube.video.comments")
    assert detail.startswith("youtube.video.comments: YouTube video comments")
    assert "- sort (string; one of top, latest; default top): " in detail
    assert "One of url, id is needed." in detail
    assert "Also accepted: trim, raw, max_age_days." in detail
    assert 'Example params: {"id": "dQw4w9WgXcQ", "sort": "latest"}' in detail

    assert "facebook.ad_library.search: " in tool.run(search="ad library")
    assert tool.run(search="zzzz nothing").startswith("No endpoint mentions")


def test_async_calls_use_the_async_client(api, async_client):
    api.add("/v1/tiktok/profile", "tiktok.profile")
    tool = SocialScrapeProfileTool(async_client=async_client)
    out = json.loads(asyncio.run(tool.arun(platform="tiktok", handle="khaby.lame")))
    assert out["data"]["handle"] == "khaby.lame"
    assert api.params() == {"handle": "khaby.lame", "trim": "true"}

    listing = asyncio.run(SocialScrapeEndpointsTool().arun(platform="kick"))
    assert listing.startswith("kick.clip")
