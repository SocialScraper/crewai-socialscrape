"""CrewAI tools for the SocialScrape API.

One tool per common job (a profile, a profile's posts, one post, its comments,
a video transcript, a search), each across platforms, plus two for everything
else: `socialscrape_list_endpoints` shows what exists and
`socialscrape_call_endpoint` runs any endpoint by its id.
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field, SecretStr, model_validator
from socialscrape import APIError, Response, SocialScrapeConnectionError, SocialScrapeTimeoutError

from ._catalog import (
    IDENTIFIERS,
    ROLES,
    TRANSCRIPT,
    Endpoint,
    common_params,
    endpoints,
    get_endpoint,
    platforms,
    route,
)
from ._client import Session

# What a tool answers with instead of raising: the agent reads it and tries again.
_FAILURES = (ValueError, APIError, SocialScrapeConnectionError, SocialScrapeTimeoutError)


def _titles(role: str) -> str:
    """'YouTube, TikTok, ...': the platforms of a role by display name."""
    return ", ".join(endpoints()[endpoint_id].title for endpoint_id in ROLES[role].values())


def _lookups(role: str) -> str:
    """'youtube: handle, url or id; tiktok: handle or id; ...': what each platform is looked up by."""
    rows = []
    for platform, endpoint_id in ROLES[role].items():
        names = [name for name in IDENTIFIERS if name in endpoints()[endpoint_id].names]
        rows.append(f"{platform}: {' or '.join(names)}")
    return "; ".join(rows)


# The platform argument of each cross-platform tool is an enum of the platforms it covers, so a model sees
# the allowed values in the tool's schema.
ProfilePlatform = Literal[tuple(sorted(ROLES["profile"]))]  # type: ignore[valid-type]
PostsPlatform = Literal[tuple(sorted(ROLES["posts"]))]  # type: ignore[valid-type]
PostPlatform = Literal[tuple(sorted(ROLES["post"]))]  # type: ignore[valid-type]
CommentsPlatform = Literal[tuple(sorted(ROLES["comments"]))]  # type: ignore[valid-type]
SearchPlatform = Literal[tuple(sorted(ROLES["search"]))]  # type: ignore[valid-type]

_PLATFORM = "Platform key."
_PLATFORM_OPTIONAL = "Platform key. Leave it out when you pass a url: the platform is read from the link."
_CURSOR = "The `cursor` of the previous answer, to get the next page. Leave it out for the first page."


class ProfileInput(BaseModel):
    """Arguments of socialscrape_get_profile."""

    platform: ProfilePlatform = Field(description=_PLATFORM)
    handle: str | None = Field(default=None, description="Username or handle, with or without the leading @.")
    url: str | None = Field(default=None, description="Full URL of the profile, channel or page.")
    id: str | None = Field(default=None, description="The platform's own id of the account, where it has one.")


class PostsInput(BaseModel):
    """Arguments of socialscrape_list_posts."""

    platform: PostsPlatform = Field(description=_PLATFORM)
    handle: str | None = Field(default=None, description="Username or handle, with or without the leading @.")
    url: str | None = Field(default=None, description="Full URL of the profile, channel or page.")
    id: str | None = Field(default=None, description="The platform's own id of the account, where it has one.")
    cursor: str | None = Field(default=None, description=_CURSOR)


class PostInput(BaseModel):
    """Arguments of socialscrape_get_post."""

    url: str | None = Field(default=None, description="Full URL of the post, video, tweet, pin or clip.")
    platform: PostPlatform | None = Field(default=None, description=_PLATFORM_OPTIONAL)
    id: str | None = Field(default=None, description="The post's id, on platforms that look a post up by id. "
                                                     "Needs platform.")


class CommentsInput(BaseModel):
    """Arguments of socialscrape_list_comments."""

    url: str | None = Field(default=None, description="Full URL of the post or video.")
    platform: CommentsPlatform | None = Field(default=None, description=_PLATFORM_OPTIONAL)
    id: str | None = Field(default=None, description="The post's id, on platforms that look a post up by id. "
                                                     "Needs platform.")
    cursor: str | None = Field(default=None, description=_CURSOR)


class TranscriptInput(BaseModel):
    """Arguments of socialscrape_get_transcript."""

    url: str = Field(description="Full URL of the video.")
    language: str | None = Field(default=None, description="Preferred caption language as a BCP 47 code, "
                                                           "for example en or pt-BR.")
    allow_long_transcription: bool = Field(
        default=False,
        description="Allow AI transcription of audio longer than 2 minutes, at 5 credits per started minute of "
                    "the whole video. Without it a longer video with no captions answers an error that names "
                    "the cost, which is free.")
    include_segments: bool = Field(default=False, description="Also return the timed segments (start, end, text). "
                                                              "The full text is always returned.")


class SearchInput(BaseModel):
    """Arguments of socialscrape_search."""

    platform: SearchPlatform = Field(description=_PLATFORM)
    query: str = Field(description="Search terms.")
    cursor: str | None = Field(default=None, description=_CURSOR)


class EndpointsInput(BaseModel):
    """Arguments of socialscrape_list_endpoints."""

    platform: str | None = Field(default=None, description="Platform key, for example tiktok: lists its endpoints "
                                                           "with what each takes.")
    endpoint: str | None = Field(default=None, description="Endpoint id, for example tiktok.profile.videos: its "
                                                           "full description and every parameter.")
    search: str | None = Field(default=None, description="Words to look for in endpoint ids and descriptions, "
                                                         "for example 'ad library' or 'followers'.")


class CallEndpointInput(BaseModel):
    """Arguments of socialscrape_call_endpoint."""

    endpoint: str = Field(description="Endpoint id from socialscrape_list_endpoints, for example "
                                      "instagram.profile.reels.")
    params: dict[str, Any] = Field(default_factory=dict, description="The endpoint's parameters by name, for "
                                                                     'example {"handle": "nasa"}.')


def _key_from_env() -> SecretStr | None:
    value = os.environ.get("SOCIALSCRAPE_API_KEY")
    return SecretStr(value) if value else None


def _failure(error: Exception) -> str:
    """What an agent is told when a call fails: the error code and message, which it can act on."""
    return f"Error: {error}"


class SocialScrapeTool(BaseTool):
    """Base of the SocialScrape tools: the API key, the client and the shape of an answer.

    Setup:
        .. code-block:: bash

            pip install -U crewai-socialscrape
            export SOCIALSCRAPE_API_KEY="ss_live_..."

    A tool answers with a JSON string: ``data`` (the endpoint's object, or a
    list page with ``items``, ``cursor`` and ``has_more``), ``credits_charged``
    and ``credits_remaining``. A request the API refuses is free, and comes
    back as the tool's answer (``Error: `` and the error code and message) so
    an agent can correct itself. A missing API key raises.
    """

    api_key: SecretStr | None = Field(default_factory=_key_from_env, repr=False)
    """SocialScrape API key. Read from SOCIALSCRAPE_API_KEY when left out."""
    base_url: str | None = None
    """API base URL override."""
    max_age_days: float | None = None
    """Accept a cached result up to this many days old (0 to 30), which costs 0 credits."""
    trim: bool = True
    """Ask for the smaller response, without null fields and the `extra` object."""
    client: Any = Field(default=None, exclude=True, repr=False)
    """A `socialscrape.SocialScrape` you configured yourself, used instead of making one."""
    async_client: Any = Field(default=None, exclude=True, repr=False)
    """A `socialscrape.AsyncSocialScrape` for async calls, used instead of making one per call."""
    session: Any = Field(default=None, exclude=True, repr=False)
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(name="SOCIALSCRAPE_API_KEY", description="API key for SocialScrape", required=True),
        ]
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["socialscrape-sdk"])

    @model_validator(mode="after")
    def _make_session(self) -> SocialScrapeTool:
        if self.session is None:
            self.session = Session(api_key=self.api_key.get_secret_value() if self.api_key else None,
                                   base_url=self.base_url, client=self.client, async_client=self.async_client)
        return self

    def _plan(self, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        """The endpoint id and parameters for these tool arguments."""
        raise NotImplementedError

    def _shape(self, data: Any, **kwargs: Any) -> Any:
        return data

    def _query(self, endpoint_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Add the tool's own trim and max_age_days; a value the caller passed wins."""
        if get_endpoint(endpoint_id).common:
            params = {"trim": True if self.trim else None, "max_age_days": self.max_age_days, **params}
        return params

    def _answer(self, resp: Response[Any], **kwargs: Any) -> str:
        out: dict[str, Any] = {"data": self._shape(resp.data, **kwargs), "credits_charged": resp.credits_charged,
                               "credits_remaining": resp.credits_remaining}
        if resp.cached:
            out["cached"] = True
        return json.dumps(out, ensure_ascii=False)

    def _run(self, **kwargs: Any) -> str:
        try:
            endpoint_id, params = self._plan(**kwargs)
            resp = self.session.call(endpoint_id, self._query(endpoint_id, params))
        except _FAILURES as e:
            return _failure(e)
        return self._answer(resp, **kwargs)

    async def _arun(self, **kwargs: Any) -> str:
        try:
            endpoint_id, params = self._plan(**kwargs)
            resp = await self.session.acall(endpoint_id, self._query(endpoint_id, params))
        except _FAILURES as e:
            return _failure(e)
        return self._answer(resp, **kwargs)


class SocialScrapeProfileTool(SocialScrapeTool):
    """Get a profile, channel or artist from a social platform.

    Example:
        .. code-block:: python

            from crewai_socialscrape import SocialScrapeProfileTool

            tool = SocialScrapeProfileTool()
            tool.run(platform="tiktok", handle="nasa")
    """

    name: str = "socialscrape_get_profile"
    description: str = (
        "Get a profile from a social platform: name, bio, links, verification and follower counts. "
        f"Platforms: {_titles('profile')}. Pass the platform and one identifier. What each platform is looked "
        f"up by: {_lookups('profile')}. Costs 1 credit; a lookup that fails is free."
    )
    args_schema: type[BaseModel] = ProfileInput

    def _plan(self, platform: str, handle: str | None = None, url: str | None = None, id: str | None = None,
              **_: Any) -> tuple[str, dict[str, Any]]:
        ep, ident = route("profile", platform, handle=handle, url=url, id=id)
        return ep.id, ident


class SocialScrapePostsTool(SocialScrapeTool):
    """List the latest posts or videos of a profile, channel or page.

    Example:
        .. code-block:: python

            from crewai_socialscrape import SocialScrapePostsTool

            tool = SocialScrapePostsTool()
            tool.run(platform="youtube", handle="@NASA")
    """

    name: str = "socialscrape_list_posts"
    description: str = (
        "List the latest posts or videos of a profile, channel or page, newest first, one page per call: text, "
        f"time, stats and media links per post. Platforms: {_titles('posts')}. Pass the platform and one "
        f"identifier. What each platform is looked up by: {_lookups('posts')}. When the answer has "
        "`has_more` true, pass its `cursor` back for the next page. Costs 1 credit per page; a failed call is free."
    )
    args_schema: type[BaseModel] = PostsInput

    def _plan(self, platform: str, handle: str | None = None, url: str | None = None, id: str | None = None,
              cursor: str | None = None, **_: Any) -> tuple[str, dict[str, Any]]:
        ep, ident = route("posts", platform, handle=handle, url=url, id=id)
        return ep.id, {**ident, "cursor": cursor}


class SocialScrapePostTool(SocialScrapeTool):
    """Get one post, video, tweet, pin or clip by its URL.

    Example:
        .. code-block:: python

            from crewai_socialscrape import SocialScrapePostTool

            tool = SocialScrapePostTool()
            tool.run(url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    """

    name: str = "socialscrape_get_post"
    description: str = (
        "Get one post, video, tweet, pin or clip: text, author, time, stats and media links. "
        f"Platforms: {_titles('post')}. Pass its url; the platform is read from the link. Costs 1 credit; "
        "a lookup that fails is free."
    )
    args_schema: type[BaseModel] = PostInput

    def _plan(self, url: str | None = None, platform: str | None = None, id: str | None = None,
              **_: Any) -> tuple[str, dict[str, Any]]:
        ep, ident = route("post", platform, url=url, id=id)
        return ep.id, ident


class SocialScrapeCommentsTool(SocialScrapeTool):
    """List the comments on a post or video.

    Example:
        .. code-block:: python

            from crewai_socialscrape import SocialScrapeCommentsTool

            tool = SocialScrapeCommentsTool()
            tool.run(url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    """

    name: str = "socialscrape_list_comments"
    description: str = (
        "List the comments on a post or video, one page per call: text, author, time and likes per comment. "
        f"Platforms: {_titles('comments')}. Pass the post's url; the platform is read from the link. When the "
        "answer has `has_more` true, pass its `cursor` back for the next page. Costs 1 credit per page; "
        "a failed call is free."
    )
    args_schema: type[BaseModel] = CommentsInput

    def _plan(self, url: str | None = None, platform: str | None = None, id: str | None = None,
              cursor: str | None = None, **_: Any) -> tuple[str, dict[str, Any]]:
        ep, ident = route("comments", platform, url=url, id=id)
        return ep.id, {**ident, "cursor": cursor}


class SocialScrapeTranscriptTool(SocialScrapeTool):
    """Get the transcript of a video from its URL.

    Example:
        .. code-block:: python

            from crewai_socialscrape import SocialScrapeTranscriptTool

            tool = SocialScrapeTranscriptTool()
            tool.run(url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    """

    name: str = "socialscrape_get_transcript"
    description: str = (
        "Get the transcript of a video from its URL: the full text, its language and where it came from "
        f"(captions or AI transcription of the audio). {endpoints()[TRANSCRIPT].description.split(' Same price')[0]} "
        f"Price: {endpoints()[TRANSCRIPT].credits}. A failed call is free."
    )
    args_schema: type[BaseModel] = TranscriptInput

    def _plan(self, url: str, language: str | None = None, allow_long_transcription: bool = False,
              **_: Any) -> tuple[str, dict[str, Any]]:
        return TRANSCRIPT, {"url": url, "language": language,
                            "allow_long_transcription": True if allow_long_transcription else None}

    def _shape(self, data: Any, include_segments: bool = False, **_: Any) -> Any:
        if isinstance(data, dict) and not include_segments:
            return {k: v for k, v in data.items() if k != "segments"}
        return data


class SocialScrapeSearchTool(SocialScrapeTool):
    """Search one platform by keyword.

    Example:
        .. code-block:: python

            from crewai_socialscrape import SocialScrapeSearchTool

            tool = SocialScrapeSearchTool()
            tool.run(platform="reddit", query="mechanical keyboards")
    """

    name: str = "socialscrape_search"
    description: str = (
        "Search one platform by keyword, one page per call. Posts and videos on most platforms; tracks, albums "
        f"and artists on Spotify and Apple Music; web results on Google. Platforms: {_titles('search')}. When "
        "the answer has `has_more` true, pass its `cursor` back for the next page. Costs 1 credit per page; a "
        "failed call is free. For a narrower search (users, hashtags, filters) find the endpoint with "
        "socialscrape_list_endpoints and run it with socialscrape_call_endpoint."
    )
    args_schema: type[BaseModel] = SearchInput

    def _plan(self, platform: str, query: str, cursor: str | None = None, **_: Any) -> tuple[str, dict[str, Any]]:
        table = ROLES["search"]
        key = platform.strip().lower().replace("-", "_")
        key = {"x": "twitter"}.get(key, key)
        if key not in table:
            raise ValueError(f"no search for {platform!r}; platforms: {', '.join(sorted(table))}")
        ep = endpoints()[table[key]]
        return ep.id, {"query": query, "cursor": cursor if "cursor" in ep.names else None}


def _line(ep: Endpoint) -> str:
    return f"{ep.id}: {ep.summary}. Takes: {ep.signature()}. Returns {ep.response}. Price: {ep.credits}."


def _detail(ep: Endpoint) -> str:
    rows = [f"{ep.id}: {ep.summary}", ep.description,
            f"Returns {ep.response}" + (", one page per call: pass `cursor` back for the next page." if ep.paginated
                                        else ".") + f" Price: {ep.credits}.",
            "Parameters:" if ep.params else "No parameters."]
    for p in ep.params:
        facts = [p["type"]]
        if p["required"]:
            facts.append("required")
        if p.get("enum"):
            facts.append("one of " + ", ".join(str(v) for v in p["enum"]))
        if "default" in p:
            facts.append(f"default {p['default']}")
        if "minimum" in p or "maximum" in p:
            facts.append(f"{p.get('minimum', '')} to {p.get('maximum', '')}".strip())
        rows.append(f"- {p['name']} ({'; '.join(facts)}): {p['description']}")
    for group in ep.one_of:
        rows.append(f"One of {', '.join(group)} is needed.")
    if ep.common:
        rows.append("Also accepted: " + ", ".join(p["name"] for p in common_params()) + ".")
    if ep.example:
        rows.append("Example params: " + json.dumps(ep.example, ensure_ascii=False))
    return "\n".join(rows)


class SocialScrapeEndpointsTool(SocialScrapeTool):
    """Look up which SocialScrape endpoints exist and what each takes. Makes no request.

    Example:
        .. code-block:: python

            from crewai_socialscrape import SocialScrapeEndpointsTool

            tool = SocialScrapeEndpointsTool()
            tool.run(platform="tiktok")
            tool.run(endpoint="tiktok.profile.videos")
    """

    name: str = "socialscrape_list_endpoints"
    description: str = (
        f"Look up what SocialScrape can fetch: {len(endpoints())} endpoints (profiles, posts, comments, search, "
        "ad libraries, music, shops, transcripts, account usage). Free and instant: it "
        "makes no request. With no arguments it lists every platform with its endpoint ids; with `platform` "
        "that platform's endpoints and what each takes; with `endpoint` one endpoint's full description and "
        "parameters; with `search` the endpoints that mention those words. Use the id with "
        "socialscrape_call_endpoint."
    )
    args_schema: type[BaseModel] = EndpointsInput

    def _lookup(self, platform: str | None = None, endpoint: str | None = None, search: str | None = None) -> str:
        if endpoint:
            return _detail(get_endpoint(endpoint.strip()))
        if platform:
            key = platform.strip().lower().replace("-", "_")
            key = {"x": "twitter"}.get(key, key)
            rows = [ep for ep in endpoints().values() if ep.platform == key]
            if not rows:
                raise ValueError(f"no platform {platform!r}; platforms: {', '.join(platforms())}")
            return "\n".join(_line(ep) for ep in rows)
        if search:
            words = search.lower().split()
            rows = [ep for ep in endpoints().values()
                    if all(w in f"{ep.id} {ep.summary} {ep.description}".lower() for w in words)]
            if not rows:
                return f"No endpoint mentions {search!r}. Call with no arguments for the full list."
            more = f"\n{len(rows) - 30} more: narrow the search." if len(rows) > 30 else ""
            return "\n".join(_line(ep) for ep in rows[:30]) + more
        rows_by_platform: dict[str, list[str]] = {}
        for ep in endpoints().values():
            rows_by_platform.setdefault(ep.platform, []).append(ep.id)
        head = (f"{len(endpoints())} endpoints, listed by platform key. Pass `platform` for what its endpoints "
                "take, or `endpoint` for one endpoint's parameters.")
        return "\n".join([head] + [f"{key} ({platforms()[key]}): {', '.join(ids)}"
                                   for key, ids in rows_by_platform.items()])

    def _run(self, **kwargs: Any) -> str:
        try:
            return self._lookup(kwargs.get("platform"), kwargs.get("endpoint"), kwargs.get("search"))
        except ValueError as e:
            return _failure(e)

    async def _arun(self, **kwargs: Any) -> str:
        return self._run(**kwargs)


class SocialScrapeCallEndpointTool(SocialScrapeTool):
    """Run any SocialScrape data endpoint by its id.

    Example:
        .. code-block:: python

            from crewai_socialscrape import SocialScrapeCallEndpointTool

            tool = SocialScrapeCallEndpointTool()
            tool.run(endpoint="github.user.repositories", params={"handle": "torvalds"})
    """

    name: str = "socialscrape_call_endpoint"
    description: str = (
        f"Run any SocialScrape endpoint by its id ({len(endpoints())} of them, for example youtube.channel.videos "
        "or facebook.ad_library.search), with its parameters by name. Find the id and the parameters with "
        "socialscrape_list_endpoints first. Parameters are checked before the request, so a wrong name is "
        "answered at once and costs nothing. Most endpoints cost 1 credit per call; a failed call is free."
    )
    args_schema: type[BaseModel] = CallEndpointInput

    def _plan(self, endpoint: str, params: dict[str, Any] | None = None, **_: Any) -> tuple[str, dict[str, Any]]:
        return get_endpoint(endpoint.strip()).id, dict(params or {})
