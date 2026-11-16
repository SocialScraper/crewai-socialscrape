# crewai-socialscrape

CrewAI tools for [SocialScrape](https://socialscrape.dev/?utm_source=github&utm_medium=integration&utm_campaign=crewai-socialscrape), a social media data API: profiles, posts, comments, search results and video transcripts from YouTube, TikTok, Instagram, X, Facebook, LinkedIn, Reddit and more, as JSON in one response shape, with one API key.

```bash
pip install -U crewai-socialscrape
export SOCIALSCRAPE_API_KEY="ss_live_..."   # get a key: https://socialscrape.dev/app
```

Python 3.11 or newer. The package builds on the official [`socialscrape-sdk`](https://pypi.org/project/socialscrape-sdk/) client, so retries, timeouts and long transcription jobs are handled for you. Guide: [socialscrape.dev/docs/crewai](https://socialscrape.dev/docs/crewai?utm_source=github&utm_medium=integration&utm_campaign=crewai-socialscrape).

## Give an agent the tools

```python
from crewai import Agent, Crew, Task
from crewai_socialscrape import get_socialscrape_tools

analyst = Agent(
    role="Social media analyst",
    goal="Answer questions about creators and what their audiences say",
    backstory="You look at the data before you answer and you say where each number came from.",
    tools=get_socialscrape_tools(),   # reads SOCIALSCRAPE_API_KEY
)

task = Task(
    description=(
        "Compare NASA's TikTok and YouTube channels: followers, the latest posts, "
        "and what people say in the comments of the newest YouTube video."
    ),
    expected_output="A short comparison with the numbers and three points from the comments.",
    agent=analyst,
)

result = Crew(agents=[analyst], tasks=[task]).kickoff()
print(result.raw)
```

`get_socialscrape_tools()` returns eight tools that share one client:

| Tool | What it does | Credits |
|---|---|---|
| `SocialScrapeProfileTool` (`socialscrape_get_profile`) | A profile, channel or artist: name, bio, links, verification, follower counts. 17 platforms. | 1 |
| `SocialScrapePostsTool` (`socialscrape_list_posts`) | The latest posts or videos of a profile, channel or page, one page per call. 14 platforms. | 1 per page |
| `SocialScrapePostTool` (`socialscrape_get_post`) | One post, video, tweet, pin or clip from its URL. 18 platforms. | 1 |
| `SocialScrapeCommentsTool` (`socialscrape_list_comments`) | The comments on a post or video, one page per call. 7 platforms. | 1 per page |
| `SocialScrapeTranscriptTool` (`socialscrape_get_transcript`) | The transcript of a video from its URL. | 1 from captions; see below |
| `SocialScrapeSearchTool` (`socialscrape_search`) | Keyword search on one platform, one page per call. 14 platforms. | 1 per page |
| `SocialScrapeEndpointsTool` (`socialscrape_list_endpoints`) | What else exists: every endpoint id with its parameters and price. Makes no request. | free |
| `SocialScrapeCallEndpointTool` (`socialscrape_call_endpoint`) | Runs any of the 193 endpoints by id: ad libraries, followers, hashtags, playlists, music, shops and the rest. | per endpoint, mostly 1 |

The platform is an argument (`platform="tiktok", handle="nasa"`), and for a post, its comments or a transcript it is read from the link. Tools that list things return a `cursor` to pass back for the next page.

## Use one tool directly

```python
from crewai_socialscrape import SocialScrapeProfileTool, SocialScrapeTranscriptTool

profile = SocialScrapeProfileTool()            # reads SOCIALSCRAPE_API_KEY
profile.run(platform="tiktok", handle="nasa")
# '{"data": {"object": "profile", "platform": "tiktok", "handle": "nasa", "name": "NASA", ...,
#   "stats": {"followers": ..., "following": ..., "posts": ..., "likes": ...}},
#   "credits_charged": 1, "credits_remaining": 4999}'

SocialScrapeTranscriptTool().run(url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
# '{"data": {"object": "transcript", "language": "en", "source": "captions", "text": "..."}, ...}'
```

Every tool answers with a JSON string: `data` (the object, or a list page with `items`, `cursor` and `has_more`), `credits_charged` and `credits_remaining`. The transcript tool leaves out the timed segments unless you pass `include_segments=True`.

Any endpoint, by id:

```python
from crewai_socialscrape import SocialScrapeCallEndpointTool, SocialScrapeEndpointsTool

SocialScrapeEndpointsTool().run(search="ad library")        # which endpoints, what they take
SocialScrapeCallEndpointTool().run(
    endpoint="facebook.ad_library.search",
    params={"query": "running shoes"},
)
```

Parameters are checked against the endpoint before the request is made, so a misspelt name is answered at once and costs nothing. Only endpoints in the shipped catalogue can be called.

### Errors

A request the API refuses (no such account, a private post, a video with no transcript) is free. The tool returns the error as its answer, for example `Error: not_found: No TikTok account with that handle. | HTTP 404 | request req_...`, so an agent can read it and try something else. A missing API key raises `MissingApiKeyError` at the first call.

### Options

Each tool, and `get_socialscrape_tools()`, takes:

- `api_key`: the SocialScrape key. Left out, it is read from `SOCIALSCRAPE_API_KEY`.
- `max_age_days`: accept a cached result up to this many days old (0 to 30). A cached result costs 0 credits.
- `trim` (default `True`): ask for the smaller response, without null fields and the `extra` object, which saves context.
- `client` / `async_client`: a `socialscrape.SocialScrape` or `AsyncSocialScrape` you configured yourself (timeouts, retries, your own `httpx` client).
- `base_url`: API base URL override.

The tools work with `run` and `arun`.

## Billing

Most requests cost 1 credit: a profile, a post, a page of a list or of search results, a transcript from captions. AI transcription of a video with no captions costs 1 credit up to 2 minutes of audio; longer audio is transcribed only when you pass `allow_long_transcription=True`, at 5 credits per started minute of the whole video. A failed request and a cached repeat cost 0. A new account starts with 100 free credits, no card needed. See [pricing](https://socialscrape.dev/pricing?utm_source=github&utm_medium=integration&utm_campaign=crewai-socialscrape).

Because an agent decides how many calls to make, set a limit on a tool with CrewAI's `max_usage_count` if you want a hard cap on a run.

## Tests

```bash
pip install -e ".[test]"
pytest
```

The tests run offline against sample responses from the API reference.

## License

MIT
