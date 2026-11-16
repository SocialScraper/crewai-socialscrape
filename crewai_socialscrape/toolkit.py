"""The SocialScrape tools as one list that shares a client."""

from __future__ import annotations

from typing import Any

from ._client import Session
from .tools import (
    SocialScrapeCallEndpointTool,
    SocialScrapeCommentsTool,
    SocialScrapeEndpointsTool,
    SocialScrapePostsTool,
    SocialScrapePostTool,
    SocialScrapeProfileTool,
    SocialScrapeSearchTool,
    SocialScrapeTool,
    SocialScrapeTranscriptTool,
)

TOOL_CLASSES: tuple[type[SocialScrapeTool], ...] = (
    SocialScrapeProfileTool,
    SocialScrapePostsTool,
    SocialScrapePostTool,
    SocialScrapeCommentsTool,
    SocialScrapeTranscriptTool,
    SocialScrapeSearchTool,
    SocialScrapeEndpointsTool,
    SocialScrapeCallEndpointTool,
)


def get_socialscrape_tools(
    api_key: str | None = None,
    *,
    base_url: str | None = None,
    max_age_days: float | None = None,
    trim: bool = True,
    client: Any = None,
    async_client: Any = None,
) -> list[SocialScrapeTool]:
    """Every SocialScrape tool, in the order an agent tends to need them, sharing one API client.

    The key comes from `api_key`, or from SOCIALSCRAPE_API_KEY when it is left out.

    Example:
        .. code-block:: python

            from crewai import Agent
            from crewai_socialscrape import get_socialscrape_tools

            analyst = Agent(
                role="Social media analyst",
                goal="Answer questions about creators and their audiences",
                backstory="You check the data before you answer.",
                tools=get_socialscrape_tools(),
            )
    """
    session = Session(api_key=api_key, base_url=base_url, client=client, async_client=async_client)
    return [
        cls(api_key=api_key, base_url=base_url, max_age_days=max_age_days, trim=trim, session=session)
        for cls in TOOL_CLASSES
    ]
