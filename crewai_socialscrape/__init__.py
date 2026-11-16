"""CrewAI tools for SocialScrape: social media data for agents and crews.

Tools for the common jobs (a profile, its posts, one post, comments, a video
transcript, a search) across platforms, and a tool that runs any endpoint by id.
"""

from importlib import metadata

from crewai_socialscrape._catalog import Endpoint, endpoints
from crewai_socialscrape.toolkit import get_socialscrape_tools
from crewai_socialscrape.tools import (
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

try:
    __version__ = metadata.version("crewai-socialscrape")
except metadata.PackageNotFoundError:
    __version__ = ""

__all__ = [
    "Endpoint",
    "SocialScrapeCallEndpointTool",
    "SocialScrapeCommentsTool",
    "SocialScrapeEndpointsTool",
    "SocialScrapePostTool",
    "SocialScrapePostsTool",
    "SocialScrapeProfileTool",
    "SocialScrapeSearchTool",
    "SocialScrapeTool",
    "SocialScrapeTranscriptTool",
    "__version__",
    "endpoints",
    "get_socialscrape_tools",
]
