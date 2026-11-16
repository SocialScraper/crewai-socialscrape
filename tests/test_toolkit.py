"""The tool list: eight tools a crew can be given as they are."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("crewai")
pytest.importorskip("socialscrape")

from crewai.tools import BaseTool

from crewai_socialscrape import SocialScrapeTool, get_socialscrape_tools

NAMES = [
    "socialscrape_get_profile", "socialscrape_list_posts", "socialscrape_get_post", "socialscrape_list_comments",
    "socialscrape_get_transcript", "socialscrape_search", "socialscrape_list_endpoints", "socialscrape_call_endpoint",
]


def test_returns_the_eight_tools_on_one_session():
    tools = get_socialscrape_tools("ss_test_key", max_age_days=1, trim=False)
    assert [t.name for t in tools] == NAMES
    assert all(isinstance(t, BaseTool) and isinstance(t, SocialScrapeTool) for t in tools)
    assert len({id(t.session) for t in tools}) == 1
    assert all(t.max_age_days == 1 and t.trim is False for t in tools)
    assert all(t.api_key.get_secret_value() == "ss_test_key" for t in tools)


def test_tools_call_through_a_client_you_pass(api, client):
    api.add("/v1/tiktok/profile", "tiktok.profile")
    tools = {t.name: t for t in get_socialscrape_tools(client=client)}
    out = json.loads(tools["socialscrape_get_profile"].run(platform="tiktok", handle="khaby.lame"))
    assert out["data"]["name"] == "Khabane lame"


def test_every_tool_has_a_schema_a_model_can_read():
    for tool in get_socialscrape_tools("ss_test_key"):
        assert tool.description and len(tool.description) < 2500
        schema = tool.args_schema.model_json_schema()
        assert schema["type"] == "object"
        assert "ss_test_key" not in json.dumps(schema) + tool.description


def test_platform_arguments_list_the_allowed_platforms():
    tools = {t.name: t for t in get_socialscrape_tools("ss_test_key")}
    platform = tools["socialscrape_get_profile"].args_schema.model_json_schema()["properties"]["platform"]
    assert "tiktok" in platform["enum"] and "reddit" not in platform["enum"]


def test_tools_can_be_given_to_an_agent_and_a_task():
    from crewai import Agent, Task

    tools = get_socialscrape_tools("ss_test_key")
    agent = Agent(role="Analyst", goal="Report on creators", backstory="Checks the data.", tools=tools, llm="gpt-4o")
    task = Task(description="Profile nasa on TikTok", expected_output="A short summary", agent=agent)
    assert [t.name for t in agent.tools] == NAMES
    assert task.agent is agent
