import json

import pytest
from httpx import AsyncClient, MockTransport, Request, Response

from ai_review.clients.gitlab.client import get_gitlab_http_client, GitLabHTTPClient
from ai_review.clients.gitlab.mr.client import GitLabMergeRequestsHTTPClient
from ai_review.clients.gitlab.mr.schema.notes import GitLabUpdateMRNoteRequestSchema


@pytest.mark.usefixtures("gitlab_http_client_config")
def test_get_gitlab_http_client_builds_ok():
    gitlab_http_client = get_gitlab_http_client()

    assert isinstance(gitlab_http_client, GitLabHTTPClient)
    assert isinstance(gitlab_http_client.mr, GitLabMergeRequestsHTTPClient)
    assert isinstance(gitlab_http_client.mr.client, AsyncClient)


@pytest.mark.asyncio
async def test_update_note_api_sends_put_with_expected_url_and_body():
    captured: dict = {}

    def handler(request: Request) -> Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return Response(200, json={"id": 3, "body": "updated"})

    async_client = AsyncClient(base_url="https://gitlab.example.com", transport=MockTransport(handler))
    client = GitLabMergeRequestsHTTPClient(async_client)

    response = await client.update_note_api(
        project_id="1",
        merge_request_id="2",
        note_id="3",
        request=GitLabUpdateMRNoteRequestSchema(body="updated"),
    )

    assert captured["method"] == "PUT"
    assert captured["url"] == "https://gitlab.example.com/api/v4/projects/1/merge_requests/2/notes/3"
    assert captured["body"] == {"body": "updated"}
    assert response.status_code == 200
