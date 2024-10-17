import pytest
from unittest.mock import MagicMock, patch
from databricks_ai_bridge.genie import Genie, _parse_query_result
import pandas as pd

@pytest.fixture
def mock_workspace_client():
    with patch("databricks_ai_bridge.genie.WorkspaceClient") as MockWorkspaceClient:
        mock_client = MockWorkspaceClient.return_value
        yield mock_client

@pytest.fixture
def genie(mock_workspace_client):
    return Genie(space_id="test_space_id")

def test_start_conversation(genie, mock_workspace_client):
    mock_workspace_client.genie._api.do.return_value = {"conversation_id": "123"}
    response = genie.start_conversation("Hello")
    assert response == {"conversation_id": "123"}
    mock_workspace_client.genie._api.do.assert_called_once_with(
        "POST",
        "/api/2.0/genie/spaces/test_space_id/start-conversation",
        body={"content": "Hello"},
        headers=genie.headers,
    )

def test_create_message(genie, mock_workspace_client):
    mock_workspace_client.genie._api.do.return_value = {"message_id": "456"}
    response = genie.create_message("123", "Hello again")
    assert response == {"message_id": "456"}
    mock_workspace_client.genie._api.do.assert_called_once_with(
        "POST",
        "/api/2.0/genie/spaces/test_space_id/conversations/123/messages",
        body={"content": "Hello again"},
        headers=genie.headers,
    )

def test_poll_for_result_completed(genie, mock_workspace_client):
    mock_workspace_client.genie._api.do.side_effect = [
        {"status": "COMPLETED", "attachments": [{"text": {"content": "Result"}}]},
    ]
    result = genie.poll_for_result("123", "456")
    assert result == "Result"

def test_poll_for_result_executing_query(genie, mock_workspace_client):
    mock_workspace_client.genie._api.do.side_effect = [
        {"status": "EXECUTING_QUERY", "attachments": [{"query": {"query": "SELECT *"}}]},
        {"statement_response": {"status": {"state": "SUCCEEDED"}, "result": {"data_typed_array": [], "manifest": {"schema": {"columns": []}}}}},
    ]
    result = genie.poll_for_result("123", "456")
    assert result == "EMPTY"

def test_ask_question(genie, mock_workspace_client):
    mock_workspace_client.genie._api.do.side_effect = [
        {"conversation_id": "123", "message_id": "456"},
        {"status": "COMPLETED", "attachments": [{"text": {"content": "Answer"}}]},
    ]
    result = genie.ask_question("What is the meaning of life?")
    assert result == "Answer"

def test_parse_query_result():
    resp = {
        "manifest": {
            "schema": {
                "columns": [
                    {"name": "col1", "type_name": "STRING"},
                    {"name": "col2", "type_name": "INT"},
                ]
            }
        },
        "result": {
            "data_typed_array": [
                {"values": [{"str": "value1"}, {"str": "1"}]},
                {"values": [{"str": "value2"}, {"str": "2"}]},
            ]
        }
    }
    expected_df = pd.DataFrame({"col1": ["value1", "value2"], "col2": [1, 2]})
    result = _parse_query_result(resp)
    assert result == expected_df.to_string()