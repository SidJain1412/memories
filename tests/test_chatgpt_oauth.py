"""Tests for chatgpt_oauth module — PKCE and token exchange helpers."""
import json
import hashlib
import base64
import pytest
from unittest.mock import patch, MagicMock


class TestPKCE:
    """Test PKCE code_verifier and code_challenge generation."""

    def test_generate_code_verifier_length(self):
        from chatgpt_oauth import generate_code_verifier
        verifier = generate_code_verifier()
        assert 43 <= len(verifier) <= 128

    def test_generate_code_verifier_is_base64url(self):
        from chatgpt_oauth import generate_code_verifier
        verifier = generate_code_verifier()
        assert "+" not in verifier
        assert "/" not in verifier
        assert "=" not in verifier

    def test_generate_code_verifier_unique(self):
        from chatgpt_oauth import generate_code_verifier
        v1 = generate_code_verifier()
        v2 = generate_code_verifier()
        assert v1 != v2

    def test_compute_code_challenge(self):
        from chatgpt_oauth import generate_code_verifier, compute_code_challenge
        verifier = generate_code_verifier()
        challenge = compute_code_challenge(verifier)
        digest = hashlib.sha256(verifier.encode("ascii")).digest()
        expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        assert challenge == expected

    def test_generate_state(self):
        from chatgpt_oauth import generate_state
        state = generate_state()
        assert len(state) >= 32
        assert "+" not in state
        assert "/" not in state


class TestBuildAuthorizeUrl:
    """Test authorize URL construction."""

    def test_includes_required_params(self):
        from chatgpt_oauth import build_authorize_url
        url = build_authorize_url(
            client_id="test-cid",
            redirect_uri="http://localhost:9876/callback",
            code_challenge="challenge123",
            state="state456",
        )
        assert "auth.openai.com/oauth/authorize" in url
        assert "client_id=test-cid" in url
        assert "code_challenge=challenge123" in url
        assert "state=state456" in url
        assert "code_challenge_method=S256" in url
        assert "response_type=code" in url
        assert "scope=" in url


class TestTokenExchange:
    """Test OAuth token exchange functions (mocked HTTP)."""

    def _mock_urlopen(self, response_data: dict):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_exchange_code_for_tokens(self):
        from chatgpt_oauth import exchange_code_for_tokens
        tokens_response = {
            "id_token": "fake-id-token",
            "access_token": "fake-access-token",
            "refresh_token": "fake-refresh-token",
            "expires_in": 3600,
        }
        with patch("chatgpt_oauth.urllib.request.urlopen",
                    return_value=self._mock_urlopen(tokens_response)) as mock_url:
            result = exchange_code_for_tokens(
                code="auth-code-123",
                code_verifier="verifier-abc",
                redirect_uri="http://localhost:9876/callback",
                client_id="test-client-id",
            )
            assert result["id_token"] == "fake-id-token"
            assert result["refresh_token"] == "fake-refresh-token"
            call_args = mock_url.call_args[0][0]
            assert "auth.openai.com/oauth/token" in call_args.full_url

    def test_exchange_code_sends_correct_params(self):
        from chatgpt_oauth import exchange_code_for_tokens
        with patch("chatgpt_oauth.urllib.request.urlopen",
                    return_value=self._mock_urlopen({"id_token": "t", "refresh_token": "r"})) as mock_url:
            exchange_code_for_tokens(
                code="code-1",
                code_verifier="verifier-1",
                redirect_uri="http://localhost:9876/callback",
                client_id="cid-1",
            )
            request_obj = mock_url.call_args[0][0]
            body = request_obj.data.decode()
            assert "grant_type=authorization_code" in body
            assert "code=code-1" in body
            assert "code_verifier=verifier-1" in body
            assert "client_id=cid-1" in body

    def test_exchange_id_token_for_api_key(self):
        from chatgpt_oauth import exchange_id_token_for_api_key
        with patch("chatgpt_oauth.urllib.request.urlopen",
                    return_value=self._mock_urlopen({"access_token": "sk-fake-key"})):
            api_key = exchange_id_token_for_api_key(
                id_token="fake-id-token",
                client_id="test-client-id",
            )
            assert api_key == "sk-fake-key"

    def test_exchange_api_key_sends_token_exchange_grant(self):
        from chatgpt_oauth import exchange_id_token_for_api_key
        with patch("chatgpt_oauth.urllib.request.urlopen",
                    return_value=self._mock_urlopen({"access_token": "sk-key"})) as mock_url:
            exchange_id_token_for_api_key(id_token="idt", client_id="cid")
            request_obj = mock_url.call_args[0][0]
            body = request_obj.data.decode()
            assert "token-exchange" in body
            assert "requested_token=openai-api-key" in body

    def test_refresh_tokens(self):
        from chatgpt_oauth import refresh_tokens
        with patch("chatgpt_oauth.urllib.request.urlopen",
                    return_value=self._mock_urlopen({
                        "id_token": "new-id",
                        "access_token": "new-access",
                        "refresh_token": "new-refresh",
                        "expires_in": 7200,
                    })):
            result = refresh_tokens(
                refresh_token="old-refresh",
                client_id="cid",
            )
            assert result["id_token"] == "new-id"
            assert result["refresh_token"] == "new-refresh"

    def test_network_error_raises(self):
        from chatgpt_oauth import exchange_code_for_tokens
        with patch("chatgpt_oauth.urllib.request.urlopen",
                    side_effect=Exception("Connection refused")):
            with pytest.raises(Exception, match="Connection refused"):
                exchange_code_for_tokens(
                    code="c", code_verifier="v",
                    redirect_uri="http://localhost:9876/callback",
                    client_id="cid",
                )
