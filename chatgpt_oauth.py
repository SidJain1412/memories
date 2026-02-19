"""ChatGPT OAuth2+PKCE helpers for token exchange.

Pure functions for:
- PKCE code_verifier / code_challenge generation
- Authorization code → tokens exchange
- Token exchange: id_token → OpenAI API key
- Refresh token → new tokens

All HTTP via urllib.request (stdlib). Zero external dependencies.
"""
import base64
import hashlib
import json
import secrets
import urllib.error
import urllib.parse
import urllib.request

OPENAI_ISSUER = "https://auth.openai.com"
TOKEN_URL = f"{OPENAI_ISSUER}/oauth/token"


def _base64url(data: bytes) -> str:
    """Base64url-encode without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def generate_code_verifier() -> str:
    """Generate a PKCE code_verifier (64 random bytes → base64url)."""
    return _base64url(secrets.token_bytes(64))


def compute_code_challenge(verifier: str) -> str:
    """Compute S256 code_challenge from verifier."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return _base64url(digest)


def generate_state() -> str:
    """Generate a random OAuth state parameter."""
    return _base64url(secrets.token_bytes(32))


def build_authorize_url(
    client_id: str,
    redirect_uri: str,
    code_challenge: str,
    state: str,
) -> str:
    """Build the OpenAI OAuth authorization URL."""
    params = urllib.parse.urlencode({
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": "openid profile email offline_access",
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "originator": "memories",
    })
    return f"{OPENAI_ISSUER}/oauth/authorize?{params}"


def _post_token(params: dict) -> dict:
    """POST to the OpenAI token endpoint with url-encoded form body."""
    body = urllib.parse.urlencode(params).encode()
    req = urllib.request.Request(
        TOKEN_URL,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Token request failed ({e.code}): {error_body}"
        ) from e


def exchange_code_for_tokens(
    code: str,
    code_verifier: str,
    redirect_uri: str,
    client_id: str,
) -> dict:
    """Exchange authorization code for id_token + refresh_token."""
    return _post_token({
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier,
        "client_id": client_id,
    })


def exchange_id_token_for_api_key(id_token: str, client_id: str) -> str:
    """Exchange id_token for an OpenAI API key via token exchange grant."""
    result = _post_token({
        "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
        "client_id": client_id,
        "subject_token": id_token,
        "subject_token_type": "urn:ietf:params:oauth:token-type:id_token",
        "requested_token": "openai-api-key",
    })
    return result["access_token"]


def refresh_tokens(refresh_token: str, client_id: str) -> dict:
    """Refresh tokens using a refresh_token."""
    return _post_token({
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "scope": "openid profile email",
    })
