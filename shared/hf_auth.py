"""
hf_auth.py — HuggingFace token loading and verification

get_token()   -> str  # returns HF_TOKEN, raises if missing
get_username() -> str  # verified username, cached after first call
load_token()  -> str  # idempotent, loads from platform secret store
verify_token(token: str) -> str  # calls HF whoami, returns username
"""

from __future__ import annotations

import os

# Module-level cache
_verified_username: str | None = None
_token_loaded: bool = False


def load_token() -> str:
    """
    Idempotent token loader. Reads HF_TOKEN from the platform-native
    secret store and sets the environment variable.

    Platform-specific sources:
      Kaggle  -> kaggle_secrets.UserSecretsClient().get_secret("HF_TOKEN")
      Colab   -> google.colab.userdata.get("HF_TOKEN")
      Local   -> os.environ["HF_TOKEN"] (raises if missing)
    """
    global _token_loaded

    if _token_loaded:
        return os.environ.get("HF_TOKEN", "")

    token: str

    # --- Kaggle ---
    try:
        from kaggle_secrets import UserSecretsClient

        client = UserSecretsClient()
        token = client.get_secret("HF_TOKEN")
        print("[hf_auth] Token loaded from Kaggle Secrets")
    except ImportError:
        pass
    else:
        os.environ["HF_TOKEN"] = token
        _token_loaded = True
        return token

    # --- Colab ---
    try:
        from google.colab import userdata

        token = userdata.get("HF_TOKEN")
        print("[hf_auth] Token loaded from Colab userdata")
    except ImportError:
        pass
    except Exception as exc:
        raise RuntimeError(f"[hf_auth] Colab userdata error: {exc}") from exc
    else:
        os.environ["HF_TOKEN"] = token
        _token_loaded = True
        return token

    # --- Local ---
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError(
            "[hf_auth] HF_TOKEN not set. Set the HF_TOKEN environment variable "
            "(e.g., export HF_TOKEN=hf_xxxx or .env file)."
        )
    print("[hf_auth] Token loaded from environment")
    _token_loaded = True
    return token


def verify_token(token: str) -> str:
    """
    Verify token by calling HF Hub whoami. Returns the authenticated username.
    Raises on failure.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    me = api.whoami()
    username: str = me["name"]
    print(f"[hf_auth] Token verified for user: {username}")
    return username


def get_username() -> str:
    """
    Returns the verified HF username, caching it after the first call.
    Loads token if not yet loaded.
    """
    global _verified_username

    if _verified_username is None:
        token = load_token()
        _verified_username = verify_token(token)

    return _verified_username


def get_token() -> str:
    """
    Returns HF_TOKEN from the environment. Loads it first if necessary.
    Raises if no token is available.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        token = load_token()
    return token