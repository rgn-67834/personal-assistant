from contextlib import contextmanager
from contextvars import ContextVar

import anthropic

# Bring-your-own-key: the server holds no Anthropic key of its own.
# Each request carries the visitor's key, and the client built from it lives
# only for that request. A ContextVar keeps concurrent requests from seeing
# each other's client.
_client: ContextVar[anthropic.Anthropic | None] = ContextVar("anthropic_client", default=None)


@contextmanager
def use_client(client):
    token = _client.set(client)
    try:
        yield
    finally:
        _client.reset(token)


def use_api_key(api_key: str):
    return use_client(anthropic.Anthropic(api_key=api_key))


def get_client() -> anthropic.Anthropic:
    client = _client.get()
    if client is None:
        raise RuntimeError("No Anthropic API key for this request.")
    return client
