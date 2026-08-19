"""Small synchronous wrapper around HTTPX's direct ASGI transport.

Starlette's thread-backed TestClient can deadlock on some Python 3.13/AnyIO
combinations.  Running each request in the caller's event loop exercises the
same ASGI application without a background portal thread.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from fastapi import FastAPI


class ASGITestClient:
    def __init__(self, app: FastAPI) -> None:
        self.app = app

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        async def send() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.request(method, url, **kwargs)

        return asyncio.run(send())

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", url, **kwargs)
