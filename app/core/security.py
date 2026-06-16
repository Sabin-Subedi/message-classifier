"""API-key based authentication dependency."""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status

from app.core.config import Settings, get_settings


async def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    """Validate the X-API-Key header when an API key is configured.

    If `settings.api_key` is empty, authentication is disabled (handy for
    local development). When set, requests without a matching header are
    rejected with 401.
    """

    expected = settings.api_key
    if not expected:
        return

    if not x_api_key or x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
