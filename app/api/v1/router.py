"""Versioned API router assembly."""

from fastapi import APIRouter

from app.api.v1.endpoints.videos import router as videos_router

router = APIRouter()
router.include_router(videos_router)
