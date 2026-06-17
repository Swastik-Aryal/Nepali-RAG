# Lazy access – avoid eagerly importing ``app`` at package level.
# Eager import causes the FastAPI app (and its OpenAPI schema) to be
# created before uvicorn sets up its worker, which breaks the Swagger UI
# rendering of multipart/form-data endpoints (e.g. file-upload shows
# gibberish instead of a file picker).


def __getattr__(name: str):
    if name == "app":
        from nepali_rag.api.main import app  # noqa: F811

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
