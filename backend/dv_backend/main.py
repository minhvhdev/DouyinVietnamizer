import os

import uvicorn

from .local_env import load_repo_dotenv


def main() -> None:
    load_repo_dotenv()
    reload = os.environ.get("DV_RELOAD", "1") == "1"
    uvicorn.run(
        "dv_backend.api:create_app",
        factory=True,
        host="127.0.0.1",
        port=int(os.environ.get("DV_BACKEND_PORT", "8765")),
        log_level="info",
        reload=reload,
    )


if __name__ == "__main__":
    main()
