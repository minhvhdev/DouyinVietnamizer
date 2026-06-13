import os

import uvicorn

from .api import create_app


def main() -> None:
    uvicorn.run(
        create_app(),
        host="127.0.0.1",
        port=int(os.environ.get("DV_BACKEND_PORT", "8765")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
