from __future__ import annotations

import uvicorn

from app import config


def main() -> None:
    uvicorn.run("app.main:app", host=config.HOST, port=config.PORT)


if __name__ == "__main__":
    main()
