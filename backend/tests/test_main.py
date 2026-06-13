from unittest.mock import Mock

from fastapi import FastAPI

from dv_backend import main


def test_main_passes_concrete_app_to_uvicorn(monkeypatch) -> None:
    app = FastAPI()
    run = Mock()
    monkeypatch.setattr(main, "create_app", lambda: app)
    monkeypatch.setattr(main.uvicorn, "run", run)

    main.main()

    assert run.call_args.args[0] is app
    assert "factory" not in run.call_args.kwargs
