from unittest.mock import Mock

from dv_backend import main


def test_main_uses_factory_app_entrypoint(monkeypatch) -> None:
    run = Mock()
    monkeypatch.setattr(main.uvicorn, "run", run)

    main.main()

    assert run.call_args.args[0] == "dv_backend.api:create_app"
    assert run.call_args.kwargs["factory"] is True
    assert run.call_args.kwargs["host"] == "127.0.0.1"
