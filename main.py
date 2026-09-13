"""Source-checkout launcher; importing this module performs no application startup."""


def main():
    import os
    import sys
    from pathlib import Path

    if getattr(sys, "frozen", False):
        os.environ["LOGFIRE_PYDANTIC_RECORD"] = "off"
    source = str(Path(__file__).resolve().parent / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from redlotus.agent_core.entrypoint import main as run

    run()


if __name__ == "__main__":
    main()
