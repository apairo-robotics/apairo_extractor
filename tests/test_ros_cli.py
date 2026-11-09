"""``apairo ros``: the package's commands under its own name.

``apairo ros extract`` is ``apairo extractor``: same parser, same output.
"""
from __future__ import annotations

import pytest

from apairo_ros import __version__
from apairo_ros.cli import main as extractor_main
from apairo_ros.ros_cli import main

from conftest import write_tiny_bag


def _run(fn, argv) -> int:
    try:
        result = fn(argv)
    except SystemExit as exc:
        return exc.code or 0
    return result or 0


def test_no_command_lists_the_commands(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "usage: apairo ros <command>" in out
    assert "extract" in out


def test_version(capsys):
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"apairo-ros {__version__}"


def test_an_unknown_command_is_a_usage_error(capsys):
    assert main(["nope"]) == 2
    assert "unknown command 'nope'" in capsys.readouterr().err


def test_extract_help_names_the_command(capsys):
    assert _run(main, ["extract", "--help"]) == 0
    assert capsys.readouterr().out.startswith("usage: apairo ros extract")


def test_extract_is_the_extractor(tmp_path, fmt, capsys):
    for name in ("a", "b"):
        write_tiny_bag(tmp_path / (f"{name}.bag" if fmt == "ros1" else name), fmt)
    argv = ["-i", str(tmp_path), "--list"]

    assert _run(extractor_main, argv) == 0
    expected = capsys.readouterr().out
    assert _run(main, ["extract", *argv]) == 0
    assert capsys.readouterr().out == expected


@pytest.mark.parametrize("prog", ["apairo-ros", "apairo ros"])
def test_the_program_name_follows_the_entry_point(prog, capsys):
    main([], prog=prog)
    assert capsys.readouterr().out.startswith(f"usage: {prog} <command>")
