"""The sample files must stay loadable and in step with the code."""

import os
import re
from pathlib import Path

import pytest
import yaml

from jevrag.cli import main
from jevrag.config import Config
from jevrag.envfile import load_env_file, parse_env_file

ROOT = Path(__file__).resolve().parent.parent
PKG_TEMPLATES = ROOT / "jevrag" / "templates"


@pytest.mark.parametrize("root_name,pkg_name", [(".env.example", "env.example"),
                                                ("jevrag.example.yaml", "jevrag.example.yaml")])
def test_repo_root_samples_match_packaged_templates(root_name, pkg_name):
    # the repo-root copies are what people browse; the packaged ones are what `jevrag init` writes
    assert (ROOT / root_name).read_text() == (PKG_TEMPLATES / pkg_name).read_text()


def test_example_yaml_loads_with_every_setting():
    cfg = Config(yaml.safe_load((ROOT / "jevrag.example.yaml").read_text()))
    assert cfg.store["kind"] == "memory" and cfg.embedder_spec
    assert cfg.chunking.boundary.style == "inline"
    assert cfg.retrieve.gate.mode == "on"  # YAML `on` normalised


def test_every_commented_store_block_is_valid():
    text = (ROOT / "jevrag.example.yaml").read_text()
    blocks = re.findall(r"^# store:.*\n((?:#   .*\n)+)", text, re.M)
    assert len(blocks) >= 4
    for b in blocks:
        body = "".join(line[2:] + "\n" for line in b.splitlines())
        spec = yaml.safe_load("store:\n" + body)["store"]
        cfg = Config({"store": spec, "embedder": {"model": "hash:8"}})
        assert cfg.store["kind"] in {"qdrant", "chroma", "pgvector", "pinecone"}


def test_env_example_lists_every_variable_the_code_reads():
    used = set()
    for f in (ROOT / "jevrag").rglob("*.py"):
        used |= set(re.findall(r'os\.environ\.get\(\s*"([A-Z][A-Z0-9_]+)"', f.read_text()))
        used |= set(re.findall(r'"([A-Z][A-Z0-9_]*_(?:API_KEY|TOKEN|URL|HOST))"', f.read_text()))
    documented = set(parse_env_file(ROOT / ".env.example"))
    assert used, "found no environment variables in the code"
    assert used <= documented, f"missing from .env.example: {sorted(used - documented)}"
    assert all(v == "" for v in parse_env_file(ROOT / ".env.example").values()), "sample must ship blank"


def test_env_parser_syntax(tmp_path):
    p = tmp_path / ".env"
    p.write_text(
        "# comment\n\n"
        "A=plain\n"
        "export B=exported\n"
        'C="two words"  # note\n'
        "D='lit $X \\n'\n"
        'E="line\\nbreak"\n'
        "F=value # trailing\n"
        "G=has#hash\n"
        "H=\n"
        "I = spaced\n"
    )
    env = parse_env_file(p)
    assert env == {"A": "plain", "B": "exported", "C": "two words", "D": "lit $X \\n", "E": "line\nbreak",
                   "F": "value", "G": "has#hash", "H": "", "I": "spaced"}


def test_env_parser_reports_bad_line(tmp_path):
    p = tmp_path / ".env"
    p.write_text("OK=1\nnot an assignment\n")
    with pytest.raises(ValueError, match=":2:"):
        parse_env_file(p)


def test_load_env_shell_wins_and_blanks_skipped(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("JR_T_SHELL=from_file\nJR_T_NEW=new\nJR_T_BLANK=\n")
    monkeypatch.setenv("JR_T_SHELL", "from_shell")
    monkeypatch.delenv("JR_T_NEW", raising=False)
    monkeypatch.delenv("JR_T_BLANK", raising=False)
    applied = load_env_file(p)
    assert applied == ["JR_T_NEW"]
    assert os.environ["JR_T_SHELL"] == "from_shell" and os.environ["JR_T_NEW"] == "new"
    assert "JR_T_BLANK" not in os.environ
    assert load_env_file(tmp_path / "missing.env") == []


def test_cli_loads_dotenv_and_init_writes_both(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["init", "--store", "memory", "--embedder", "hash:16"]) == 0
    env = tmp_path / ".env"
    assert env.exists() and parse_env_file(env) == parse_env_file(ROOT / ".env.example")
    if os.name == "posix":
        assert env.stat().st_mode & 0o077 == 0
    # init never overwrites an existing .env, even with --force
    env.write_text("JR_T_KEY=mine\n")
    env.chmod(0o600)
    assert main(["init", "--force", "--full"]) == 0
    assert env.read_text() == "JR_T_KEY=mine\n"
    assert "kept existing" in capsys.readouterr().out
    assert Config.load("jevrag.yaml").chunking.boundary.window_units == 24  # --full wrote the whole sample

    monkeypatch.delenv("JR_T_KEY", raising=False)
    main(["inspect", "--collection", "none"])  # any command loads .env first
    assert os.environ.get("JR_T_KEY") == "mine"

    monkeypatch.delenv("JR_T_KEY", raising=False)
    main(["--no-env-file", "inspect", "--collection", "none"])
    assert "JR_T_KEY" not in os.environ

    assert main(["--env-file", "nope.env", "inspect", "--collection", "none"]) == 1
    assert "nope.env not found" in capsys.readouterr().err


def test_cli_warns_on_world_readable_env(tmp_path, monkeypatch, capsys):
    if os.name != "posix":
        pytest.skip("POSIX permissions")
    monkeypatch.chdir(tmp_path)
    main(["init", "--embedder", "hash:16"])
    (tmp_path / ".env").chmod(0o644)
    main(["inspect", "--collection", "none"])
    assert "chmod 600" in capsys.readouterr().err
