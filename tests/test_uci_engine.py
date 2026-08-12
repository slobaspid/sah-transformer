import io
import importlib.util

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_uci_search_option_plays_legal(monkeypatch, capsys):
    mod = _load("uci_engine", "scripts/uci_engine.py")
    cmds = ("uci\nisready\nsetoption name Search value 8\n"
            "position startpos moves e2e4 e7e5\ngo wtime 60000 btime 60000\nquit\n")
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(cmds))
    mod.main()
    out = capsys.readouterr().out
    assert "option name Search" in out
    bm = [l for l in out.splitlines() if l.startswith("bestmove ")]
    assert bm and len(bm[0].split()[1]) >= 4

def test_uci_protocol_and_move(monkeypatch, capsys):
    mod = _load("uci_engine", "scripts/uci_engine.py")
    cmds = ("uci\nisready\nsetoption name UCI_Elo value 1800\n"
            "position startpos moves e2e4 e7e5\ngo wtime 60000 btime 60000\nquit\n")
    monkeypatch.setattr("sys.stdin", io.StringIO(cmds))
    mod.main()
    out = capsys.readouterr().out
    assert "uciok" in out
    assert "readyok" in out
    # a bestmove line with a plausible 4-char UCI move
    bm = [l for l in out.splitlines() if l.startswith("bestmove ")]
    assert bm and len(bm[0].split()[1]) >= 4
