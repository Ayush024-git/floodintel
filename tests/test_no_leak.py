import pathlib

def test_src_never_touches_eval():
    bad = ["eval/", "emsr", "EMSR", "unosat", "UNOSAT"]
    for f in pathlib.Path("src").rglob("*.py"):
        text = f.read_text()
        for b in bad:
            assert b not in text, f"LEAK RISK: {b} in {f}"