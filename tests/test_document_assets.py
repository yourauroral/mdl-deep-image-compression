import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_defense_uses_repository_assets_instead_of_ignored_pdfs():
    source = (ROOT / "slides" / "defense.tex").read_text(encoding="utf-8")
    assets = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", source)

    assert assets
    for asset in assets:
        assert Path(asset).suffix.lower() != ".pdf"
        candidates = [ROOT / "slides" / asset, ROOT / "figures" / asset]
        assert any(path.is_file() for path in candidates), asset
