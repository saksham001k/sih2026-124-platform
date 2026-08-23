import ast
from pathlib import Path


def test_deprecated_container_width_is_not_used() -> None:
    tree = ast.parse(Path("dashboard.py").read_text(encoding="utf-8"))
    deprecated_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and any(keyword.arg == "use_container_width" for keyword in node.keywords)
    ]

    assert deprecated_calls == []
