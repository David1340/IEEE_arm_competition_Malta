#!/usr/bin/env python3
"""Build docs/knowledge_graph.md: a map of the code for people and for Claude, so nobody has to search for files.

For every Python file: what it is for (its docstring's first line), what it defines, which project modules it
imports, which modules import it, and which config files it reads or writes. Plus the package dependency graph
(Mermaid) and what each entry point runs. Built from the code itself (ast), so it is never hand-edited.

    python dev/tools/knowledge_graph.py            # rewrite docs/knowledge_graph.md
    python dev/tools/knowledge_graph.py --check    # exit 1 if it is out of date (e.g. in CI or before a commit)

Run it after adding, moving or renaming files or functions, and commit docs/knowledge_graph.md with the change.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "knowledge_graph.md"
PACKAGES = ["core", "hardware", "calibration", "teleop", "perception", "planning", "state_machine", "games"]
ENTRY_POINTS = ["01_setup_usb.py", "02_calibrate.py", "03_teleop.py", "run_game.py"]
SKIP_DIRS = {".git", "__pycache__", "datasets", "runs", "outputs", ".venv", "venv"}

# Which config file a module touches, recognised by the names it uses (core/paths.py and the config helpers).
CONFIG_MARKERS = {
    "config/arms.yaml": [r"\bARMS_FILE\b", r"from core import arms\b", r"from core\.arms import", r"arm_config\."],
    "config/perception.yaml": [r"\bPERCEPTION_FILE\b", r"\bload_roi\b", r"\blatest_weights\b"],
    "config/teleop.yaml": [r"teleop\.yaml"],
    "config/games/<game>.yaml": [r"\bGAMES_CONFIG\b", r"from games import rules", r"task_config\.", r"rules\.load"],
    "config/calibration/camera.yaml": [r"\bCAMERA_CALIBRATION_FILE\b", r"calibrate\.load\(", r"camera_to_table\("],
    "config/calibration/camera_transform.npy": [r"\bCAMERA_TRANSFORM_FILE\b", r"\bCALIBRATION_FILE\b"],
    "config/calibration/<arm>/motors.yaml": [r"calibration_file\(", r"read_motor_calibration", r"lerobot_sync"],
    "config/calibration/<arm>/joint_offsets.yaml": [r"load_fine_joint_offsets", r"own_offsets_file",
                                                    r"correct_joint_position_deg", r"motor_position_deg"],
    "config/calibration/colours.yaml": [r"\bCOLOURS_CALIBRATION_FILE\b", r"\bload_colors\b"],
}
INFRASTRUCTURE = {"core.paths", "core.config"}     # every config path / all YAML I/O: not listed per file
WRITE_MARKERS = [r"config\.save\(", r"np\.save\(", r"\.write_text\(", r"save_fine_joint_offsets\(", r"arm_config\.save\("]


def python_files() -> list[Path]:
    files = []
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        files.append(rel)
    return files


def module_name(rel: Path) -> str:
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def first_line(doc: str | None) -> str:
    """The docstring's first sentence (its first paragraph, up to the first full stop)."""
    if not doc:
        return ""
    paragraph = " ".join(line.strip() for line in doc.strip().split("\n\n")[0].splitlines())
    match = re.search(r"^(.+?\.)(\s+[A-Z(`\[]|$)", paragraph)    # a full stop that ends a sentence
    sentence = match.group(1) if match else paragraph
    return sentence if len(sentence) <= 200 else sentence[:197] + "..."


def analyse(rel: Path, known: set[str]) -> dict:
    source = (ROOT / rel).read_text(encoding="utf-8")
    tree = ast.parse(source)
    me = module_name(rel)
    package = me if rel.name == "__init__.py" else me.rpartition(".")[0]
    imports = set()
    for node in ast.walk(tree):                       # top-level and lazy (inside functions) imports alike
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:                            # relative import
                base = ".".join(package.split(".")[: len(package.split(".")) - node.level + 1] + ([base] if base else []))
            names = [base] + [f"{base}.{alias.name}" for alias in node.names]
        else:
            continue
        for name in names:
            while name and name not in known:     # "from core.arms import HOME_DEG" -> core.arms
                name = name.rpartition(".")[0]
            if name and name != me:
                imports.add(name)
    defines = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            defines.append(f"class {node.name}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            defines.append(f"{node.name}()")
    # a package listed only because a submodule of it was imported ("from core import arms") is left out
    imports = {m for m in imports if not any(other.startswith(m + ".") for other in imports)}
    writes = any(re.search(p, source) for p in WRITE_MARKERS)
    configs = ([] if me in INFRASTRUCTURE else
               [name for name, patterns in CONFIG_MARKERS.items() if any(re.search(p, source) for p in patterns)])
    return {"module": me, "path": rel.as_posix(), "doc": first_line(ast.get_docstring(tree)), "imports": imports,
            "defines": defines, "configs": configs, "writes": writes,
            "main": any(isinstance(n, ast.If) and "__main__" in ast.unparse(n.test) for n in tree.body)}


def group(module: str) -> str:
    top = module.split(".")[0]
    if top == "dev":
        return "dev"
    return top if top in PACKAGES else "entry"


def build() -> str:
    files = python_files()
    known = {module_name(f) for f in files}
    infos = {i["module"]: i for i in (analyse(f, known) for f in files)}
    used_by = defaultdict(set)
    for info in infos.values():
        for target in info["imports"]:
            used_by[target].add(info["module"])

    out = ["# Knowledge graph of the code",
           "",
           "Generated by `python dev/tools/knowledge_graph.py` from the code itself; do not edit by hand (rebuild it",
           "after changing files). Use it to find where something lives before searching. Per file: purpose,",
           "what it defines, imports (project modules only, lazy imports included), who imports it, and the config",
           "files it uses, directly or through core/arms.py, games/rules.py, ... (every path is in `core/paths.py`, all",
           "YAML reading and writing in `core/config.py`). `dev/` is listed briefly: nothing outside `dev/` imports it.",
           ""]

    # package graph
    edges = defaultdict(int)
    for info in infos.values():
        a = group(info["module"])
        for target in info["imports"]:
            b = group(target)
            if a != b and "dev" not in (a, b):
                edges[(a, b)] += 1
    out += ["## Package dependencies", "", "```mermaid", "graph LR"]
    out += [f"    {a} --> {b}" for (a, b) in sorted(edges)]
    out += ["```", "", "(`entry` = the numbered scripts and run_game.py at the repository root.)", ""]

    # entry points
    out += ["## Entry points", "", "| command | runs | purpose |", "|---|---|---|"]
    for name in ENTRY_POINTS:
        info = infos.get(module_name(Path(name)))
        if info:
            runs = ", ".join(f"`{m}`" for m in sorted(info["imports"])) or "-"
            out.append(f"| `python {name}` | {runs} | {info['doc']} |")
    out.append("")

    # config files
    by_config = defaultdict(list)
    for info in infos.values():
        if group(info["module"]) == "dev":
            continue
        for name in info["configs"]:
            by_config[name].append(info["module"])
    out += ["## Config files and the modules that use them", ""]
    for name in CONFIG_MARKERS:
        if by_config.get(name):
            out.append(f"- `{name}`: " + ", ".join(f"`{m}`" for m in sorted(by_config[name])))
    out.append("")

    # modules
    out += ["## Modules", ""]
    order = PACKAGES + ["entry", "dev"]
    for package in order:
        members = sorted((i for i in infos.values() if group(i["module"]) == package), key=lambda i: i["path"])
        if not members:
            continue
        out += [f"### {package}/" if package not in ("entry", "dev") else
                ("### Entry points (repository root)" if package == "entry" else "### dev/ (experiments and tools)"), ""]
        for info in members:
            if info["path"].endswith("__init__.py") and not info["doc"] and not info["defines"]:
                continue
            if package == "dev":
                deps = ", ".join(sorted(m for m in info["imports"] if not m.startswith("dev")))
                out.append(f"- `{info['path']}`: {info['doc'] or '(no docstring)'}" + (f" Uses: {deps}." if deps else ""))
                continue
            out.append(f"**`{info['path']}`**" + (" (runnable)" if info["main"] else "") + f": {info['doc']}")
            if info["defines"]:
                out.append(f"- defines: {', '.join(info['defines'])}")
            if info["imports"]:
                out.append(f"- imports: {', '.join(sorted(info['imports']))}")
            users = sorted(used_by.get(info["module"], ()))
            if users:
                main_users = [u for u in users if not u.startswith("dev")]
                dev_users = len(users) - len(main_users)
                out.append(f"- used by: {', '.join(main_users) or '-'}" + (f" (+{dev_users} in dev/)" if dev_users else ""))
            if info["configs"]:
                out.append(f"- config: {', '.join(info['configs'])}" + (" (also saves files)" if info["writes"] else ""))
            out.append("")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 if docs/knowledge_graph.md is out of date")
    ap.add_argument("--quiet", action="store_true", help="print nothing (e.g. from a git or Claude Code hook)")
    args = ap.parse_args(argv)
    text = build()
    current = OUT.read_text(encoding="utf-8") if OUT.is_file() else ""
    if args.check:
        if text != current:
            print(f"{OUT.relative_to(ROOT)} is out of date: python dev/tools/knowledge_graph.py")
            sys.exit(1)
        print(f"{OUT.relative_to(ROOT)} is up to date")
        return
    if text != current:
        OUT.write_text(text, encoding="utf-8")
        if not args.quiet:
            print(f"wrote {OUT.relative_to(ROOT)}")
    elif not args.quiet:
        print(f"{OUT.relative_to(ROOT)} already up to date")


if __name__ == "__main__":
    main()
