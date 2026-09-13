"""Real local parsing/conversion checks. This does not claim model recognition passed."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from acceptance_assets import make_assets


async def validate_references(project: Path, truth: dict, *, include_video=True):
    from redlotus.cli.file_ref import load_file_refs
    from redlotus.references.readers import OfficeConverter
    from redlotus.ModelGateway.input_policy import ModelInputPolicy
    from redlotus.workspace.workspace import set_workspace
    from pydantic_ai import BinaryContent

    set_workspace(project)
    converter = OfficeConverter()
    legacy = project / "legacy"
    for name, fmt in [
        ("reference.docx", "doc:MS Word 97"),
        ("reference.pptx", "ppt:MS PowerPoint 97"),
        ("sales.xlsx", "xls:MS Excel 97"),
    ]:
        await converter.convert(project / name, fmt, legacy)
    names = [
        "reference.pdf",
        "reference.docx",
        "legacy/reference.doc",
        "sales.csv",
        "sales.xlsx",
        "legacy/sales.xls",
        "reference.pptx",
        "legacy/reference.ppt",
        "reference.md",
        "reference.html",
        "图片 样例.png",
    ]
    if include_video:
        names.append("视频 样例.mp4")
    refs = await load_file_refs(" ".join('@"' + name + '"' for name in names))
    assert len(refs) == len(names)
    results = []
    for ref in refs:
        text = "\n".join(part.text for part in ref.parts)
        if ref.name.endswith((".doc", ".docx")):
            assert "DOC-6719" in text and "37" in text
        if ref.name.endswith((".ppt", ".pptx")):
            assert "PPT-4281" in text and "Speaker note" in text
        if ref.name.endswith((".xls", ".xlsx")):
            assert "SUM(B2:B5)" in text and "North" in text
        if ref.name.endswith(".pdf"):
            assert "PDF-8357" in text and "93" in text
        if ref.name.endswith(".html"):
            assert "HTML-3064" in text and "must not execute" not in text
        if ref.name.endswith((".png", ".mp4")):
            native = [
                item for item in ref.to_prompt() if isinstance(item, BinaryContent)
            ]
            assert native and native[0].data == ref.snapshot.read_bytes()
        results.append(
            dict(name=ref.name, parts=[part.kind for part in ref.parts], passed=True)
        )
    twenty = await load_file_refs(" ".join("@ref-%02d.txt" % i for i in range(20)))
    assert len(twenty) == 20
    try:
        await load_file_refs(" ".join("@ref-%02d.txt" % i for i in range(21)))
        raise AssertionError("21 files were accepted")
    except ValueError:
        pass
    duplicate = await load_file_refs(
        '@ref-00.txt @"' + str(project / "ref-00.txt") + '"'
    )
    assert len(duplicate) == 1
    policy = ModelInputPolicy.for_role()
    too_large = project / "too-large.txt"
    with too_large.open("wb") as stream:
        stream.seek(policy.max_file_bytes)
        stream.write(b"x")
    try:
        await load_file_refs("@too-large.txt")
        raise AssertionError("Over-limit file was accepted")
    except ValueError:
        pass
    original = twenty[0]
    (project / "ref-00.txt").write_text("new source version", encoding="utf-8")
    assert original.snapshot.read_text(encoding="utf-8") == "Fixture entry 0."
    changed = (await load_file_refs("@ref-00.txt"))[0]
    assert changed.id != original.id
    (project / "ref-00.txt").write_bytes(original.snapshot.read_bytes())
    report = dict(
        scope="Real local files and LibreOffice conversion; LLM recognition still requires live acceptance.",
        formats=results,
        max_files=20,
        max_file_bytes=policy.max_file_bytes,
        dedup=True,
        immutable_snapshot=True,
        expected_image=truth["image_code"],
        expected_video=truth["video_codes"],
    )
    return report, names


async def run(root: Path):
    project = root / "project"
    truth = make_assets(project, root / "truth.json")
    report, _ = await validate_references(project, truth)
    (root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(dict(formats=len(report["formats"]), local_checks="passed")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["REDLOTUS_DATA_DIR"] = str(root / "state")
    asyncio.run(run(root))


if __name__ == "__main__":
    main()
