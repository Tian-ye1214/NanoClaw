"""Deterministic local fixtures with answers kept outside the Agent workspace."""

from __future__ import annotations

import csv
import io
import json
import secrets
from pathlib import Path


def make_assets(project: Path, truth_path: Path) -> dict:
    import cv2
    import fitz
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from docx import Document
    from openpyxl import Workbook
    from pptx import Presentation
    from pptx.util import Inches

    project.mkdir(parents=True, exist_ok=True)
    code = str(secrets.randbelow(900000) + 100000)
    video_codes = [str(secrets.randbelow(9000) + 1000) for _ in range(3)]
    memo = "LOTUS-" + secrets.token_hex(5).upper()
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 76)
    pic = Image.new("RGB", (900, 500), "white")
    draw = ImageDraw.Draw(pic)
    draw.polygon([(100, 400), (240, 150), (380, 400)], fill="red")
    draw.text((460, 220), code, font=font, fill="black")
    pic.save(project / "图片 样例.png")
    pic.save(project / "image.png")
    writer = cv2.VideoWriter(
        str(project / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 3, (640, 360)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the original MP4 fixture")
    for label, color in zip(video_codes, [(0, 0, 220), (0, 170, 0), (220, 0, 0)]):
        for _ in range(12):
            frame = np.full((360, 640, 3), color, dtype=np.uint8)
            cv2.putText(
                frame,
                label,
                (150, 205),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (255, 255, 255),
                6,
            )
            writer.write(frame)
    writer.release()
    (project / "视频 样例.mp4").write_bytes((project / "video.mp4").read_bytes())
    rows = [("North", 10), ("South", 15), ("North", 25), ("South", 25)]
    with (project / "sales.csv").open("w", encoding="utf-8", newline="") as stream:
        out = csv.writer(stream)
        out.writerow(["region", "amount"])
        out.writerows(rows)
    wb = Workbook()
    ws = wb.active
    ws.title = "Sales"
    ws.append(["region", "amount"])
    for row in rows:
        ws.append(row)
    ws["D1"] = "formula"
    ws["D2"] = "=SUM(B2:B5)"
    wb.save(project / "sales.xlsx")
    document_codes = {
        kind: str(secrets.randbelow(900000) + 100000)
        for kind in ("word", "slides", "pdf")
    }

    def inline_image(kind):
        picture = Image.new("RGB", (420, 160), "white")
        ImageDraw.Draw(picture).text(
            (30, 40), document_codes[kind], font=font, fill="black"
        )
        data = io.BytesIO()
        picture.save(data, format="PNG")
        return data.getvalue()

    word = Document()
    word.add_heading("Acceptance reference", 0)
    word.add_paragraph("Document marker: DOC-6719")
    table = word.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "category"
    table.rows[0].cells[1].text = "number"
    table.add_row().cells[0].text = "blue"
    table.rows[1].cells[1].text = "37"
    word.add_picture(io.BytesIO(inline_image("word")), width=Inches(3))
    word.save(project / "reference.docx")
    slides = Presentation()
    slide = slides.slides.add_slide(slides.slide_layouts[6])
    slide.shapes.add_textbox(
        Inches(1), Inches(1), Inches(7), Inches(1)
    ).text = "Slide marker: PPT-4281"
    slide.notes_slide.notes_text_frame.text = "Speaker note: inspect the second source."
    slide.shapes.add_picture(
        io.BytesIO(inline_image("slides")), Inches(1), Inches(3), width=Inches(3)
    )
    slides.save(project / "reference.pptx")
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "PDF marker: PDF-8357. Contract amount: 93.")
    page.insert_image(fitz.Rect(72, 100, 400, 225), stream=inline_image("pdf"))
    pdf.save(str(project / "reference.pdf"))
    pdf.close()
    (project / "reference.md").write_text(
        "# Reference\nMarkdown marker: MD-2596\n", encoding="utf-8"
    )
    (project / "reference.html").write_text(
        '<h1>Reference</h1><p>HTML marker: HTML-3064</p><script>throw Error("must not execute")</script>',
        encoding="utf-8",
    )
    for index in range(21):
        (project / f"ref-{index:02}.txt").write_text(
            f"Fixture entry {index}.", encoding="utf-8"
        )
    truth = dict(
        image_code=code,
        video_codes=video_codes,
        memo=memo,
        totals={"North": 35, "South": 40},
        document_codes=document_codes,
        document_markers=["DOC-6719", "PPT-4281", "PDF-8357", "MD-2596", "HTML-3064"],
    )
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path.write_text(
        json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return truth
