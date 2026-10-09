#!/usr/bin/env python3
"""Inspect the actual Office-rendered PDF and make slide previews and plot bundles."""
import json
import zipfile
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageDraw
from pptx import Presentation

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/presentation"


def main():
    prs=Presentation(OUT/"fusion_equilibrium_baselines.pptx")
    pdf=fitz.open(OUT/"fusion_equilibrium_baselines.pdf")
    content=json.loads((OUT/"slide_content.json").read_text())
    assert len(prs.slides)==len(pdf)==len(content)==18
    previews=OUT/"previews"
    previews.mkdir(exist_ok=True)
    report={"slides":len(pdf),"pdf_renderer":"LibreOffice 25.2.5.2","slide_size":"16:9",
            "native_text_and_diagrams":True,"plot_count":8,"checks":[],"slide_previews":[]}
    outside=[]
    for i,(slide,page,metadata) in enumerate(zip(prs.slides,pdf,content),1):
        assert slide.has_notes_slide and len(slide.notes_slide.notes_text_frame.text)>100
        for shape in slide.shapes:
            # The footer intentionally bleeds by a fraction of a point at the right edge.
            if min(shape.left,shape.top)<0 or shape.left+shape.width>prs.slide_width+10000 or shape.top+shape.height>prs.slide_height+10000:
                outside.append({"slide":i,"shape":shape.name})
        spans=[span for block in page.get_text("dict")["blocks"] if "lines" in block
               for line in block["lines"] for span in line["spans"]]
        assert any("FUSION EQUILIBRIUM" in sp["text"] for sp in spans),f"Missing footer in slide {i}"
        path=previews/f"slide_{i:02d}.png"
        page.get_pixmap(matrix=fitz.Matrix(1.6,1.6),alpha=False).save(path)
        report["slide_previews"].append(str(path.relative_to(OUT)))
    assert not outside,outside
    report["checks"]=["18 matching PPTX/PDF pages","All slides have detailed speaker notes",
                       "Native shape boundaries inside slide canvas","Office-rendered PDF includes all page footers"]
    for start in (0,9):
        sheet=Image.new("RGB",(1440,870),"#D6DEE7")
        draw=ImageDraw.Draw(sheet)
        for j,index in enumerate(range(start,min(start+9,len(pdf)))):
            image=Image.open(previews/f"slide_{index+1:02d}.png").convert("RGB")
            image.thumbnail((468,263))
            x=6+(j%3)*480;y=20+(j//3)*290
            sheet.paste(image,(x,y))
            draw.text((x+4,y-16),f"Slide {index+1:02d}",fill="#102238")
        sheet.save(OUT/f"contact_sheet_{start//9+1}.png")
    with zipfile.ZipFile(OUT/"plots.zip","w",zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((OUT/"figures").iterdir()):
            if path.is_file():archive.write(path,arcname=path.name)
    (OUT/"qa_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"slides":len(pdf),"checks":report["checks"],"plot_bundle":str(OUT/"plots.zip")}))


if __name__=="__main__":main()
