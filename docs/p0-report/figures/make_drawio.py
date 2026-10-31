"""Generate an editable draw.io companion to the report's architecture figure."""

from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    document = ET.Element("mxfile", host="app.diagrams.net", version="26.0")
    page = ET.SubElement(document, "diagram", name="P0 encoder-decoder", id="p0")
    graph = ET.SubElement(page, "mxGraphModel", grid="1", gridSize="10",
                          page="1", pageScale="1", pageWidth="1100", pageHeight="1000")
    root = ET.SubElement(graph, "root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")
    nodes = {
        "src": (60, 880, "Kazakh source tokens x", "plain"),
        "tgt": (640, 880, "Shifted Russian tokens y<t", "plain"),
        "se": (60, 790, "Token embedding E", "enc"),
        "te": (640, 790, "Token embedding E (shared)", "dec"),
        "ea": (60, 650, "Pre-RMSNorm → bidirectional self-attention\n8 query / 8 KV heads\nQ/K RMSNorm + RoPE; residual add", "enc"),
        "ef": (60, 510, "Pre-RMSNorm → SwiGLU\n512 → 4096 → 512; residual add", "enc"),
        "da": (640, 650, "Pre-RMSNorm → causal self-attention\n8 query / 4 KV heads (GQA)\nQ/K RMSNorm + RoPE; residual add", "dec"),
        "dc": (640, 510, "Pre-RMSNorm → cross-attention\n8 query / 8 KV heads; residual add", "dec"),
        "df": (640, 390, "Pre-RMSNorm → SwiGLU\n512 → 4096 → 512; residual add", "dec"),
        "mem": (60, 320, "Encoder memory Hx", "enc"),
        "norm": (640, 270, "Final decoder RMSNorm", "dec"),
        "logits": (640, 160, "Tied projection h Eᵀ\n32,000 vocabulary logits", "plain"),
        "out": (640, 60, "Next Russian token\nRepeat until EOS", "plain"),
    }
    fills = {
        "se": "#FCE0E1", "te": "#FCE0E1",
        "ea": "#FFE2BB", "da": "#FFE2BB", "dc": "#FFE2BB",
        "ef": "#C2E8F7", "df": "#C2E8F7", "norm": "#F2F4C1",
        "logits": "#DBDFEF", "out": "#CCE7CF",
    }
    for name, (x, y, label, kind) in nodes.items():
        fill, stroke = fills.get(name, "#FFFFFF"), "#000000"
        cell = ET.SubElement(root, "mxCell", id=name, value=label, vertex="1", parent="1",
                             style=f"rounded=1;whiteSpace=wrap;html=0;fontFamily=Helvetica;fontSize=15;fillColor={fill};strokeColor={stroke};")
        ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width="360",
                      height="85" if name in ("ea", "da") else "65", **{"as": "geometry"})
    edges = [("src", "se"), ("se", "ea"), ("ea", "ef"), ("ef", "mem"),
             ("tgt", "te"), ("te", "da"), ("da", "dc"), ("dc", "df"),
             ("df", "norm"), ("norm", "logits"), ("logits", "out"), ("mem", "dc")]
    for index, (source, target) in enumerate(edges):
        label = "K, V" if source == "mem" else ""
        edge = ET.SubElement(root, "mxCell", id=f"edge-{index}", value=label,
                             source=source, target=target, edge="1", parent="1",
                             style="edgeStyle=orthogonalEdgeStyle;endArrow=block;strokeColor=#000000;fontSize=14;")
        ET.SubElement(edge, "mxGeometry", relative="1", **{"as": "geometry"})
    for name, x, y, label in [
        ("enc-label", 60, 470, "Encoder block ×12"),
        ("dec-label", 640, 350, "Decoder block ×12"),
        ("notes", 50, 60, "P0 · 202,016,256 parameters\nHidden size 512; head size 64\nShared vocabulary 32,000; limit 256 tokens\nTraining: teacher forcing\nInference: self-attention KV-cache\nNo final encoder stack norm"),
    ]:
        cell = ET.SubElement(root, "mxCell", id=name, value=label, vertex="1", parent="1",
                             style="text;whiteSpace=wrap;html=0;fontSize=16;fontFamily=Helvetica;fontColor=#000000;align=left;")
        ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width="470",
                      height="170" if name == "notes" else "35", **{"as": "geometry"})
    path = Path(__file__).with_name("p0-architecture.drawio")
    ET.indent(document, space="  ")
    ET.ElementTree(document).write(path, encoding="utf-8", xml_declaration=True)
    print(path)


if __name__ == "__main__":
    main()
