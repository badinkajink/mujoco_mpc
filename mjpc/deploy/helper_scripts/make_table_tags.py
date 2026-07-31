#!/usr/bin/env python3
"""make_table_tags.py -- print-exact AprilTag 36h11 sheet for the adjustable lab
table (top 118 x 59.5 cm, height 57-123 cm), sized for the D435i head camera.

LAYOUT (7 tags, chosen from the D435i pixel budget ~13.9/d px per cm):
  IDs 0-3   10 cm  TOP surface, along the front (118 cm) edge zone, centers
                   ~3 cm behind the edge, spaced ~28 cm -> >=2 visible on
                   approach, >=1 survives arm occlusion while leaning.
  IDs 10-11  5 cm  TOP surface at the working-edge centre, either side of the
                   brace point -> own the deep-brace range (0.3-0.5 m, where
                   the FOV is only ~42 cm wide and 10 cm tags overflow).
  ID 20     12 cm  TOP surface centre-back -> long-range acquisition (~2 m).

SIZE CONVENTION: "tag size" = the OUTER edge of the BLACK border square (what
apriltag_ros / OpenCV solvePnP expect). Each tag is printed with a >=1-module
white quiet zone. A 100.0 mm calibration ruler is printed on every page --
after printing at 100% scale (NO "fit to page"), measure it; if it is not
100.0 +- 0.5 mm, multiply all tag sizes you enter in the detector config by
(measured/100).

Output: table_tags_36h11.pdf (A4 pages, 300 dpi) + table_tag_bundle.yaml
template with the recommended layout coordinates to fill in after mounting.

Run:  .venv/bin/python make_table_tags.py [outdir]
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DPI = 300
MM = DPI / 25.4                       # px per mm
A4 = (int(210 * MM), int(297 * MM))   # portrait, px

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
MODULES = 10                          # 36h11: 8 black-bordered modules + we add 1 white each side

TAGS = [  # (id, black-square edge mm, label)
    (0, 100.0, "TOP front-edge zone, leftmost  (10 cm)"),
    (1, 100.0, "TOP front-edge zone, mid-left  (10 cm)"),
    (2, 100.0, "TOP front-edge zone, mid-right (10 cm)"),
    (3, 100.0, "TOP front-edge zone, rightmost (10 cm)"),
    (10, 50.0, "TOP working edge, left of brace point  (5 cm)"),
    (11, 50.0, "TOP working edge, right of brace point (5 cm)"),
    (20, 120.0, "TOP centre-back, far acquisition (12 cm)"),
]


def tag_image(tid, edge_mm):
    """Full printable tile: black tag (edge_mm) + 1-module quiet zone, exact px."""
    module_mm = edge_mm / 8.0                       # 36h11 tag = 8 modules across
    quiet_mm = module_mm                            # >=1 module white border
    tag_px = int(round(edge_mm * MM))
    # render at a multiple of 8 then resize NEAREST to exact px (keeps modules crisp)
    raw = cv2.aruco.generateImageMarker(DICT, tid, 8 * 40)
    tag = cv2.resize(raw, (tag_px, tag_px), interpolation=cv2.INTER_NEAREST)
    q = int(round(quiet_mm * MM))
    tile = np.full((tag_px + 2 * q, tag_px + 2 * q), 255, np.uint8)
    tile[q:q + tag_px, q:q + tag_px] = tag
    return Image.fromarray(tile), quiet_mm


def _font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def ruler(draw, x_mm, y_mm):
    """100.0 mm calibration bar with end ticks."""
    x0, y0 = int(x_mm * MM), int(y_mm * MM)
    x1 = int((x_mm + 100.0) * MM)
    draw.line([(x0, y0), (x1, y0)], fill=0, width=3)
    for xx in (x0, x1):
        draw.line([(xx, y0 - int(3 * MM)), (xx, y0 + int(3 * MM))], fill=0, width=3)
    draw.text((x0, y0 + int(1.5 * MM)),
              "calibration ruler: must measure EXACTLY 100.0 mm printed "
              "(print at 100% scale, never 'fit to page')", font=_font(int(2.6 * MM)), fill=0)


def page(items, title):
    """items: list of (tag_tile PIL, id, edge_mm, label). Stacked vertically."""
    im = Image.new("L", A4, 255)
    d = ImageDraw.Draw(im)
    d.text((int(15 * MM), int(8 * MM)),
           f"H1-2 lean-table AprilTags -- 36h11 -- {title}", font=_font(int(4 * MM)), fill=0)
    ruler(d, 15, 20)
    y = int(30 * MM)
    for tile, tid, edge, label in items:
        im.paste(tile, (int(15 * MM), y))
        ty = y + tile.size[1] // 2
        d.text((int(15 * MM) + tile.size[0] + int(6 * MM), ty - int(6 * MM)),
               f"ID {tid}  ({edge / 10:.0f} cm)", font=_font(int(4 * MM)), fill=0)
        d.text((int(15 * MM) + tile.size[0] + int(6 * MM), ty + int(1 * MM)),
               label, font=_font(int(2.8 * MM)), fill=0)
        y += tile.size[1] + int(8 * MM)
    return im


def main():
    outdir = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.expanduser("~/Desktop/h12/table_tags")
    os.makedirs(outdir, exist_ok=True)

    tiles = {tid: tag_image(tid, edge) for tid, edge, _ in TAGS}
    pages = [
        page([(tiles[0][0], 0, 100.0, TAGS[0][2]), (tiles[1][0], 1, 100.0, TAGS[1][2])],
             "page 1/4 -- 10 cm tags A"),
        page([(tiles[2][0], 2, 100.0, TAGS[2][2]), (tiles[3][0], 3, 100.0, TAGS[3][2])],
             "page 2/4 -- 10 cm tags B"),
        page([(tiles[20][0], 20, 120.0, TAGS[6][2])], "page 3/4 -- 12 cm far tag"),
        page([(tiles[10][0], 10, 50.0, TAGS[4][2]), (tiles[11][0], 11, 50.0, TAGS[5][2])],
             "page 4/4 -- 5 cm working-edge tags"),
    ]
    pdf = os.path.join(outdir, "table_tags_36h11.pdf")
    pages[0].save(pdf, save_all=True, append_images=pages[1:],
                  resolution=DPI, dpi=(DPI, DPI))
    print(f"[tags] wrote {pdf} ({len(pages)} pages, {DPI} dpi)")

    # bundle template: fill x/y after mounting (table frame: origin = front-left
    # corner of the TOP surface, x along the 118 cm front edge, y toward the
    # back, z up; all tag centres z=0 on the top surface).
    yaml = os.path.join(outdir, "table_tag_bundle.yaml")
    with open(yaml, "w") as f:
        f.write("""# AprilTag bundle for the adjustable lean table (fill x/y AFTER mounting).
# Frame: origin = front-left corner of the TOP surface; x along the 118cm front
# edge (left->right seen from the robot), y from front edge toward the back,
# z up. Measure each tag CENTRE with calipers/tape to ~1mm; also re-measure the
# printed black-square 'size' of each tag (printers scale silently).
# Suggested mounting (centres): ids 0-3 at y=0.08m, x = 0.15/0.44/0.74/1.03;
# ids 10-11 at y=0.05m, x = 0.49/0.69 (brace point ~x=0.59); id 20 at
# x=0.59, y=0.45.
standalone_tags: []
tag_bundles:
  - name: lean_table
    layout:
      - {id: 0,  size: 0.100, x: 0.000, y: 0.000, z: 0.0, qw: 1, qx: 0, qy: 0, qz: 0}
      - {id: 1,  size: 0.100, x: 0.000, y: 0.000, z: 0.0, qw: 1, qx: 0, qy: 0, qz: 0}
      - {id: 2,  size: 0.100, x: 0.000, y: 0.000, z: 0.0, qw: 1, qx: 0, qy: 0, qz: 0}
      - {id: 3,  size: 0.100, x: 0.000, y: 0.000, z: 0.0, qw: 1, qx: 0, qy: 0, qz: 0}
      - {id: 10, size: 0.050, x: 0.000, y: 0.000, z: 0.0, qw: 1, qx: 0, qy: 0, qz: 0}
      - {id: 11, size: 0.050, x: 0.000, y: 0.000, z: 0.0, qw: 1, qx: 0, qy: 0, qz: 0}
      - {id: 20, size: 0.120, x: 0.000, y: 0.000, z: 0.0, qw: 1, qx: 0, qy: 0, qz: 0}
""")
    print(f"[tags] wrote {yaml} (bundle template -- fill coordinates after mounting)")


if __name__ == "__main__":
    main()
