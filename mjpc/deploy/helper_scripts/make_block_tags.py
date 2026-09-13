#!/usr/bin/env python3
"""make_block_tags.py -- print-exact AprilTag 36h11 sheet for the 5x5x5 cm target block.

IDs 31 (TOP), 32 (LEFT side, robot's left), 33 (RIGHT side), 34 (BACK), 45 mm black square
(same size as the existing front tag 30, tag30_36h11_45mm_x6.pdf). Two copies of each id.
Print at 100 % (no fit-to-page), measure the 100.0 mm ruler; scale tag sizes by (measured/100).
Mount each tag CENTRED on its face. The block is 50 mm so a 45 mm tag + 1-module quiet zone
(5.6 mm) just fits the face; trim the quiet zone to the face edge if needed, keep the black square.
Output: ~/Desktop/h12/table_tags/block_tags_31-34_45mm.pdf
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_table_tags import tag_image, ruler, _font, MM, A4
from PIL import Image, ImageDraw
TAGS=[(31,"TOP"),(32,"LEFT side (robot's left)"),(33,"RIGHT side (robot's right)"),(34,"BACK")]
out=os.path.expanduser(sys.argv[1] if len(sys.argv)>1 else "~/Desktop/h12/table_tags/block_tags_31-34_45mm.pdf")
page=Image.new("L",A4,255); d=ImageDraw.Draw(page)
d.text((int(15*MM),int(8*MM)),"5 cm BLOCK TAGS  AprilTag 36h11, 45.0 mm black square, x2 each. Print 100%, check the ruler.",font=_font(26),fill=0)
ruler(d,15,20)
y=34.0
for tid,lab in TAGS:
    x=15.0
    for copy in range(2):
        tile,q=tag_image(tid,45.0); page.paste(tile,(int(x*MM),int(y*MM)))
        d.text((int(x*MM),int((y+45+2*q+2)*MM)),f"id {tid}  {lab}",font=_font(20),fill=0)
        x+=80.0
    y+=63.0
page.save(out,"PDF",resolution=300); print("wrote",out)
