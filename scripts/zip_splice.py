"""Exact per-scene splice of scored packages: copy compressed members byte-for-byte. usage: zip_splice.py out.zip scene=src.zip ..."""
import sys, zipfile, shutil
out=sys.argv[1]; m=dict(a.split('=') for a in sys.argv[2:])
first=zipfile.ZipFile(list(m.values())[0])
with zipfile.ZipFile(out,'w') as zo:
    zo.writestr(first.getinfo('team_name.txt'), first.read('team_name.txt'))
    n=0
    for sc,src in m.items():
        zi=zipfile.ZipFile(src)
        for info in zi.infolist():
            if f'/renders/{sc}/' in info.filename and not info.is_dir():
                with zi.open(info) as f, zo.open(info,'w') as g: shutil.copyfileobj(f,g,1<<24)
                n+=1
print(out,n)
