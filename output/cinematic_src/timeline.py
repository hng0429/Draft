"""Global timeline: frame index -> shot + local time. Renders frame ranges to PNG."""
import sys, os, time, numpy as np, cv2
from engine import FPS, finish
SHOTS = [('S1', 3.6), ('S2', 3.8), ('S3', 2.6), ('S4', 3.6), ('S5', 5.2)]
STARTS = np.cumsum([0]+[d for _, d in SHOTS])
TOTAL = float(STARTS[-1]); NFR = int(round(TOTAL*FPS))
def locate(fi):
    t = fi/FPS
    for k, (name, d) in enumerate(SHOTS):
        if t < STARTS[k+1]-1e-9 or k == len(SHOTS)-1: return name, float(t-STARTS[k]), float(d)
_fn = {}
def render(fi):
    name, tl, d = locate(fi)
    if name not in _fn:
        if name in ('S1', 'S2', 'S3'):
            import hell; _fn.update(S1=hell.shot_s1, S2=hell.shot_s2, S3=hell.shot_s3)
        else:
            import hope; _fn.update(S4=hope.shot_s4, S5=hope.shot_s5)
    return finish(_fn[name](tl, fi, d), fi)
if __name__ == '__main__':
    out = sys.argv[1]; frames = [int(v) for v in sys.argv[2].split(',')] if ',' in sys.argv[2] else list(range(*[int(v) for v in sys.argv[2].split(':')]))
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    for fi in frames:
        p = f'{out}/f{fi:04d}.png'
        if os.path.exists(p): continue
        cv2.imwrite(p, render(fi), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    print('done', frames[0], frames[-1], round(time.time()-t0, 1), 's', flush=True)
