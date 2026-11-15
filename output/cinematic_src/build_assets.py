import cv2, numpy as np, pickle
from scipy.interpolate import LinearNDInterpolator
from face_warp import Head, OVAL, FH, UP_IN, LO_IN
EXPR = {
 'scream': [dict(jaw=j, brow=min(1, j*1.1), lid=min(1, j*1.5)) for j in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)],
 'joy':    [dict(smile=0.5), dict(smile=1.0, jaw=0.15), dict(smile=1.0, jaw=0.35), dict(smile=1.0, jaw=0.55)],
 'awe':    [dict(jaw=0.0, brow=0.5), dict(jaw=0.18, brow=0.7, lid=0.6), dict(jaw=0.32, brow=0.8, lid=0.8)],
}
ANCHOR = (360, 596)
def relief_map(h):
    """high-pass face relief from the 3D landmarks (nose, sockets, lips), zero outside the face"""
    lm = h.lm3
    interp = LinearNDInterpolator(lm[:, :2], -lm[:, 2], fill_value=np.nan)
    yy, xx = np.mgrid[0:h.h, 0:h.w]
    Hf = interp(xx, yy).astype(np.float32); inside = ~np.isnan(Hf)
    # fill outside with nearest inside value so the blur does not bleed zeros in
    Hf = np.where(inside, Hf, 0)
    w = inside.astype(np.float32)
    def nblur(a, s): return cv2.GaussianBlur(a*w, (0, 0), s)/np.maximum(cv2.GaussianBlur(w, (0, 0), s), 1e-3)
    hp = nblur(Hf, 3.0) - nblur(Hf, 38.0)
    m = cv2.erode(inside.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), 14)
    return (hp*m).astype(np.float32)
def dome(al):
    a = (al > 0.5).astype(np.uint8)
    a = cv2.morphologyEx(a, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    d = cv2.distanceTransform(a, cv2.DIST_L2, 5)
    R = 0.46*np.ptp(np.where(a.any(0))[0])
    Hb = R*np.sqrt(np.clip(1-(1-np.clip(d/R, 0, 1))**2, 0, 1))
    return cv2.GaussianBlur(Hb.astype(np.float32), (0, 0), 11)
out = []
for i in range(4):
    h = Head(f'head_{i}.npz'); rel = relief_map(h)
    sets = {}; boxes = []
    for name, plist in EXPR.items():
        frames = []
        for kw in plist:
            rgb, al, (rl,) = h.render(extra=[rel], **kw)
            rl = cv2.GaussianBlur(rl, (0, 0), 2.5)
            H = dome(al) + 1.25*rl
            gx = cv2.Sobel(H, cv2.CV_32F, 1, 0, ksize=3)/8; gy = cv2.Sobel(H, cv2.CV_32F, 0, 1, ksize=3)/8
            n = np.dstack([-gx, -gy, np.ones_like(gx)]); n /= np.linalg.norm(n, axis=2, keepdims=True)
            # edge distance (for rim light) and 2D outward direction
            a8 = (al > 0.5).astype(np.uint8)
            ed = np.clip(cv2.distanceTransform(a8, cv2.DIST_L2, 5), 0, 63)
            frames.append((rgb, al, n, ed)); ys, xs = np.where(al > 0.01); boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
        sets[name] = frames
    b = np.array(boxes); x0, y0, x1, y1 = max(0, b[:, 0].min()-12), max(0, b[:, 1].min()-12), min(h.w, b[:, 2].max()+13), min(h.h, b[:, 3].max()+13)
    packed = {nm: [dict(rgb=np.clip(r[y0:y1, x0:x1], 0, 255).astype(np.uint8), alpha=(a[y0:y1, x0:x1]*255).astype(np.uint8),
                        nrm=np.round(n[y0:y1, x0:x1]*127).astype(np.int8), edge=(ed[y0:y1, x0:x1]*4).astype(np.uint8)) for r, a, n, ed in fr] for nm, fr in sets.items()}
    out.append(dict(expr=packed, anchor=(int(ANCHOR[0]-x0), int(ANCHOR[1]-y0)), size=(int(x1-x0), int(y1-y0)), skin=h.skin, FH=FH,
                    face_c=(float(h.cen[0]-x0), float(h.cen[1]-y0))))
    print(i, 'bbox', x0, y0, x1, y1, 'anchor', out[-1]['anchor'])
pickle.dump(out, open('heads.pkl', 'wb'))
t = []
for i in range(4):
    e = out[i]['expr']['scream'][4]; rgb, al = e['rgb'], e['alpha']; n = e['nrm'].astype(np.float32)/127; ed = e['edge'].astype(np.float32)/4
    a = al[..., None]/255.
    nv = ((n*0.5+0.5)[..., ::-1]*255)
    l = np.array([0, 0.8, 0.6]); l /= np.linalg.norm(l); up = np.clip((n*l).sum(2), 0, 1)
    rim = np.exp(-ed/9.0)*np.clip(0.35+0.65*np.hypot(n[..., 0], n[..., 1]), 0, 1)
    alb = rgb.astype(np.float32)
    lit = alb*(0.20*np.array([1.0, 0.75, 0.7]) + up[..., None]*np.array([0.25, 0.6, 1.25])*0.95) + rim[..., None]*np.array([40, 140, 255])*1.0
    bg = np.array((12, 8, 20))
    row = np.hstack([(nv*a+bg*(1-a)), (np.clip(lit, 0, 255)*a+bg*(1-a))]).astype(np.uint8)
    t.append(cv2.resize(row, (700, int(row.shape[0]*700/row.shape[1]))))
hh = min(x.shape[0] for x in t)
cv2.imwrite('relight.jpg', np.vstack([np.hstack([x[:hh] for x in t[:2]]), np.hstack([x[:hh] for x in t[2:]])]))
