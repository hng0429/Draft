"""Land of hope: S4 (front, emerging into the light) and S5 (wide vista + title)."""
import numpy as np, cv2, math, sys
from engine import *
from chars import Camera, draw_character, comp, rgb, HOPE_LIGHT
from hell import char_shadow, ground_shadows, F0

M = 90
VW, VH = W+2*M, H+2*M
def C(r, g, b): return np.array([b, g, r], np.float32)

def sky_layer(sun, seed=0, cloud_amt=1.0, horizon=0.52, clear=None):
    yv = np.linspace(0, 1, VH, dtype=np.float32)
    stops = [(0.0, (26, 60, 140)), (0.20, (56, 110, 192)), (0.36, (118, 166, 216)), (horizon-0.05, (204, 200, 196)), (horizon+0.02, (236, 204, 156)), (1.0, (236, 204, 156))]
    ys = [s[0] for s in stops]
    sky = np.stack([np.interp(yv, ys, [s[1][c]/255.0 for s in stops]) for c in (2, 1, 0)], 1).astype(np.float32)[:, None, :]*np.ones((1, VW, 1), np.float32)
    yy, xx = np.mgrid[0:VH, 0:VW].astype(np.float32)
    r = np.sqrt((xx-sun[0])**2+(yy-sun[1])**2)
    glow = np.exp(-r/120.0)[..., None]*C(1.0, 0.93, 0.74)*0.55+np.exp(-r/420.0)[..., None]*C(1.0, 0.78, 0.48)*0.26
    sky = sky+glow
    # clouds
    n = 0.80*NC.sample(xx*0.24+seed*40, yy*0.72+seed*13)+0.20*NA.sample(xx*0.50+seed*7, yy*1.4)
    dxs = (sun[0]-xx)/(r+1e-3)*22; dys = (sun[1]-yy)/(r+1e-3)*22
    n2 = 0.80*NC.sample((xx+dxs)*0.24+seed*40, (yy+dys)*0.72+seed*13)+0.20*NA.sample((xx+dxs)*0.50+seed*7, (yy+dys)*1.4)
    band = smv(0.02*VH, 0.10*VH, yy)*(1-smv(0.36*VH, (horizon-0.03)*VH, yy))
    near_sun = 1-0.55*np.exp(-r/260.0)
    if clear is not None:
        near_sun = near_sun*(1-0.93*np.exp(-((xx-clear[0])/clear[2])**2-((yy-clear[1])/clear[3])**2))
    d = smv(0.52, 0.70, n)*band*near_sun*cloud_amt
    d2 = smv(0.52, 0.70, n2)*band*near_sun*cloud_amt
    light = np.clip(0.50+2.4*(d-d2), 0, 1)[..., None]
    warm = (1+0.9*np.exp(-r/520.0))[..., None]
    ccol = (C(0.52, 0.58, 0.74)*(1-light)+C(1.0, 0.94, 0.82)*light)*warm
    a = (d**0.8*0.93)[..., None]
    sky = sky*(1-a)+ccol*a
    cir = smv(0.60, 0.88, NB.sample(xx*0.16+90, yy*2.6+seed*3))*(1-smv(0.25*VH, 0.42*VH, yy))*0.22
    sky = sky*(1-cir[..., None])+C(1.0, 0.97, 0.94)*cir[..., None]
    sky = cv2.GaussianBlur(sky, (0, 0), 1.6)
    # sun disc
    disc = np.clip((30-r)/6.0, 0, 1)[..., None]
    sky = sky+disc*C(1.0, 0.95, 0.8)*3.2
    return sky.astype(np.float32), a[..., 0]

def god_rays(sky, occl, sun, strength=0.35):
    src = np.exp(-np.sqrt(((np.mgrid[0:VH, 0:VW][1]-sun[0])**2+(np.mgrid[0:VH, 0:VW][0]-sun[1])**2))/300.0).astype(np.float32)*(1-0.92*occl)
    s4 = cv2.resize(src, (VW//4, VH//4), interpolation=cv2.INTER_AREA)
    acc = np.zeros_like(s4); n = 0
    for k in range(18):
        s = 1+0.85*k/17.0
        Mx = np.array([[s, 0, sun[0]/4*(1-s)], [0, s, sun[1]/4*(1-s)]], np.float32)
        acc += cv2.warpAffine(s4, Mx, (s4.shape[1], s4.shape[0]), flags=cv2.INTER_LINEAR)*(1-0.03*k); n += 1
    acc = cv2.GaussianBlur(acc/n, (0, 0), 1.5)
    rays = cv2.resize(acc, (VW, VH), interpolation=cv2.INTER_LINEAR)[..., None]
    return sky+rays*C(1.0, 0.90, 0.68)*strength*2.2

def ridge(base, amp, seed, fs=1.0, beta=2.0, ridged=0.6, detail=5.0):
    xi = (np.arange(VW)*fs).astype(int) % 4096
    n = noise1d(4096, beta, seed)[xi]; n2 = noise1d(4096, beta+0.4, seed+50)[xi]
    r = (1-np.abs(2*n-1))**1.3
    h = amp*(ridged*r+(1-ridged)*n2)
    h += detail*(noise1d(4096, 2.2, seed+7, lo=40)[(np.arange(VW)*2) % 4096]-0.5)
    return (base-h).astype(np.float32), h.astype(np.float32)

def mountain_layer(base, amp, seed, col, sun, snow=0.0, fs=1.0, haze=C(1.0, 0.90, 0.72), haze_amt=0.5, facet=0.16, detail=0.0):
    yr, h = ridge(base, amp, seed, fs, detail=detail)
    h = cv2.GaussianBlur(h[None, :], (0, 0), 10)[0]
    yy, xx = np.mgrid[0:VH, 0:VW].astype(np.float32)
    d = yy-yr[None, :]
    a = np.clip(d+0.5, 0, 1)
    lo = NC.sample(xx*0.55+seed*3, yy*0.55+seed)
    shade = (1+facet*(lo-0.5)*1.2-0.10*smv(0, 0.9*amp, d))[..., None]
    colr = np.asarray(col, np.float32)[None, None, :]*shade
    if snow > 0:
        sl = (h[None, :]-snow*amp)*0.95+46*(NC.sample(xx*0.9+seed*5, yy*0.9)-0.5)*2
        sm_ = np.clip((sl-d)/7.0, 0, 1)[..., None]*(d > 0)[..., None]
        colr = colr*(1-sm_)+C(0.93, 0.95, 1.0)*(0.94+0.10*(lo[..., None]-0.5))*sm_
    warm = np.exp(-np.abs(xx-sun[0])/380.0)[..., None]*0.30
    colr = colr*(1-warm)+haze*warm
    mist = smv(base-0.25*amp, base+70, yy)[..., None]*haze_amt
    colr = colr*(1-mist)+haze*mist
    return colr.astype(np.float32), a.astype(np.float32)

def hill_layer(base, amp, seed, col, sun, fs=0.6, rim=0.5, trees=10.0, tex=0.25):
    xi = (np.arange(VW)*fs).astype(int) % 4096
    n = noise1d(4096, 2.8, seed)[xi]
    yr = base-amp*n
    tb = noise1d(4096, 0.9, seed+3)[(np.arange(VW)*2) % 4096]
    yr = yr-trees*np.clip(tb-0.45, 0, 1)*2*(0.5+noise1d(4096, 2.0, seed+9)[xi])
    yy, xx = np.mgrid[0:VH, 0:VW].astype(np.float32)
    d = yy-yr[None, :]
    a = np.clip(d+0.5, 0, 1)
    mott = (1-tex/2+tex*NA.sample(xx*0.9+seed*11, yy*1.6))[..., None]*(0.80+0.45*NC.sample(xx*0.35+seed*5, yy*0.9+seed))[..., None]
    colr = np.asarray(col, np.float32)[None, None, :]*mott
    colr = colr+np.exp(-np.abs(xx-sun[0])/420.0)[..., None]*np.exp(-np.clip(d, 0, 300)/90.0)[..., None]*C(0.55, 0.50, 0.20)*0.35
    dark = smv(0, 260, d)[..., None]
    colr = colr*(1-0.25*dark)
    rl = np.exp(-np.clip(d, 0, 100)/3.5)[..., None]*rim*(0.5+0.9*np.exp(-np.abs(xx-sun[0])/500.0))[..., None]
    colr = colr+rl*C(1.0, 0.93, 0.58)
    return colr.astype(np.float32), a.astype(np.float32), yr

class Vista:
    def __init__(self):
        self.sun = (M+0.585*W, M+0.455*H)
        sun = self.sun
        sky, ca = sky_layer(sun, seed=1, clear=(M+0.5*W, M+BAR+0.255*(H-2*BAR), 520.0, 190.0))
        # mountains
        m0 = mountain_layer(M+0.545*H, 330, 11, C(0.44, 0.55, 0.78), sun, snow=0.42, fs=0.70, haze_amt=0.50, facet=0.22)
        m1 = mountain_layer(M+0.568*H, 230, 23, C(0.28, 0.40, 0.60), sun, snow=0.62, fs=1.0, haze_amt=0.42, facet=0.20)
        m2 = mountain_layer(M+0.596*H, 160, 37, C(0.17, 0.29, 0.40), sun, snow=0.0, fs=1.3, haze_amt=0.36, facet=0.16, detail=5.0)
        occ = np.maximum(ca, np.maximum(m0[1], np.maximum(m1[1], m2[1])))
        sky = god_rays(sky, occ, sun, 0.30)
        self.sky = sky
        # merge far mountains into one RGBA layer (same parallax)
        far = np.zeros((VH, VW, 3), np.float32); fa = np.zeros((VH, VW), np.float32)
        for c, a in (m0, m1):
            far = far*(1-a[..., None])+c*a[..., None]; fa = fa*(1-a)+a
        self.far = (far*fa[..., None], fa)                # premultiplied
        # mid: m2 + lake + hills
        mid = m2[0]*m2[1][..., None]; ma = m2[1].copy()
        yy, xx = np.mgrid[0:VH, 0:VW].astype(np.float32)
        ly0, ly1 = M+0.618*H, M+0.668*H
        self.lake = (ly0, ly1)
        src_y = np.clip(ly0-(yy-ly0)*2.6, 0, VH-1)
        refl = cv2.remap(cv2.GaussianBlur(sky, (0, 0), sigmaX=9, sigmaY=3), xx, src_y, cv2.INTER_LINEAR)
        refl = refl*0.80+C(0.20, 0.34, 0.48)*0.20
        la = ((yy >= ly0-1) & (yy <= ly1+40)).astype(np.float32)
        mid = mid*(1-la[..., None])+refl*la[..., None]; ma = np.maximum(ma, la)
        h1 = hill_layer(M+0.640*H, 95, 5, C(0.33, 0.47, 0.30), sun, fs=0.5, rim=0.35, trees=12)
        # leave an opening for the lake in the centre-right
        gap = np.exp(-((xx-(M+0.60*W))/260.0)**2)*np.clip(1-(yy-ly0)/(ly1-ly0+30), 0, 1)
        a1 = h1[1]*np.clip(1-gap*1.6, 0, 1)*np.clip((yy-(h1[2][None, :]+gap*70))+0.5, 0, 1)
        mid = mid*(1-a1[..., None])+h1[0]*a1[..., None]; ma = np.maximum(ma, a1)
        self.mid = (mid, ma)
        h2 = hill_layer(M+0.705*H, 110, 15, C(0.25, 0.41, 0.21), sun, fs=0.4, rim=0.42, trees=16)
        h3 = hill_layer(M+0.745*H, 90, 29, C(0.18, 0.33, 0.15), sun, fs=0.33, rim=0.50, trees=20)
        near = h2[0]*h2[1][..., None]; na = h2[1].copy()
        near = near*(1-h3[1][..., None])+h3[0]*h3[1][..., None]; na = np.maximum(na, h3[1])
        self.near = (near, na)
        # foreground hill with crest
        x = np.arange(VW, dtype=np.float32)
        crest = (M+0.805*H)-70*np.exp(-((x-(M+0.50*W))/430.0)**2)+18*(noise1d(4096, 2.5, 41)[(x*0.5).astype(int)]-0.5)
        self.crest = crest
        d = yy-crest[None, :]
        fa2 = np.clip(d+0.5, 0, 1)
        blade = NB.sample(xx*3.2, yy*0.5)
        fcol = C(0.105, 0.225, 0.075)[None, None, :]*(0.75+0.5*blade)[..., None]
        fcol = fcol*(1-0.45*smv(0, 420, d))[..., None]
        rl = np.exp(-np.clip(d, 0, 200)/6.0)[..., None]*(0.6+0.9*np.exp(-np.abs(xx-sun[0])/520.0))[..., None]
        fcol = fcol+rl*C(0.85, 0.86, 0.40)*0.42+np.exp(-np.clip(d, 0, 400)/38.0)[..., None]*C(0.34, 0.44, 0.13)*0.42
        # flowers
        rng = np.random.default_rng(5)
        fl = np.zeros((VH, VW, 3), np.float32); fm = np.zeros((VH, VW), np.float32)
        cols = [C(1.0, 0.98, 0.92), C(1.0, 0.86, 0.25), C(1.0, 0.62, 0.74), C(0.80, 0.66, 1.0), C(1.0, 0.98, 0.92)]
        for _ in range(520):
            fx = rng.uniform(0, VW); cy_ = crest[int(fx)]
            fy = cy_+rng.uniform(0.02, 1.0)**1.4*(VH-cy_)
            dd = (fy-cy_)/(VH-cy_)
            rr = 1.2+5.2*dd
            c = cols[rng.integers(len(cols))]*(0.75+0.5*rng.uniform())
            cv2.circle(fl, (int(fx), int(fy)), int(rr), tuple(float(v) for v in c), -1, cv2.LINE_AA)
            cv2.circle(fm, (int(fx), int(fy)), int(rr), 1.0, -1, cv2.LINE_AA)
            if rr > 4: cv2.circle(fl, (int(fx), int(fy)), max(1, int(rr*0.35)), tuple(float(v) for v in C(1.0, 0.8, 0.2)), -1, cv2.LINE_AA)
        fm = fm[..., None]
        fcol = fcol*(1-fm)+fl*fm*0.9
        nb = smv(M+0.90*H, VH, yy)[..., None]
        fcol = fcol*(1-nb)+cv2.GaussianBlur(fcol, (0, 0), 6)*nb
        self.fg = (fcol*fa2[..., None], fa2)
        self.rng = np.random.default_rng(3)
        g = np.random.default_rng(8)
        self.glit = g.uniform(0, 1, (400, 4)).astype(np.float32)

    def frame(self, t, zoom, crane, chars_fn=None):
        """compose with parallax. zoom: 0..~0.08, crane: px. returns HDR BGR float (H, W, 3)"""
        def tf(p):
            s = 1+zoom*(0.25+0.75*p)
            cx, cy = VW/2, M+0.50*H
            return np.array([[s, 0, cx*(1-s)-M], [0, s, cy*(1-s)-M+crane*p]], np.float32)
        out = cv2.warpAffine(self.sky, tf(0.0), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        for (c, a), p in ((self.far, 0.12), (self.mid, 0.30), (self.near, 0.55)):
            Mx = tf(p)
            cw = cv2.warpAffine(c, Mx, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            aw = cv2.warpAffine(a, Mx, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[..., None]
            if p == 0.30:  # lake glitter under the sun (animated)
                ly0, ly1 = self.lake; g = self.glit
                sx = self.sun[0]
                lay = np.zeros((VH//2, VW//2), np.float32)
                for k in range(len(g)):
                    yy_ = ly0+g[k, 1]*(ly1-ly0+25)
                    spread = 40+150*(yy_-ly0)/(ly1-ly0)
                    xx_ = sx+(g[k, 0]-0.5)*2*spread*(0.3+0.7*g[k, 3])
                    b = 0.5+0.5*math.sin(t*(4+7*g[k, 2])+g[k, 3]*40)
                    if b > 0.55:
                        cv2.line(lay, (int((xx_-5)/2), int(yy_/2)), (int((xx_+5)/2), int(yy_/2)), float(2.2*(b-0.55)/0.45), 1, cv2.LINE_AA)
                lay = cv2.resize(lay, (VW, VH), interpolation=cv2.INTER_LINEAR)
                cw = cw+cv2.warpAffine(lay, Mx, (W, H), flags=cv2.INTER_LINEAR)[..., None]*C(1.0, 0.95, 0.8)*aw
            out = out*(1-aw)+cw*aw if p != 0.12 and p != 0.55 else out*(1-aw)+cw
            if p == 0.30: pass
        # atmosphere / haze over the distance
        yy = np.arange(H, dtype=np.float32)[:, None, None]
        hz = np.exp(-((yy-0.56*H)/(0.10*H))**2)*0.09
        out = out*(1-hz)+C(1.0, 0.90, 0.70)*hz*1.2
        fgc, fga = self.fg
        if chars_fn is not None:
            fgc = fgc.copy(); fga = fga.copy()
            chars_fn(fgc, fga)
        Mx = tf(1.0)
        cw = cv2.warpAffine(fgc, Mx, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        aw = cv2.warpAffine(fga, Mx, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[..., None]
        out = out*(1-aw)+cw
        sunp = (Mx[0, 0]*0+tf(0.0)[0, 0]*self.sun[0]+tf(0.0)[0, 2], tf(0.0)[1, 1]*self.sun[1]+tf(0.0)[1, 2])
        return out, sunp

def lens_flare(img, sun, strength=1.0):
    cx, cy = W/2, H/2
    lay = np.zeros((H//2, W//2, 3), np.float32)
    for k, rad, col, a in ((-0.35, 26, C(1.0, 0.7, 0.4), 0.10), (0.45, 44, C(0.4, 0.8, 1.0), 0.06), (0.9, 18, C(1.0, 0.9, 0.5), 0.10), (1.45, 70, C(0.6, 1.0, 0.7), 0.045), (1.9, 30, C(1.0, 0.6, 0.8), 0.06)):
        px = sun[0]+(cx-sun[0])*k*1.0; py = sun[1]+(cy-sun[1])*k*1.0
        cv2.circle(lay, (int(px/2), int(py/2)), int(rad/2), tuple(float(v*a*strength) for v in col), -1, cv2.LINE_AA)
    lay = cv2.GaussianBlur(lay, (0, 0), 2.0)
    img += cv2.resize(lay, (W, H), interpolation=cv2.INTER_LINEAR)
    return img

def birds(img, t, n=7, seed=3, y0=0.425, col=(0.10, 0.10, 0.14)):
    for j in range(n):
        bx = (0.15+0.09*j+0.035*t+0.02*math.sin(j*2.1))*W
        by = (y0+0.018*((j*37) % 5)-0.008*j-0.004*t+0.004*math.sin(t*1.3+j))*H
        s = 9+3*((j*13) % 3)
        fl = math.sin(t*9+j*1.7)*0.9
        pts = np.array([[bx-s, by-fl*s*0.55], [bx-s*0.45, by-fl*s*0.15-s*0.12], [bx, by], [bx+s*0.45, by-fl*s*0.15-s*0.12], [bx+s, by-fl*s*0.55]], np.int32)
        cv2.polylines(img, [pts], False, tuple(float(v) for v in C(*col)), 2, cv2.LINE_AA)

LUT_HOPE = make_lut(exposure=1.0, contrast=1.08, pivot=0.42, lift=(0.020, 0.014, 0.012), gamma=(0.98, 1.0, 1.02), gain=(0.96, 1.0, 1.04), shoulder=0.78)
def post_hope(fr, fi, vig=0.36, bloom_s=0.5, sat=1.12, fade=1.0, thr=0.98):
    fr = bloom(fr, thr, bloom_s, tint=C(1.0, 0.93, 0.80))
    fr = streak_flare(fr, 2.0, 0.22, tint=(1.0, 0.85, 0.65))
    fr = chroma_ab(fr, 1.0012)
    fr *= vignette(vig, 2.4)
    out = apply_lut(fr, LUT_HOPE, sat)
    if fade != 1.0: out *= fade
    return out

# ---------------------------------------------------------------- title
_HT = {}
def hope_title(frame, t):
    """'희망의 땅' elegant serif reveal. t: time since title start (draws into graded 0..1 frame)"""
    if t <= 0: return
    a_in = sm(0.0, 0.9, t)
    trk = lerp(0.34, 0.20, sm(0.0, 2.2, t)**0.7)
    key = round(trk*250)/250
    size = 168
    if key not in _HT:
        a = text_mask('희망의 땅', 'NotoSerifKR-VF.ttf', size, 700, key, space=0.30)
        _HT[key] = a
    a = _HT[key]
    blur = lerp(14, 0, sm(0.0, 0.9, t))
    if blur > 0.3: a = cv2.GaussianBlur(a, (0, 0), blur)
    h, w = a.shape
    g = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    fill = (C(1.0, 0.995, 0.97)*(1-g)+C(1.0, 0.90, 0.66)*g)*np.ones((1, w, 1), np.float32)
    bv = bevel(a, 3.0, (-0.4, -0.9), 0.8)[..., None]
    fill = np.clip(fill*(1+0.10*bv), 0, 1.3)
    # light sweep
    xs = np.arange(w, dtype=np.float32)[None, :, None]; ys = np.arange(h, dtype=np.float32)[:, None, None]
    sp = (t-1.0)/1.6
    band = np.exp(-((xs+ys*0.5-(w+h*0.5)*sp)/70.0)**2)
    fill = fill*(1+0.35*band)
    sc = lerp(1.06, 1.0, sm(0, 2.8, t))
    cx, cy = W/2, BAR+0.235*(H-2*BAR)
    sh = cv2.GaussianBlur(a, (0, 0), 14)
    place(frame, np.zeros((h, w, 3), np.float32)+C(0.05, 0.11, 0.24), cv2.GaussianBlur(a, (0, 0), 30), cx, cy+6, sc, opacity=0.55*a_in)
    place(frame, np.zeros((h, w, 3), np.float32)+C(0.06, 0.12, 0.25), sh, cx, cy+6, sc, opacity=0.60*a_in)
    place(frame, np.zeros((h, w, 3), np.float32)+C(0.08, 0.13, 0.26), cv2.GaussianBlur(a, (0, 0), 3.5), cx, cy+3, sc, opacity=0.70*a_in)
    place(frame, np.zeros((h, w, 3), np.float32)+C(1.0, 0.92, 0.72), cv2.GaussianBlur(a, (0, 0), 26), cx, cy, sc, additive=True, opacity=0.30*a_in)
    place(frame, fill, a, cx, cy, sc, opacity=a_in)
    s_in = sm(0.7, 1.5, t)
    if s_in > 0:
        sa = text_mask('THE LAND OF HOPE', 'Cinzel-VF.ttf', 42, 700, 0.42, space=0.50)
        sy = cy+size*0.70
        col = np.zeros(sa.shape+(3,), np.float32)+C(1.0, 0.97, 0.90)
        place(frame, col*0+C(0.10, 0.16, 0.26), cv2.GaussianBlur(sa, (0, 0), 4), W/2, sy+2, 1.0, opacity=0.8*s_in)
        place(frame, col, sa, W/2, sy, 1.0, opacity=s_in)
        ln = int(lerp(0, 270, s_in))
        for yl in (sy+40,):
            cv2.line(frame, (int(W/2-ln), int(yl)), (int(W/2+ln), int(yl)), tuple(float(v) for v in C(1.0, 0.96, 0.88)*0.9), 1, cv2.LINE_AA)

# ---------------------------------------------------------------- S5 vista
_V = None
def vista():
    global _V
    if _V is None: _V = Vista()
    return _V
LIGHT_S5 = dict(lights=[((0.0, 1.0, 0.2), rgb(.50, .56, .66), 'wrap', 1.5), ((0.05, 0.45, 1.0), rgb(1.5, 1.2, .8), 'lambert', 1.0)],
                amb=rgb(.17, .20, .24), rim=(rgb(1.7, 1.35, 0.8), 1.0, (0.0, -1.0)), head_gain=1.0)
S5_POSE = [dict(arms=('cheer', 'cheer', 0)), dict(arms=('cheer', 'fists', 0.0)), dict(arms=('shade', 'shade', 0)), dict(arms=('point', 'point', 0))]
def shot_s5(t, fi, dur=5.2):
    V = vista()
    zoom = 0.075*sm(-0.5, dur+0.5, t)
    crane = 26*sm(-0.5, dur+0.5, t)
    def chars_fn(fgc, fga):
        cam = Camera(F0, VW/2, 0, 52.0, 0.0)
        Zc = 600.0
        xs = [-88, -30, 30, 90]
        layers = []
        for j, i in enumerate([0, 1, 2, 3]):
            jump = 0.0
            if i == 1: jump = 16*abs(math.sin(t*4.2))
            if i == 0: jump = 10*abs(math.sin(t*3.6+1.0))
            st = dict(pos=(xs[j], Zc+(j % 2)*18), phase=0.0, run=0.0, yaw=math.pi, back=True, t=t, jump=jump, head_roll=0.05*math.sin(t*1.7+i),
                      expr=('joy', 0), **S5_POSE[i])
            cu, cv_, _ = cam.project((xs[j], 0, Zc+(j % 2)*18))
            camj = Camera(F0, VW/2, 0, 52.0, 0.0)
            L = draw_character(i, st, camj, LIGHT_S5, (VH, VW))
            gy = V.crest[int(np.clip(L['ground'][0], 0, VW-1))]+10
            L['y0'] += int(round(gy-L['ground'][1]))
            layers.append(L)
        for L in sorted(layers, key=lambda q: -q['z']):
            x0, y0 = L['x0'], L['y0']; h, w = L['alpha'].shape
            X0, Y0 = max(0, x0), max(0, y0); X1, Y1 = min(VW, x0+w), min(VH, y0+h)
            c = L['rgb'][Y0-y0:Y1-y0, X0-x0:X1-x0]; a = L['alpha'][Y0-y0:Y1-y0, X0-x0:X1-x0]
            fgc[Y0:Y1, X0:X1] = fgc[Y0:Y1, X0:X1]*(1-a[..., None])+c
            fga[Y0:Y1, X0:X1] = np.maximum(fga[Y0:Y1, X0:X1], a)
            # grass tufts over the feet
            rr = np.random.default_rng(100+int(x0))
            gx = L['ground'][0]; gyy = y0+(L['ground'][1]-(y0-int(round(V.crest[int(np.clip(L['ground'][0], 0, VW-1))]+10-L['ground'][1]))))
            base_y = V.crest[int(np.clip(gx, 0, VW-1))]+12
            for k in range(46):
                bx = gx+rr.uniform(-34, 34); hh = rr.uniform(8, 22); lean = rr.uniform(-5, 5)+2.5*math.sin(t*2+k)
                cv2.line(fgc, (int(bx), int(base_y+4)), (int(bx+lean), int(base_y-hh)), tuple(float(v) for v in (C(0.09, 0.20, 0.065) if k % 6 else C(0.42, 0.52, 0.20))), 2, cv2.LINE_AA)
    fr, sunp = V.frame(t, zoom, crane, chars_fn)
    birds(fr, t)
    fr = lens_flare(fr, sunp, 0.9)
    # light motes
    glow = np.zeros_like(fr)
    cam = Camera(F0, W/2, H*0.62, 60.0, 0.0)
    MOTES.draw(glow, cam, t, gain=0.8, color=(0.55, 0.85, 1.0), dt=1/24)
    fr += cv2.GaussianBlur(glow, (0, 0), 1.2)
    out = post_hope(fr, fi, fade=1.0)
    hope_title(out, t-0.55)
    fade = 1-sm(dur-0.85, dur-0.08, t)
    return out*fade
MOTES = Embers(160, 21, xr=380, zr=(60, 900), yr=(10, 260), rise=(4, 16))

# ---------------------------------------------------------------- S4 front view in the meadow
def grass_texture(n=1024, seed=4):
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    a = NA.sample(xx*0.5, yy*0.5); b = NC.sample(xx*0.25+30, yy*0.25); s = NB.sample(xx*3.0, yy*0.6)
    col = C(0.17, 0.36, 0.10)[None, None, :]*(1-a[..., None])+C(0.36, 0.56, 0.17)[None, None, :]*a[..., None]
    col = col*(0.72+0.5*b[..., None])*(0.82+0.36*s[..., None])
    rng = np.random.default_rng(seed)
    cols = [C(1.0, 0.98, 0.92), C(1.0, 0.86, 0.25), C(1.0, 0.62, 0.74), C(0.80, 0.66, 1.0), C(1.0, 0.98, 0.92), C(1.0, 0.9, 0.3)]
    for _ in range(420):
        x, y = rng.integers(8, n-8, 2); r = int(rng.integers(3, 7))
        cv2.circle(col, (int(x), int(y)), r, tuple(float(v) for v in cols[rng.integers(len(cols))]*(0.85+0.3*rng.uniform())), -1, cv2.LINE_AA)
    return (np.clip(col, 0, 1.5)).astype(np.float32)
_G = {}
def s4_assets():
    if 'tex' not in _G:
        _G['tex'] = pyramid(grass_texture(), 6)
        sun = (M+0.10*W, M+0.335*H)
        sky, ca = sky_layer(sun, seed=5, cloud_amt=0.8, horizon=0.56)
        m0 = mountain_layer(M+0.585*H, 300, 61, C(0.60, 0.69, 0.85), sun, snow=0.5, fs=0.6, haze_amt=0.7)
        m1 = mountain_layer(M+0.60*H, 200, 73, C(0.40, 0.52, 0.66), sun, snow=0.0, fs=0.9, haze_amt=0.6)
        occ = np.maximum(ca, np.maximum(m0[1], m1[1]))
        sky = god_rays(sky, occ, sun, 0.40)
        for c, a in (m0, m1): sky = sky*(1-a[..., None])+c*a[..., None]
        h1 = hill_layer(M+0.605*H, 120, 81, C(0.24, 0.40, 0.20), sun, fs=0.35, rim=0.6, trees=34)
        sky = sky*(1-h1[1][..., None])+h1[0]*h1[1][..., None]
        bd = sky[M:M+H, M:M+W]
        bd = cv2.GaussianBlur(bd, (0, 0), 7)
        _G['bd'] = bd; _G['sun'] = (sun[0]-M, sun[1]-M); _G['hrow'] = 0.605*H
    return _G
POLLEN = Embers(260, 31, xr=330, zr=(40, 900), yr=(5, 250), rise=(3, 18))
S4_X = [-75, -22, 24, 65]; S4_Z = [300, 258, 300, 262]
def shot_s4(t, fi, dur=3.6):
    G = s4_assets()
    cam = Camera(F0, W/2, H/2+30, 48.0, math.radians(-3.0))
    hy = cam.horizon()
    fr = np.zeros((H, W, 3), np.float32)
    dyb = int(round(hy-G['hrow']))
    Mx = np.float32([[1.02, 0, -0.01*W+6*math.sin(t*0.4)], [0, 1.02, dyb-0.01*H]])
    fr = cv2.warpAffine(G['bd'], Mx, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    sun = (G['sun'][0]*1.02-0.01*W, G['sun'][1]*1.02+dyb-0.01*H)
    # meadow floor
    y0 = floor_start(cam)
    tex, X, Z = sample_ground(G['tex'], cam, y0, H, 300.0, 300.0, 0.0, flip_v=False)
    ff = (1-np.exp(-(np.maximum(Z, 0)/900.0)**1.2))[..., None]
    back = np.exp(-np.maximum(Z, 0)/2500.0)[..., None]
    tex = tex*(0.80+0.25*(1-back))
    tex = tex*(1-ff)+C(0.80, 0.86, 0.50)*1.0*ff
    # warm sheen towards the sun
    sheen = np.exp(-((X+150-0.25*np.maximum(Z, 0))/(160+0.35*np.maximum(Z, 0)))**2)[..., None]*np.clip(Z/700.0, 0, 1)[..., None]
    tex = tex+sheen*C(1.0, 0.92, 0.55)*0.30
    fr[y0:] = tex
    yy = np.arange(H, dtype=np.float32)[:, None, None]
    b = cv2.GaussianBlur(fr, (0, 0), 3.0); mk = 1-smv(hy+40, hy+330, yy); fr = fr*(1-mk)+b*mk
    half = cv2.resize(fr, (W//2, H//2), interpolation=cv2.INTER_AREA)
    b = cv2.resize(cv2.GaussianBlur(half, (0, 0), 4.5), (W, H), interpolation=cv2.INTER_LINEAR); mk = smv(0.80*H, 0.99*H, yy); fr = fr*(1-mk)+b*mk
    # choreography
    w_run = 2*math.pi*1.55
    layers = []
    light = dict(HOPE_LIGHT)
    for i in range(4):
        stop = 1.10+0.10*i                      # when each one stops running
        run = 1-sm(stop-0.30, stop+0.10, t)
        zin = lerp(S4_Z[i]+330, S4_Z[i], sm(-0.2, stop, t)**0.9)
        ph = w_run*min(t, stop+0.1)+i*1.9
        cheer_t = 2.05+0.16*((i*3) % 4)
        cheer = sm(cheer_t, cheer_t+0.30, t)
        awe = sm(stop+0.1, stop+0.5, t)*(1-cheer)
        jt = t-cheer_t-0.12
        jump = 0.0
        if jt > 0:
            per = 0.62+0.05*i; k = (jt % per)/per
            jump = (20+5*(i % 2))*4*k*(1-k)*min(1.0, jt*3)
        crouch = 0.35*(1-run)*(1-sm(stop+0.15, stop+0.7, t))+0.25*math.exp(-((jt % 0.65)/0.12)**2)*(jt > 0)
        if cheer > 0.5: expr = ('joy', 1+2*cheer+0.6*math.sin(t*5+i))
        elif run > 0.5: expr = ('scream', lerp(4.5, 1.0, sm(0.2, stop, t)))
        else: expr = ('awe', 2.2*awe)
        armA = 'pump' if run > 0.5 else 'idle'
        armB = ['cheer', 'cheer', 'fists', 'cheer'][i]
        st = dict(pos=(S4_X[i]+2*math.sin(t+i), zin), phase=ph, run=run, lean=0.30, arms=(armA if cheer < 0.01 else 'idle', armB, cheer),
                  expr=expr, t=t, jump=jump, crouch=crouch, head_roll=0.08*math.sin(ph+0.6)*run+0.06*math.sin(t*1.3+i*2)*(1-run),
                  lean_extra=0.12*(1-run)*(1-sm(stop+0.15, stop+0.8, t))-0.06*awe)
        layers.append(draw_character(i, st, cam, light, fr.shape))
    for L in layers: char_shadow(fr, L, (sun[0]-140, hy-10), length=1.5, strength=0.42)
    ground_shadows(fr, layers, 0.30, 20.0, 5.0)
    # atmosphere glow behind the characters
    yy2, xx2 = np.mgrid[0:H, 0:W].astype(np.float32)
    rs = np.sqrt((xx2-sun[0])**2+(yy2-sun[1])**2)
    fr += (np.exp(-rs/420.0)[..., None]*C(1.0, 0.88, 0.62)*0.42)
    glow = np.zeros_like(fr)
    POLLEN.draw(glow, cam, t, zmin=330, gain=0.7, color=(0.6, 0.88, 1.0), dt=1/24)
    fr += cv2.GaussianBlur(glow, (0, 0), 1.5)
    for L in sorted(layers, key=lambda q: -q['z']): comp(fr, L)
    fr += (np.exp(-rs/300.0)[..., None]*C(1.0, 0.90, 0.66)*0.16)        # light wrap over subjects
    glow = np.zeros_like(fr)
    POLLEN.draw(glow, cam, t, zmax=330, gain=0.8, color=(0.6, 0.88, 1.0), dt=1/24)
    fr += cv2.GaussianBlur(glow, (0, 0), 2.5)
    fr = lens_flare(fr, sun, 1.2)
    dx = 2.0*vnoise(t*1.3, 21)*(1-sm(1.2, 1.8, t)*0.6); dy = 2.0*vnoise(t*1.1, 22)
    fr = shake2d(fr, dx, dy, 0.12*vnoise(t*0.9, 23), 1.02+0.012*sm(0.8, dur, t))
    out = post_hope(fr, fi, vig=0.34)
    white = 1-sm(0.0, 0.75, t)
    if white > 0: out = out*(1-white)+white*np.array([0.985, 0.99, 1.0], np.float32)
    return out

if __name__ == '__main__':
    import time
    which = sys.argv[1]; ts = [float(v) for v in sys.argv[2:]]
    fn = dict(S4=shot_s4, S5=shot_s5)[which]
    for t in ts:
        t0 = time.time(); fr = fn(t, int(t*24)); out = finish(fr, int(t*24))
        cv2.imwrite(f'still_{which}_{t:.2f}.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 93]); print(which, t, round(time.time()-t0, 2), 's')
