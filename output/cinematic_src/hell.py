"""Hell shots: S1 establishing (tilt-up reveal + title), S2 the escape (front), S3 run to the light (back)."""
import numpy as np, cv2, math, sys
from engine import *
from chars import Camera, draw_character, comp, rgb, HELL_LIGHT
FIRE = np.load('fire.npy', mmap_mode='r')
TILE_W = 260.0; TILE_H = TILE_W*1920/1080
F0 = 1500.0
HB = 1240; BH = 1300      # backdrop horizon row / height

def fire_frame(tv):
    n = FIRE.shape[0]; k = int(tv*30.0) % (2*(n-1)); k = k if k < n else 2*(n-1)-k
    return FIRE[k]

# ---------------------------------------------------------------- static backdrop parts
def _ridges():
    x = np.arange(W)
    n = noise1d(2048, 2.2, 3)[:W]; r = (1-np.abs(2*n-1))**1.6
    far = HB-50-190*r-22*noise1d(2048, 1.2, 9)[:W]
    spires = [(20, 640, 150), (150, 430, 110), (255, 250, 80), (1065, 700, 160), (940, 470, 115), (845, 280, 85), (420, 95, 70), (640, 80, 90), (540, 55, 120)]
    near = np.zeros(W)
    for cx, hh, ww in spires:
        near = np.maximum(near, hh*np.clip(1-np.abs(x-cx)/ww, 0, 1)**0.85)
    near = HB-12-near-26*noise1d(2048, 1.0, 21)[:W]*(0.4+np.clip((HB-12-near), 0, 1))
    near -= 14*noise1d(4096, 0.6, 5)[:W]
    return far.astype(np.float32), near.astype(np.float32)
RFAR, RNEAR = _ridges()
_yy = np.arange(BH, dtype=np.float32)[:, None]
MFAR = np.clip(_yy-RFAR[None, :], 0, 1)[..., None]      # 1 below the far ridge
MNEAR = np.clip(_yy-RNEAR[None, :], 0, 1)[..., None]
_t = np.clip((HB-_yy)/HB, 0, 1)
GRAD = (bgr(1.0, 0.40, 0.09)[None, None, :]*np.exp(-_t/0.085)[..., None]*1.45
        + bgr(0.42, 0.07, 0.03)[None, None, :]*np.exp(-_t/0.32)[..., None]
        + bgr(0.035, 0.012, 0.02)[None, None, :]).astype(np.float32)*np.ones((1, W, 1), np.float32)
_fx = noise1d(512, 2.0, 7)
FIRE_XMOD = np.interp(np.linspace(0, 511, 300), np.arange(512), 0.45+0.75*_fx).astype(np.float32)

def hell_backdrop(t, glow=1.0, fire_gain=1.0):
    """returns (BH, W, 3) HDR float BGR; horizon at row HB"""
    # smoke (low res)
    w4, h4 = W//4, BH//4
    xs = np.arange(w4, dtype=np.float32)[None, :]*np.ones((h4, 1), np.float32); ys = np.arange(h4, dtype=np.float32)[:, None]*np.ones((1, w4), np.float32)
    n1 = NA.sample(xs*0.9+t*6, ys*1.5-t*9); n2 = NB.sample(xs*2.3-t*10, ys*3.0-t*16+50)
    sm_ = 0.62*n1+0.38*n2
    sm_ = cv2.resize(sm_, (W, BH), interpolation=cv2.INTER_CUBIC)[..., None]
    bd = GRAD*glow*(0.50+0.95*sm_)
    bd = bd+bgr(0.55, 0.12, 0.035)*(np.clip(sm_-0.45, 0, 1)**2*3.0*np.exp(-_t/0.55)[..., None])*glow
    # far mountains
    bd = bd*(1-MFAR)+bgr(0.16, 0.04, 0.03)*MFAR*(0.6+0.8*np.exp(-(_yy-RFAR[None, :]).clip(0, 400)/160)[..., None])
    # fire wall between far ridge and near spires
    fh = 760; gw_, gh_ = 300, 200
    yb = np.linspace(0, 1, gh_, dtype=np.float32)[:, None]            # 0 top ... 1 bottom
    mk = np.clip(yb**1.05*1.0*FIRE_XMOD[None, :], 0, 0.86)*np.ones((1, gw_), np.float32)
    I = fire_field(gw_, gh_, t, mk, seed=1.0, sx=2.4, sy=1.15, speed=0.42, soft=0.50)
    fc = fire_ramp(I)*fire_gain*0.9
    fc = cv2.resize(fc, (W, fh), interpolation=cv2.INTER_CUBIC)
    bd[HB-fh:HB] += np.maximum(fc, 0)
    # near spires
    edge = np.clip(1-np.abs(_yy-RNEAR[None, :]-2.0)/3.0, 0, 1)[..., None]
    hz = np.exp(-np.clip(HB-_yy, 0, 2000)/150.0)[..., None]
    spc = bgr(0.030, 0.011, 0.014)*(1-hz)+bgr(0.55, 0.16, 0.05)*hz*0.8
    bd = bd*(1-MNEAR)+spc*MNEAR
    bd += edge*bgr(0.9, 0.30, 0.06)*0.8
    return bd.astype(np.float32)

CROP = (210, 1500)
def meteors(frame, t, hy, n=7, gain=1.0):
    lay = np.zeros((H//2, W//2, 3), np.float32)
    for j in range(n):
        per = 1.3+1.9*hash01(j, 1.0); ph = hash01(j, 2.0)
        a = (t/per+ph); k = a-math.floor(a); cyc = math.floor(a)
        x0 = W*(0.1+1.1*hash01(j*7+cyc, 3.0)); y0 = -80-200*hash01(j*5+cyc, 4.0)
        dx = -(0.25+0.5*hash01(j+cyc, 5.0)); dy = 1.0
        L = (hy+250-y0)
        x = x0+dx*L*k; y = y0+dy*L*k
        if y > hy+10: continue
        ln = 70+90*hash01(j, 6.0)
        c = bgr(1.0, 0.62, 0.22)*gain*(0.9+0.6*hash01(j, 8.0))
        for s_, w_ in ((1.0, 2), (0.55, 4)):
            cv2.line(lay, (int((x-dx*ln*s_)/2), int((y-dy*ln*s_)/2)), (int(x/2), int(y/2)), tuple(float(v) for v in c*(0.5 if w_ == 4 else 1.4)), w_, cv2.LINE_AA)
        cv2.circle(lay, (int(x/2), int(y/2)), 3, tuple(float(v) for v in bgr(1.0, 0.9, 0.6)*2.0*gain), -1, cv2.LINE_AA)
    lay = lay+cv2.GaussianBlur(lay, (0, 0), 3)*0.9
    frame += cv2.resize(lay, (W, H), interpolation=cv2.INTER_LINEAR)

def hell_floor(cam, tv, scroll, fog_z=1050.0, gain=1.0, xoff=0.0, crop=False):
    y0 = floor_start(cam)
    if y0 >= H: return None, y0, None, None
    fr = fire_frame(tv)
    if crop:
        fr = np.ascontiguousarray(fr[CROP[0]:CROP[1]])
        tex, X, Z = sample_ground(pyramid(fr, 5), cam, y0, H, TILE_W, TILE_W*(CROP[1]-CROP[0])/1080.0, scroll, xoff=xoff)
    else:
        tex, X, Z = sample_ground(pyramid(fr, 5), cam, y0, H, TILE_W, TILE_H, scroll, xoff=xoff)
    tex *= (1/255.0)
    L = tex.max(2, keepdims=True)
    tex = tex**1.10*1.12*gain
    tex *= (1+0.75*smv(0.50, 0.95, L))          # emissive boost of glowing coals
    ff = 1-np.exp(-(np.maximum(Z, 0)/fog_z)**1.3)
    fogc = bgr(1.0, 0.40, 0.10)*1.05
    tex = tex*(1-ff[..., None])+fogc*ff[..., None]
    return tex, y0, X, Z

def heat_haze(img, t, hy, amp=4.0):
    w4, h4 = W//4, H//4
    xs = np.arange(w4, dtype=np.float32)[None, :]*np.ones((h4, 1), np.float32); ys = np.arange(h4, dtype=np.float32)[:, None]*np.ones((1, w4), np.float32)
    dx = (NB.sample(xs*1.3, ys*1.3+t*85)-0.5)*2; dy = (NB.sample(xs*1.3+77, ys*1.3+t*85+33)-0.5)*2
    yy = np.arange(H, dtype=np.float32)[:, None]
    m = np.exp(-((yy-hy)/330.0)**2)*amp+0.8
    dx = cv2.resize(dx, (W, H), interpolation=cv2.INTER_LINEAR)*m; dy = cv2.resize(dy, (W, H), interpolation=cv2.INTER_LINEAR)*m
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return cv2.remap(img, (gx+dx).astype(np.float32), (gy+dy).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def geyser(frame, cam, pos, age, t, seed, width=150.0, height=430.0, gain=1.0):
    """fire column billboard at ground pos (X, Z). additive HDR."""
    if age < 0 or age > 2.6: return 0.0
    grow = sm(0, 0.22, age); die = 1-sm(1.5, 2.6, age)
    hh = height*(0.25+0.75*grow)*(0.55+0.45*die); ww = width*(0.7+0.3*grow)
    u, v, z = cam.project((pos[0], 0.0, pos[1]))
    if z < 30: return 0.0
    s = cam.f/z
    pw, ph = int(ww*s), int(hh*s)
    if pw < 6 or ph < 6: return 0.0
    gw, gh = 110, 230
    xn = (np.linspace(0, 1, gw, dtype=np.float32)[None, :]-0.5)*2
    yb = np.linspace(0, 1, gh, dtype=np.float32)[:, None]
    wdt = 0.30+0.50*yb**1.5                                   # column widens to the base
    mk = np.clip(np.exp(-(xn/wdt)**2)*(0.10+0.80*yb**0.8)*(0.6+0.4*die), 0, 0.84)
    I = fire_field(gw, gh, t, mk, seed=seed, sx=0.85, sy=1.5, speed=1.5, soft=0.52, warp=60)
    col = fire_ramp(I)*gain*(0.85+0.30*(1-sm(0, 0.5, age)))
    col = cv2.resize(col, (pw, ph), interpolation=cv2.INTER_CUBIC)
    col = np.maximum(cv2.GaussianBlur(col, (0, 0), max(0.6, 0.006*pw)), 0)
    x0 = int(u-pw/2); y0 = int(v-ph)
    X0, Y0 = max(0, x0), max(0, y0); X1, Y1 = min(W, x0+pw), min(H, y0+ph)
    if X1 <= X0 or Y1 <= Y0: return 0.0
    frame[Y0:Y1, X0:X1] += col[Y0-y0:Y1-y0, X0-x0:X1-x0]
    # ground glow
    gl = np.zeros((H//4, W//4), np.float32)
    cv2.ellipse(gl, (int(u/4), int(v/4)), (int(ww*s*0.55/4)+1, int(ww*s*0.12/4)+1), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    gl = cv2.resize(cv2.GaussianBlur(gl, (0, 0), 6), (W, H), interpolation=cv2.INTER_LINEAR)
    frame += gl[..., None]*bgr(1.0, 0.42, 0.10)*0.55*die*grow
    return grow*die

def ground_shadows(frame, layers, strength=0.6, ax=24.0, ay=6.5):
    m = np.zeros((H//2, W//2), np.float32)
    for L in layers:
        if L is None: continue
        gu, gv = L['ground']; s = L['spx']
        cv2.ellipse(m, (int(gu/2), int(gv/2)), (max(1, int(ax*s/2)), max(1, int(ay*s/2))), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    m = cv2.GaussianBlur(m, (0, 0), 5)
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    frame *= (1-strength*np.clip(m, 0, 1))

def dof(frame, hy, far_sigma=2.2, near_sigma=6.0, near_from=0.74):
    yy = np.arange(H, dtype=np.float32)[:, None, None]
    if far_sigma > 0:
        b = cv2.GaussianBlur(frame, (0, 0), far_sigma)
        mk = 1-smv(hy+30, hy+240, yy)
        frame = frame*(1-mk)+b*mk
    if near_sigma > 0:
        half = cv2.resize(frame, (W//2, H//2), interpolation=cv2.INTER_AREA)
        b = cv2.resize(cv2.GaussianBlur(half, (0, 0), near_sigma/2), (W, H), interpolation=cv2.INTER_LINEAR)
        mk = smv(near_from*H, 0.99*H, yy)
        frame = frame*(1-mk)+b*mk
    return frame

LUT_HELL = make_lut(exposure=1.0, contrast=1.16, pivot=0.36, lift=(0.016, 0.008, 0.004), gamma=(0.97, 1.0, 1.03), gain=(0.90, 0.99, 1.06), shoulder=0.70)
def post_hell(fr, fi, vig=0.62, bloom_s=0.55, sat=1.06, extra_white=0.0, fade=1.0, flare=0.35):
    fr = bloom(fr, 0.92, bloom_s)
    if flare: fr = streak_flare(fr, 1.35, flare, tint=(0.55, 0.75, 1.0))
    fr = chroma_ab(fr)
    fr *= vignette(vig, 2.2)
    if extra_white: fr = fr+extra_white
    out = apply_lut(fr, LUT_HELL, sat)
    if fade != 1.0: out *= fade
    return out

# ---------------------------------------------------------------- title
_TC = {}
def hell_title(frame, t, a_in):
    """draw '지옥' title, t: time since title start, a_in: 0..1 reveal"""
    if a_in <= 0: return
    size = 330
    trk = 0.02+0.05*sm(0, 2.2, t)
    key = round(trk, 3)
    if key not in _TC:
        a = text_mask('지옥', 'NotoSerifKR-VF.ttf', size, 900, trk)
        h, w = a.shape
        yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]; ys = np.where(a.max(1) > 0.5)[0]
        g = np.clip((np.arange(h, dtype=np.float32)[:, None]-ys.min())/max(1, ys.max()-ys.min()), 0, 1)
        top, mid, bot = bgr(1.0, 0.95, 0.66), bgr(1.0, 0.56, 0.12), bgr(0.72, 0.10, 0.03)
        fill = np.where(g[..., None] < 0.5, top+(mid-top)*(g[..., None]/0.5), mid+(bot-mid)*((g[..., None]-0.5)/0.5))*np.ones((1, w, 1), np.float32)
        xs = np.arange(w, dtype=np.float32)[None, :]*np.ones((h, 1), np.float32); yv = np.arange(h, dtype=np.float32)[:, None]*np.ones((1, w), np.float32)
        cr = NB.sample(xs*0.9, yv*0.9); cr2 = ND.sample(xs*2.2+31, yv*2.2)
        crack = np.clip(0.35+1.25*(0.65*cr+0.35*cr2), 0.3, 1.6)[..., None]
        bv = bevel(a, 5.0, (-0.45, -0.85), 1.1)[..., None]
        inner = cv2.GaussianBlur(a, (0, 0), 9)[..., None]         # hotter core
        colr = fill*crack*(0.80+0.55*bv)*(0.75+0.55*inner)+np.clip(bv, 0, 1)**2*bgr(1.0, 0.92, 0.7)*0.7
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dil = cv2.dilate(a, k)
        sh = cv2.GaussianBlur(dil, (0, 0), 20)
        _TC[key] = (a, colr.astype(np.float32), dil, sh)
    a, colr, dil, sh = _TC[key]
    fl = 1.0+0.10*vnoise(t*13, 4)+0.06*vnoise(t*31, 5)
    sc = lerp(1.22, 1.0, sm(0, 0.45, t))*(1.0+0.035*sm(0.3, 2.4, t))
    cx, cy = W/2, BAR+0.245*(H-2*BAR)
    place(frame, np.zeros_like(colr), sh, cx, cy+6, sc, opacity=0.60*a_in)
    place(frame, np.zeros_like(colr)+bgr(0.10, 0.012, 0.006), dil, cx, cy, sc, opacity=0.92*a_in)
    place(frame, colr*(1.75*fl), a, cx, cy, sc, opacity=a_in)
    if t < 0.5:   # impact flash
        f = (1-sm(0, 0.5, t))
        place(frame, colr*0+bgr(1.0, 0.8, 0.5), cv2.GaussianBlur(a, (0, 0), 12), cx, cy, sc, additive=True, opacity=1.6*f)
    # subtitle
    s_in = sm(0.35, 0.9, t)
    if s_in > 0:
        sa = text_mask('INFERNO', 'Cinzel-VF.ttf', 44, 600, 0.78)
        sy = cy+size*0.60*sc
        colr2 = np.zeros(sa.shape+(3,), np.float32)+bgr(1.0, 0.80, 0.56)*1.15
        place(frame, colr2*0, cv2.GaussianBlur(sa, (0, 0), 6), W/2, sy+2, 1.0, opacity=0.7*s_in)
        place(frame, colr2, sa, W/2, sy, 1.0, opacity=s_in)
        half = sa.shape[1]/2-30
        for sgn in (-1, 1):
            x0 = int(W/2+sgn*(half+18)); x1 = int(W/2+sgn*(half+18+110*s_in))
            cv2.line(frame, (x0, int(sy)), (x1, int(sy)), tuple(float(v) for v in bgr(1.0, 0.78, 0.5)*s_in), 2, cv2.LINE_AA)

# ---------------------------------------------------------------- S1
EMB = Embers(520, 1)
def shot_s1(t, fi, dur=3.6):
    k = sm(0.55, 2.55, t); k = k*k*(3-2*k) if False else k
    pitch = lerp(math.pi/2, math.radians(5.5), k**0.85)
    h0 = F0*TILE_W/W
    height = lerp(h0*lerp(1.0, 0.93, sm(0, 0.8, t)), 62.0, k)
    cam = Camera(F0, W/2, H/2, height, pitch)
    hy = cam.horizon() if pitch < 1.2 else -1e6
    fr = np.zeros((H, W, 3), np.float32)
    drift = 12*t
    if hy > -BH+50 and hy > 0:
        bd = hell_backdrop(t)
        y_top = int(round(hy))-HB
        s0, s1 = max(0, -y_top), min(BH, H-y_top)
        if s1 > s0: fr[s0+y_top:s1+y_top] = bd[s0:s1]
    fl, y0, X, Z = hell_floor(cam, t, scroll=drift+0.0*k, fog_z=lerp(4000, 1050, k))
    if fl is not None: fr[y0:] = fl
    if k > 0.02:
        fr = heat_haze(fr, t, hy if hy > -1e5 else 0, amp=4.0*k)
        glow = np.zeros_like(fr)
        EMB.draw(glow, cam, t, zflow=0.0, gain=1.0*k)
        fr += glow+cv2.GaussianBlur(glow, (0, 0), 5)*0.8
        fr = dof(fr, hy, 2.0*k, 5.0*k)
    hell_title(fr, t-1.95, sm(1.95, 2.3, t))
    fade = 0.22+0.78*sm(0.0, 0.6, t)
    return post_hell(fr, fi, vig=lerp(0.45, 0.62, k), fade=fade)

# ---------------------------------------------------------------- S2
def _tau_table(dur, slow=(1.05, 1.80), rate=0.30, ramp=0.14, n=4000):
    ts = np.linspace(0, dur, n)
    r = 1-(1-rate)*(smv(slow[0]-ramp, slow[0]+ramp, ts)-smv(slow[1]-ramp, slow[1]+ramp, ts))
    return ts, np.concatenate([[0], np.cumsum((r[1:]+r[:-1])/2*np.diff(ts))])
_S2T = _tau_table(4.0)
XA = [-72, -19, 72, 19]; ZA = [300, 226, 298, 232]
XB = [-19, -72, 19, 72]; ZB = [226, 300, 232, 298]
ARMS = [('pump', 'flail'), ('flail', 'flail'), ('pump', 'flail'), ('flail', 'pump')]
GEYS = [(0.10, 190, 700, 3.0, 170, 460), (0.98, -30, 560, 1.0, 240, 620), (1.95, -200, 620, 5.0, 170, 470), (2.55, 170, 600, 7.0, 180, 480), (3.15, -90, 680, 9.0, 170, 450)]
V_RUN = 380.0
def shot_s2(t, fi, dur=3.8):
    tau = float(np.interp(t, _S2T[0], _S2T[1]))
    slow = sm(0.95, 1.15, t)*(1-sm(1.75, 1.95, t))
    partB = t >= 1.9                      # hard cut to a second, lower angle with the other pair in front
    cam = Camera(F0, W/2, H/2+40, 46.0, math.radians(-3.5)) if not partB else Camera(F0, W/2, H/2+75, 38.0, math.radians(-5.5))
    hy = cam.horizon()
    fr = np.zeros((H, W, 3), np.float32)
    burst = 0.0
    bd = hell_backdrop(tau+(10 if not partB else 23), glow=1.15, fire_gain=1.35)
    y_top = int(round(hy))-HB
    s0, s1 = max(0, -y_top), min(BH, H-y_top)
    fr[s0+y_top:s1+y_top] = bd[s0:s1]
    meteors(fr, tau, hy)
    scroll = -V_RUN*tau+(140.0 if partB else 0.0)
    fl, y0, X, Z = hell_floor(cam, tau+0.5, scroll=scroll, fog_z=1000.0, crop=True, xoff=(60.0 if partB else 0.0))
    fr[y0:] = fl
    # motion blur on the floor (zoom towards vanishing point)
    mk = smv(hy+60, H, np.arange(H, dtype=np.float32)[:, None, None])*lerp(1.0, 0.35, slow)
    fr = zoom_blur(fr, (W/2, hy), amount=0.05, steps=5, mask=mk)
    fr = heat_haze(fr, tau, hy, amp=4.5)
    fr = dof(fr, hy, 2.4, 7.0, 0.76)
    # characters
    w_run = 2*math.pi*1.62
    states = []
    kz = kx = 1.0 if partB else 0.0
    for i in range(4):
        X_ = lerp(XA[i], XB[i], kx)+4.0*math.sin(1.3*tau+1.7*i)
        Z_ = lerp(ZA[i], ZB[i], kz)+9.0*math.sin(0.9*tau+2.1*i)
        ph = w_run*tau+i*1.9
        bl = 0.5+0.5*math.sin(0.8*tau+i*2.3); bl = sm(0.35, 0.65, bl)
        lvl = 3.3+1.7*(0.5+0.5*math.sin(2.4*tau+i*1.3)); lvl = lerp(lvl, 5.0, slow)
        states.append(dict(pos=(X_, Z_), phase=ph, run=1.0, lean=0.30, arms=(ARMS[i][0], ARMS[i][1], bl), expr=('scream', lvl), t=tau,
                           head_roll=0.10*math.sin(ph+0.6)+0.05*math.sin(0.7*tau+i), idx=i))
    # geysers (world anchored; recede with the floor)
    items = []
    for (t0, gx, gz, seed, gw, gh) in GEYS:
        tau0 = float(np.interp(t0, _S2T[0], _S2T[1])); age = tau-tau0
        items.append(('g', gz+V_RUN*0.55*max(age, 0), (gx, gz+V_RUN*0.55*max(age, 0)), age, seed, gw, gh))
    main_age = tau-float(np.interp(GEYS[1][0], _S2T[0], _S2T[1]))
    flash = 1.3*math.exp(-max(main_age, 0)*2.2) if main_age > 0 else 0.0
    light = dict(HELL_LIGHT); light['flicker'] = 0.92+0.10*vnoise(t*11, 1)+0.05*vnoise(t*27, 2)+flash*0.35
    layers = [draw_character(s['idx'], s, cam, light, fr.shape) for s in states]
    ground_shadows(fr, layers, 0.62)
    glow = np.zeros_like(fr)
    EMB.draw(glow, cam, tau, zflow=V_RUN*tau, zmin=430, flow_speed=V_RUN, gain=1.0)
    fr += glow+cv2.GaussianBlur(glow, (0, 0), 5)*0.8
    for L in layers: items.append(('c', L['z'], L))
    for it in sorted(items, key=lambda q: -q[1]):
        if it[0] == 'g': geyser(fr, cam, it[2], it[3], tau, it[4], it[5], it[6])
        else:
            # thin haze between depth layers so silhouettes separate
            comp(fr, it[2])
    glow = np.zeros_like(fr)
    EMB.draw(glow, cam, tau, zflow=V_RUN*tau, zmax=430, flow_speed=V_RUN, gain=1.1)
    fr += cv2.GaussianBlur(glow, (0, 0), 1.5)+cv2.GaussianBlur(glow, (0, 0), 7)*0.7
    # camera shake
    kick = 16*math.exp(-max(t-0.98, 0)*5.0) if t > 0.98 else 0.0
    dx = 5*vnoise(t*3.1, 1)+2.2*vnoise(t*9.3, 2)+kick*vnoise(t*37, 6)
    dy = 5*vnoise(t*2.7, 3)+2.2*vnoise(t*8.1, 4)+kick*vnoise(t*41, 7)
    fr = shake2d(fr, dx, dy, 0.35*vnoise(t*2.1, 5), 1.035)
    return post_hell(fr, fi, extra_white=0.03*flash, fade=1.0)

# ---------------------------------------------------------------- S3 (back view, towards the light)
_RAYS = None
def portal_rays(t, cx, cy, gain):
    global _RAYS
    if _RAYS is None:
        y, x = np.mgrid[0:H//4, 0:W//4].astype(np.float32)
        _RAYS = (x, y)
    x, y = _RAYS
    dx = x-cx/4; dy = (y-cy/4)
    ang = np.arctan2(dy, dx); r = np.sqrt(dx*dx+dy*dy)
    a1 = NB.sample((ang/np.pi*256+256+t*5)[...], np.full_like(ang, 13.0+t*9))
    a2 = ND.sample((ang/np.pi*512+t*-9), np.full_like(ang, 77.0+t*14))
    rays = np.clip(0.55*a1+0.45*a2-0.30, 0, 1)**1.6*3.2
    fall = 1/(1+(r/95.0)**1.25)
    out = rays*fall+1.4*np.exp(-r/42.0)+0.7*np.exp(-r/150.0)
    return cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR)*gain

def char_shadow(frame, L, src_pt, length=2.4, strength=0.72):
    """long cast shadow away from a light at screen point src_pt, using the character silhouette"""
    a = L['alpha']; h, w = a.shape
    gu, gv = L['ground']
    ys = np.where(a.max(1) > 0.3)[0]
    if len(ys) == 0: return
    top = ys.min(); bot = int(min(h-1, gv-L['y0']))
    if bot-top < 8: return
    xs = np.where(a[top:bot].max(0) > 0.3)[0]; xl, xr = xs.min(), xs.max()
    src = np.float32([[xl, bot], [xr, bot], [xr, top], [xl, top]])
    d = np.array([gu-src_pt[0], gv-src_pt[1]], np.float32); d /= (np.linalg.norm(d)+1e-6)
    Ls = (bot-top)*length
    wid = (xr-xl)
    k = (gv+Ls*d[1]-src_pt[1])/max(1.0, gv-src_pt[1])
    cxn = gu; far = np.array([gu+d[0]*Ls, gv+d[1]*Ls])
    cx0 = L['x0']+(xl+xr)/2
    dst = np.float32([[L['x0']+xl, gv], [L['x0']+xr, gv],
                      [far[0]+(cx0-gu)*k+wid*k/2, far[1]], [far[0]+(cx0-gu)*k-wid*k/2, far[1]]])
    M = cv2.getPerspectiveTransform(src, dst)
    sh = cv2.warpPerspective(a, M, (W, H), flags=cv2.INTER_LINEAR)
    sb = cv2.GaussianBlur(sh, (0, 0), 3)
    half = cv2.resize(sh, (W//2, H//2), interpolation=cv2.INTER_AREA)
    sb2 = cv2.resize(cv2.GaussianBlur(half, (0, 0), 9), (W, H), interpolation=cv2.INTER_LINEAR)
    yy = np.arange(H, dtype=np.float32)[:, None]
    far_m = smv(gv, gv+Ls*0.8, yy)
    s = sb*(1-far_m)+sb2*far_m
    frame *= (1-strength*(1-0.45*far_m)[..., None]*np.clip(s, 0, 1)[..., None])

LIGHT_S3 = dict(lights=[((0, -1, -0.2), rgb(0.55, .22, .07), 'wrap', 1.6), ((0.0, 0.25, 1.0), rgb(1.6, 1.35, 1.0), 'lambert', 1.0)],
                amb=rgb(.05, .03, .03), rim=(rgb(3.2, 2.6, 1.8), 1.0, None), head_gain=1.0)
EMB3 = Embers(380, 7)
def shot_s3(t, fi, dur=2.6):
    cam = Camera(F0, W/2, H/2+40, 52.0, math.radians(1.0))
    hy = cam.horizon()
    px, py = W/2+10*vnoise(t*0.8, 3), hy
    grow = t/dur
    fr = np.zeros((H, W, 3), np.float32)
    bd = hell_backdrop(t+30, glow=0.55, fire_gain=0.7)
    y_top = int(round(hy))-HB
    s0, s1 = max(0, -y_top), min(BH, H-y_top)
    fr[s0+y_top:s1+y_top] = bd[s0:s1]
    # canyon walls with a tall crack of light
    yy = np.arange(H, dtype=np.float32)[:, None]; xx = np.arange(W, dtype=np.float32)[None, :]
    top = hy-lerp(760, 1050, grow)
    hfrac = np.clip((hy-yy)/(hy-top), 0, 1)
    halfw = lerp(34, 120, grow**1.4)*(1-hfrac**1.7)**0.9
    jag = (noise1d(4096, 1.6, 12)[(np.arange(H)*3) % 4096][:, None]-0.5)*46*(0.3+hfrac)
    jag2 = (noise1d(4096, 1.6, 31)[(np.arange(H)*3) % 4096][:, None]-0.5)*46*(0.3+hfrac)
    left = px-halfw+jag; right = px+halfw+jag2
    gap = np.clip((xx-left)/2.0, 0, 1)*np.clip((right-xx)/2.0, 0, 1)*(yy < hy+1)*(yy > top)
    wall = (yy < hy+1).astype(np.float32)*(1-gap)
    rock = 0.6+0.8*NA.sample(xx*0.6+np.zeros_like(yy), yy*0.6+np.zeros_like(xx))
    wallcol = bgr(0.050, 0.020, 0.022)*rock[..., None]
    lit = np.exp(-np.abs(xx-px)/lerp(150, 300, grow))[..., None]*bgr(0.9, 0.55, 0.28)*0.55*rock[..., None]   # light spilling on the walls
    fr = fr*(1-wall[..., None]*0.93)+(wallcol+lit)*wall[..., None]*0.93
    core = bgr(1.0, 0.96, 0.86)*lerp(3.2, 6.0, grow)
    fr = fr*(1-gap[..., None])+core*gap[..., None]
    # floor
    tau = t
    V = 360.0
    fl, y0, X, Z = hell_floor(cam, t+4.0, scroll=V*tau, fog_z=900.0, gain=0.85, crop=True)
    path = np.exp(-(X/(40+0.12*np.maximum(Z, 0)))**2)*(0.25+0.75*np.clip(Z/900, 0, 1))
    fl = fl+path[..., None]*bgr(1.0, 0.85, 0.6)*lerp(0.55, 1.5, grow)
    ffw = (1-np.exp(-(np.maximum(Z, 0)/700.0)**1.5))[..., None]*np.exp(-(X/420.0)**2)[..., None]
    fl = fl*(1-0.8*ffw)+bgr(1.0, 0.9, 0.7)*lerp(1.6, 3.0, grow)*0.8*ffw
    fr[y0:] = fl
    mk = smv(hy+60, H, yy[..., None])
    fr = zoom_blur(fr, (px, hy), amount=0.05, steps=5, mask=mk)
    fr = heat_haze(fr, t, hy, amp=3.0)
    fr = dof(fr, hy, 1.6, 6.5, 0.78)
    # characters running away from camera
    w_run = 2*math.pi*1.55
    Xs = [-66, -22, 24, 68]; Zs = [330, 300, 322, 296]
    order = [0, 1, 2, 3]
    light = dict(LIGHT_S3); light['flicker'] = 1.0
    layers = []
    for j, i in enumerate(order):
        ph = w_run*tau+i*1.7
        st = dict(pos=(Xs[j]+3*math.sin(1.1*t+i), Zs[j]+62*grow**1.2+6*math.sin(0.9*t+2*i)), phase=ph, run=1.0, lean=0.30, yaw=math.pi,
                  arms=('pump', 'flail', 0.0 if i % 2 == 0 else 0.35), expr=('scream', 0), t=t, back=True, head_roll=0.07*math.sin(ph))
        layers.append(draw_character(i, st, cam, light, fr.shape))
    for L in layers: char_shadow(fr, L, (px, hy), length=2.6, strength=0.78)
    rays = portal_rays(t, px, hy-lerp(250, 330, grow), lerp(0.55, 1.05, grow))
    fr += rays[..., None]*bgr(1.0, 0.86, 0.62)*0.55
    for L in sorted(layers, key=lambda q: -q['z']): comp(fr, L)
    fr += rays[..., None]*bgr(1.0, 0.86, 0.62)*0.22          # light wrap / atmosphere in front
    glow = np.zeros_like(fr)
    EMB3.draw(glow, cam, t, zflow=-V*tau, flow_speed=-V, gain=0.9, color=(0.35, 0.75, 1.0))
    fr += glow+cv2.GaussianBlur(glow, (0, 0), 5)*0.8
    dx = 4*vnoise(t*3.3, 11)+2*vnoise(t*9.1, 12); dy = 4*vnoise(t*2.9, 13)+2*vnoise(t*8.3, 14)
    fr = shake2d(fr, dx, dy, 0.3*vnoise(t*2.0, 15), 1.03)
    white = sm(dur-0.85, dur-0.05, t)
    out = post_hell(fr, fi, vig=lerp(0.55, 0.25, white), bloom_s=1.0+0.8*white, sat=lerp(1.02, 0.85, white), flare=0.6)
    if white > 0: out = out*(1-white)+white*np.array([0.985, 0.99, 1.0], np.float32)
    return out

if __name__ == '__main__':
    import time
    which = sys.argv[1]; ts = [float(v) for v in sys.argv[2:]]
    fn = dict(S1=shot_s1, S2=shot_s2, S3=shot_s3)[which]
    for t in ts:
        t0 = time.time(); fr = fn(t, int(t*24)); out = finish(fr, int(t*24))
        cv2.imwrite(f'still_{which}_{t:.2f}.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 93]); print(which, t, round(time.time()-t0, 2), 's')
