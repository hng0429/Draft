"""Stylised 3D-ish characters: capsule G-buffer body + relit photo head billboard."""
import numpy as np, cv2, pickle, math

HEADS = pickle.load(open('heads.pkl', 'rb'))
HEAD_UNITS = 32.0     # world units for the 300px face height
SS = 2

def nz(v):
    v = np.asarray(v, np.float64); n = np.linalg.norm(v); return v/n if n > 1e-9 else v

class Camera:
    def __init__(self, f, cx, cy, height, pitch):
        self.f, self.cx, self.cy, self.h, self.pitch = f, cx, cy, height, pitch
        c, s = math.cos(pitch), math.sin(pitch)
        self.R = np.array([1.0, 0, 0]); self.U = np.array([0, c, s]); self.F = np.array([0, -s, c])
        self.C = np.array([0, height, 0.0])
    def project(self, P):
        d = np.asarray(P, np.float64)-self.C
        x = d@self.R; y = d@self.U; z = d@self.F
        return self.cx+self.f*x/z, self.cy-self.f*y/z, z
    def horizon(self):
        return self.cy-self.f*math.tan(self.pitch)
    def to_screen_dir(self, l):
        l = nz(l); return np.array([l@self.R, -(l@self.U), -(l@self.F)])

# ---------------------------------------------------------------- appearance
def rgb(r, g, b): return np.array([b, g, r], np.float32)   # store BGR
LOOKS = [
 dict(scale=1.05, jacket=rgb(.055, .055, .065), jacket2=None, gloss=0.55, quilt=7, collar=rgb(.05, .05, .06), cuff=None, pants=rgb(.10, .10, .12), shoes=rgb(.70, .70, .68), hood=False, prop=None),
 dict(scale=1.02, jacket=rgb(.40, .27, .20), jacket2=None, gloss=0.12, quilt=0, collar=rgb(.86, .77, .60), cuff=rgb(.86, .77, .60), pants=rgb(.17, .24, .38), shoes=rgb(.72, .72, .70), hood=False, prop='racket'),
 dict(scale=0.95, jacket=rgb(.07, .07, .08), jacket2=rgb(.40, .40, .39), gloss=0.05, quilt=0, collar=rgb(.42, .22, .78), cuff=None, pants=rgb(.06, .06, .07), shoes=rgb(.72, .72, .71), hood=False, prop='ball'),
 dict(scale=0.94, jacket=rgb(.085, .11, .21), jacket2=None, gloss=0.06, quilt=0, collar=rgb(.075, .10, .19), cuff=None, pants=rgb(.07, .07, .08), shoes=rgb(.72, .72, .70), hood=True, prop=None),
]

def lerp(a, b, t): return a+(b-a)*t
def smooth(a, b, x):
    t = min(1.0, max(0.0, (x-a)/(b-a))); return t*t*(3-2*t)

# ---------------------------------------------------------------- pose
def arm_dirs(style, s, ph, t):
    """returns (upper dir, fore dir) in local coords (x lateral outward for side s, y up, z forward)"""
    if style == 'pump':
        a = -1.0*math.sin(ph if s > 0 else ph+math.pi)*-1   # opposite to same-side leg
        a = -a
        up = np.array([s*0.20, -math.cos(a), math.sin(a)])
        b = a+1.55
        fo = np.array([-s*0.12, -math.cos(b), math.sin(b)])
    elif style == 'flail':
        up = np.array([s*(0.60+0.22*math.sin(1.0*ph+s)), 0.70+0.12*math.sin(0.5*ph+0.5*s), 0.30])
        fo = np.array([s*(0.10+0.55*math.sin(1.5*ph+1.3*s)), 0.90, 0.20+0.30*math.cos(1.0*ph+s)])
    elif style == 'cheer':
        up = np.array([s*(0.42+0.06*math.sin(6*t+s)), 0.90, 0.12])
        fo = np.array([s*(0.22+0.10*math.sin(7*t+2*s)), 0.96, 0.10])
    elif style == 'fists':   # fists pumped at shoulder height
        up = np.array([s*0.55, -0.35, 0.45]); fo = np.array([s*0.05, 0.95, 0.25+0.1*math.sin(9*t+s)])
    elif style == 'knees':   # hands on knees, panting
        up = np.array([s*0.25, -0.80, 0.55]); fo = np.array([-s*0.10, -0.85, 0.50])
    elif style == 'shade':   # one hand shading eyes (s>0), other relaxed
        if s > 0: up = np.array([s*0.55, 0.25, 0.60]); fo = np.array([-s*0.80, 0.55, 0.15])
        else: up = np.array([s*0.18, -0.97, 0.05]); fo = np.array([s*0.10, -0.97, 0.20])
    elif style == 'point':   # one arm pointing forward-up
        if s > 0: up = np.array([s*0.25, 0.25, 0.93]); fo = np.array([s*0.10, 0.30, 0.95])
        else: up = np.array([s*0.18, -0.97, 0.05]); fo = np.array([s*0.10, -0.97, 0.20])
    else:                     # relaxed
        up = np.array([s*0.16, -0.98, 0.04+0.05*math.sin(1.5*t+s)]); fo = np.array([s*0.08, -0.97, 0.22])
    return nz(up), nz(fo)

def build_pose(st):
    """st: dict(phase, run, lean, arms=(styleA, styleB, blend), jump, t, crouch) -> joints (local, units)"""
    ph = st.get('phase', 0.0); run = st.get('run', 1.0); t = st.get('t', 0.0)
    lean = st.get('lean', 0.30)*run + st.get('lean_extra', 0.0)
    crouch = st.get('crouch', 0.0)
    J = {}
    bounce = 3.2*(-math.cos(2*ph))*run
    pel = np.array([1.3*math.sin(ph)*run, 44.0+bounce-crouch*9, 0.0])
    pyaw = 0.14*math.sin(ph)*run
    hipW, shW = 8.0, 16.5
    feet_min = 1e9
    for s, name in ((1, 'L'), (-1, 'R')):
        phs = ph if s > 0 else ph+math.pi
        th = (0.22+0.98*math.sin(phs))*run + crouch*0.55
        kf = (0.30+1.75*(0.5+0.5*math.cos(phs-0.55))**1.6)*run + (1-run)*0.06 + crouch*1.0
        hip = pel+np.array([s*hipW*math.cos(pyaw), 0, s*hipW*math.sin(pyaw)])
        knee = hip+23*nz([s*0.07, -math.cos(th), math.sin(th)])
        be = th-kf
        ank = knee+23*nz([0, -math.cos(be), math.sin(be)])
        fa = be+1.72
        toe = ank+12.0*nz([s*0.10, -math.cos(fa), math.sin(fa)])
        J['hip'+name], J['knee'+name], J['ank'+name], J['toe'+name] = hip, knee, ank, toe
        feet_min = min(feet_min, ank[1]-5.2, toe[1]-4.6)
    up = max(0.0, -feet_min) if run > 0.5 else -feet_min
    jump = st.get('jump', 0.0)
    for k in J: J[k] = J[k]+np.array([0, up+jump, 0])
    pel = pel+np.array([0, up+jump, 0]); J['pel'] = pel
    sp = nz([0, math.cos(lean), math.sin(lean)])
    belly = pel+sp*8; chest = pel+sp*30; J['belly'], J['chest'] = belly, chest
    syaw = -0.38*math.sin(ph)*run
    sroll = 0.05*math.sin(ph)*run
    A, B, bl = st.get('arms', ('pump', 'pump', 0.0))
    for s, name in ((1, 'L'), (-1, 'R')):
        sh = chest+sp*4+np.array([s*shW*math.cos(syaw), s*shW*sroll, s*shW*math.sin(syaw)])
        u1, f1 = arm_dirs(A, s, ph, t)
        if bl > 0:
            u2, f2 = arm_dirs(B, s, ph, t); u1 = nz(lerp(u1, u2, bl)); f1 = nz(lerp(f1, f2, bl))
        el = sh+19*u1; ha = el+17.5*f1
        J['sh'+name], J['el'+name], J['ha'+name] = sh, el, ha
    neck = chest+sp*8.5; J['neck'] = neck
    hb = 1.4*math.cos(2*ph-0.9)*run
    J['head'] = neck+sp*4.5+np.array([0, hb, 0])
    return J

# ---------------------------------------------------------------- rasteriser
class GBuf:
    def __init__(self, x0, y0, w, h):
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.z = np.full((h, w), 1e9, np.float32)
        self.n = np.zeros((h, w, 3), np.float32); self.n[..., 2] = 1
        self.alb = np.zeros((h, w, 3), np.float32)
        self.gloss = np.zeros((h, w), np.float32)
        self.flag = np.zeros((h, w), np.uint8)     # 1: may occlude head, 2: emissive
    def capsule(self, a, b, ra, rb, col, gloss=0.0, flag=0, col2=None, split=0.5, stripes=0, zscale=1.0):
        """a, b: (u, v, z) screen px (already *SS and ROI-relative) + depth; radii in px"""
        ax, ay, az = a; bx, by, bz = b
        R = max(ra, rb)+2
        x0 = int(max(0, math.floor(min(ax, bx)-R))); x1 = int(min(self.w, math.ceil(max(ax, bx)+R)))
        y0 = int(max(0, math.floor(min(ay, by)-R))); y1 = int(min(self.h, math.ceil(max(ay, by)+R)))
        if x1 <= x0 or y1 <= y0: return
        X = np.arange(x0, x1, dtype=np.float32)[None, :]+0.5; Y = np.arange(y0, y1, dtype=np.float32)[:, None]+0.5
        dx, dy = bx-ax, by-ay; L2 = dx*dx+dy*dy
        if L2 < 1e-4: t = np.zeros((y1-y0, x1-x0), np.float32)
        else: t = np.clip(((X-ax)*dx+(Y-ay)*dy)/L2, 0, 1).astype(np.float32)
        r = ra+(rb-ra)*t
        px = X-(ax+t*dx); py = Y-(ay+t*dy)
        d2 = px*px+py*py
        ins = d2 < r*r
        if not ins.any(): return
        h = np.sqrt(np.maximum(r*r-d2, 0))
        z = az+(bz-az)*t-h*zscale
        zb = self.z[y0:y1, x0:x1]
        m = ins & (z < zb)
        if not m.any(): return
        zb[m] = z[m]
        nn = self.n[y0:y1, x0:x1]
        ir = 1.0/np.maximum(r, 1e-3)
        nn[..., 0][m] = (px*ir)[m]; nn[..., 1][m] = (py*ir)[m]; nn[..., 2][m] = (h*ir)[m]
        al = self.alb[y0:y1, x0:x1]
        if col2 is not None:
            c = np.where((t < split)[..., None], np.asarray(col, np.float32), np.asarray(col2, np.float32))
        else:
            c = np.broadcast_to(np.asarray(col, np.float32), (y1-y0, x1-x0, 3))
        if stripes:
            sv = 0.5+0.5*np.cos(2*np.pi*t*stripes)
            c = c*(1-0.45*np.clip((sv-0.86)/0.14, 0, 1))[..., None]
        al[m] = c[m]
        self.gloss[y0:y1, x0:x1][m] = gloss
        self.flag[y0:y1, x0:x1][m] = flag

def shade(g, lights, amb, rim=None, ao_k=0.07, ao_sig=7.0, rim_w=6.0):
    """lights: list of (dir_screen, color(BGR), mode, power); returns premultiplied BGR float + alpha (at SS)"""
    valid = g.z < 1e8
    n = g.n
    col = np.zeros_like(g.alb)
    diff = np.zeros_like(g.alb); spec = np.zeros_like(g.alb)
    for L, c, mode, p in lights:
        ndl = n[..., 0]*L[0]+n[..., 1]*L[1]+n[..., 2]*L[2]
        if mode == 'wrap': d = (0.5+0.5*ndl)**p
        else: d = np.maximum(ndl, 0)**p
        diff += d[..., None]*np.asarray(c, np.float32)
        hv = nz(np.asarray(L)+np.array([0, 0, 1.0]))
        sp = np.maximum(n[..., 0]*hv[0]+n[..., 1]*hv[1]+n[..., 2]*hv[2], 0)**28
        spec += (sp*g.gloss)[..., None]*np.asarray(c, np.float32)
    zv = np.where(valid, g.z, 0).astype(np.float32); wv = valid.astype(np.float32)
    zb = cv2.GaussianBlur(zv, (0, 0), ao_sig)/np.maximum(cv2.GaussianBlur(wv, (0, 0), ao_sig), 1e-3)
    ao = np.clip(1-ao_k*np.maximum(g.z-zb, 0), 0.35, 1)
    col = g.alb*(np.asarray(amb, np.float32)+diff)*ao[..., None]+spec*0.6
    if rim is not None:
        rc, rp, rdir = rim
        f = np.clip(1-n[..., 2], 0, 1)**rp
        dist = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 3)
        f = f*np.exp(-dist/(rim_w*SS))
        if rdir is not None:
            f = f*np.clip(0.30+0.70*(n[..., 0]*rdir[0]+n[..., 1]*rdir[1]), 0, 1)
        col += f[..., None]*np.asarray(rc, np.float32)*(0.45+0.55*np.clip(g.alb.mean(2, keepdims=True)*3+0.25, 0, 1))
    col *= wv[..., None]
    return col, wv

def shade_head(e, lights, amb, rim, roll=0.0, gain=1.0):
    """e: expression dict; returns BGR float (0..1+) and alpha, in sprite space"""
    rgbf = e['rgb'].astype(np.float32)/255.0
    n = e['nrm'].astype(np.float32)/127.0
    al = e['alpha'].astype(np.float32)/255.0
    cr, sr = math.cos(roll), math.sin(roll)
    diff = np.zeros_like(rgbf)
    for L, c, mode, p in lights:
        Lx = L[0]*cr+L[1]*sr; Ly = -L[0]*sr+L[1]*cr          # rotate light into sprite space
        ndl = n[..., 0]*Lx+n[..., 1]*Ly+n[..., 2]*L[2]
        d = (0.5+0.5*ndl)**p if mode == 'wrap' else np.maximum(ndl, 0)**p
        diff += d[..., None]*np.asarray(c, np.float32)
    col = rgbf*gain*(np.asarray(amb, np.float32)+diff)
    if rim is not None:
        rc, rp, rdir = rim
        ed = e['edge'].astype(np.float32)/4.0
        f = np.exp(-ed/5.0)
        if rdir is not None:
            rx = rdir[0]*cr+rdir[1]*sr; ry = -rdir[0]*sr+rdir[1]*cr
            nl = np.hypot(n[..., 0], n[..., 1])+1e-4
            f = f*np.clip(0.30+0.70*(n[..., 0]*rx+n[..., 1]*ry)/nl, 0, 1)
        col += f[..., None]*np.asarray(rc, np.float32)*(0.55+0.45*rgbf.mean(2, keepdims=True)*2)
    return col, al

# ---------------------------------------------------------------- character assembly
def draw_character(idx, st, cam, light, frame_shape, mirror_back=False):
    """Returns dict(x0,y0, rgb(premult float), alpha, feet(u,v), headbox) for compositing.
       st: state dict: pos=(X,Z), yaw (0 faces camera), plus pose params, expr=(name, level), head_roll
       light: dict(lights=[(world_dir,color,mode,p)], amb, rim=(color,power,world_dir2d or None), head_gain, flicker)"""
    look = LOOKS[idx]; sc = look['scale']*st.get('scale', 1.0)
    J = build_pose(st)
    yaw = st.get('yaw', 0.0)+math.pi      # local +z (forward) -> world -z (toward camera) when yaw=0
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    X0, Z0 = st['pos']
    def W(p):
        x, y, z = p[0]*sc, p[1]*sc, p[2]*sc
        return np.array([X0+x*cy_+z*sy_, y, Z0-x*sy_+z*cy_])
    JW = {k: W(v) for k, v in J.items()}
    P = {k: cam.project(v) for k, v in JW.items()}
    pxu = lambda z: cam.f/z
    # ROI
    us = [p[0] for p in P.values()]; vs = [p[1] for p in P.values()]
    zc = P['pel'][2]; s_px = pxu(zc)*sc
    hu_, hv_ = P['head'][0], P['head'][1]
    x0 = int(math.floor(min(min(us)-24*s_px, hu_-46*s_px))); x1 = int(math.ceil(max(max(us)+24*s_px, hu_+46*s_px)))
    y0 = int(math.floor(min(min(vs)-12*s_px, hv_-82*s_px))); y1 = int(math.ceil(max(vs)+9*s_px))
    x0c, y0c = max(x0, -400), max(y0, -600); x1c, y1c = min(x1, frame_shape[1]+400), min(y1, frame_shape[0]+500)
    w, h = x1c-x0c, y1c-y0c
    if w <= 4 or h <= 4: return None
    g = GBuf(x0c, y0c, w*SS, h*SS)
    def S(name): 
        u, v, z = P[name]; return ((u-x0c)*SS, (v-y0c)*SS, z)
    def R(r, name): return r*sc*pxu(P[name][2])*SS
    def cap(a, b, ra, rb, col, **kw):
        A, B = S(a), S(b)
        zs = P[a][2]/cam.f/SS
        g.capsule(A, B, R(ra, a), R(rb, b), col, zscale=zs, **kw)
    def cap_pts(pa, pb, ra, rb, col, **kw):
        ua, va, za = cam.project(pa); ub, vb, zb = cam.project(pb)
        g.capsule(((ua-x0c)*SS, (va-y0c)*SS, za), ((ub-x0c)*SS, (vb-y0c)*SS, zb), ra*sc*pxu(za)*SS, rb*sc*pxu(zb)*SS, col, zscale=za/cam.f/SS, **kw)
    skin = (HEADS[idx]['skin']/255.0*0.92).astype(np.float32)
    jk, pants, shoes = look['jacket'], look['pants'], look['shoes']
    gl = look['gloss']
    # legs
    for sd in ('L', 'R'):
        cap('hip'+sd, 'knee'+sd, 8.4, 6.7, pants, gloss=0.04)
        cap('knee'+sd, 'ank'+sd, 6.4, 5.0, pants, gloss=0.04)
        cap('ank'+sd, 'toe'+sd, 5.2, 4.6, shoes, gloss=0.25)
    # torso (boxy winter jacket): two vertical lobes + shoulder bar + hem
    upv = nz(JW['chest']-JW['pel']); latv0 = nz(JW['hipL']-JW['hipR']); lats = nz(JW['shL']-JW['shR'])
    cap('hipL', 'hipR', 8.4, 8.4, pants, gloss=0.04)
    j2 = look['jacket2']
    for sgn in (1, -1):
        lo_ = JW['pel']+upv*3.5*sc+latv0*sgn*5.6*sc; hi_ = JW['chest']+lats*sgn*6.6*sc
        if j2 is not None: cap_pts(lo_, hi_, 10.9, 11.8, j2, col2=jk, split=0.50, gloss=gl)
        else: cap_pts(lo_, hi_, 10.9, 11.8, jk, gloss=gl, stripes=look['quilt']*0.5)
    cap_pts(JW['pel']+upv*2.2*sc+latv0*6.2*sc, JW['pel']+upv*2.2*sc-latv0*6.2*sc, 10.5, 10.5, jk if j2 is None else j2, gloss=gl)
    lsh = JW['chest']+(JW['shL']-JW['chest'])*0.66; rsh = JW['chest']+(JW['shR']-JW['chest'])*0.66
    cap_pts(lsh+upv*2.0*sc, rsh+upv*2.0*sc, 10.6, 10.6, jk, gloss=gl)
    # neck + collar
    cap('chest', 'neck', 5.0, 4.6, skin)
    up = nz(JW['neck']-JW['chest'])
    cb = JW['chest']+up*(4.6 if idx == 1 else 6.0)*sc
    if look['collar'] is not None:
        latv = nz(JW['shL']-JW['shR'])
        fw = nz(np.cross(latv, up))
        ccol = look['collar']; cr_ = 8.6 if idx == 1 else 6.4; cw = 4.8 if idx == 1 else 3.0
        ring = [cb+(latv*math.cos(a)+fw*math.sin(a))*cr_*sc+up*(0.6*math.cos(a*2))*sc for a in np.linspace(0, 2*math.pi, 11)]
        for a_, b_ in zip(ring[:-1], ring[1:]): cap_pts(a_, b_, cw, cw, ccol, gloss=0.02)
    if look['hood']:
        bk = -nz(np.cross(nz(JW['shL']-JW['shR']), up))
        cap_pts(cb+bk*6*sc+up*1*sc+nz(JW['shL']-JW['shR'])*6*sc, cb+bk*6*sc+up*1*sc-nz(JW['shL']-JW['shR'])*6*sc, 6.0, 6.0, jk)
    # arms
    for sd in ('L', 'R'):
        cap('sh'+sd, 'el'+sd, 6.6, 5.5, jk, gloss=gl, flag=1, stripes=look['quilt']*0.35)
        cap('el'+sd, 'ha'+sd, 5.4, 4.5, jk, gloss=gl, flag=1, stripes=look['quilt']*0.35)
        if look['cuff'] is not None:
            wr = JW['el'+sd]+(JW['ha'+sd]-JW['el'+sd])*0.78
            cap_pts(wr, JW['ha'+sd]-(JW['ha'+sd]-JW['el'+sd])*0.04, 5.6, 5.6, look['cuff'], flag=1)
        hd = JW['ha'+sd]+nz(JW['ha'+sd]-JW['el'+sd])*3.0*sc
        cap_pts(hd, hd, 4.9, 4.9, skin, flag=1)
    # props
    if look['prop'] == 'ball':
        hd = JW['haL']+nz(JW['haL']-JW['elL'])*6.5*sc
        cap_pts(hd, hd, 3.6, 3.6, rgb(.80, .92, .16), flag=1, gloss=0.1)
    if look['prop'] == 'racket':
        f = nz(JW['haR']-JW['elR']); hd = JW['haR']+f*3*sc
        side = nz(np.cross(f, np.array([0.2, 0.3, 1.0])))
        blk = rgb(.05, .05, .05); lime = rgb(.78, .92, .10)
        cap_pts(hd-f*4*sc, hd+f*10*sc, 1.7, 1.4, rgb(.9, .9, .9), flag=1)
        c0 = hd+f*25*sc; A_, B_ = 14.0, 10.5
        ringp = [c0+(f*A_*math.cos(a)+side*B_*math.sin(a))*sc for a in np.linspace(0, 2*math.pi, 21)]
        for k, (a_, b_) in enumerate(zip(ringp[:-1], ringp[1:])):
            cap_pts(a_, b_, 1.5, 1.5, lime if (k % 10) in (2, 3, 7, 8) else blk, flag=1, gloss=0.3)
        cap_pts(hd+f*10*sc, ringp[9], 1.1, 1.1, blk, flag=1); cap_pts(hd+f*10*sc, ringp[11], 1.1, 1.1, blk, flag=1)
        for q in np.linspace(-0.75, 0.75, 6):
            hx = math.sqrt(1-q*q)
            cap_pts(c0+(f*A_*q-side*B_*hx)*sc, c0+(f*A_*q+side*B_*hx)*sc, 0.28, 0.28, rgb(.8, .8, .78), flag=1)
            cap_pts(c0+(side*B_*q-f*A_*hx)*sc, c0+(side*B_*q+f*A_*hx)*sc, 0.28, 0.28, rgb(.8, .8, .78), flag=1)
    # ---- shading
    fl = light.get('flicker', 1.0)
    Ls = [(cam.to_screen_dir(d), np.asarray(c, np.float32)*(fl if i == 0 else 1.0), m, p) for i, (d, c, m, p) in enumerate(light['lights'])]
    rim = None
    if light.get('rim') is not None:
        rc, rp, rd = light['rim']
        rim = (np.asarray(rc, np.float32)*fl, rp, None if rd is None else np.array([rd[0], rd[1]]))
    col, a = shade(g, Ls, light['amb'], rim, rim_w=max(1.6, 1.15*s_px))
    bodyz = g.z; flag = g.flag
    def down(x): return cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA)
    col_s = down(col); a_s = down(a)
    front = down(((flag == 1) & (bodyz < P['head'][2]-3.0)).astype(np.float32))   # arm pixels in front of head plane
    # ---- head billboard
    H = HEADS[idx]; name, lvl = st.get('expr', ('scream', 0))
    lst = H['expr'][name]; e = lst[int(max(0, min(len(lst)-1, round(lvl))))]
    hu, hv, hz = P['head']
    hs = HEAD_UNITS/H['FH']*pxu(hz)*sc*st.get('head_scale', 1.0)
    roll = st.get('head_roll', 0.0)
    back = st.get('back', False)
    hcol, hal = shade_head(e, Ls, light['amb'], rim, roll=roll, gain=light.get('head_gain', 1.0))
    if back:      # seen from behind: no face, hair-coloured silhouette
        hair = np.percentile(e['rgb'].reshape(-1, 3)[e['alpha'].reshape(-1) > 200], 12, axis=0).astype(np.float32)/255.0
        n = e['nrm'].astype(np.float32)/127.0
        base = np.broadcast_to(hair, hcol.shape).copy()
        dsum = np.zeros_like(base)
        for L, c, m, p in Ls:
            ndl = n[..., 0]*L[0]+n[..., 1]*L[1]+n[..., 2]*L[2]
            dsum += ((0.5+0.5*ndl)**p if m == 'wrap' else np.maximum(ndl, 0)**p)[..., None]*c
        hcol2 = base*(np.asarray(light['amb'], np.float32)+dsum)
        if rim is not None:
            ed = e['edge'].astype(np.float32)/4.0
            hcol2 += np.exp(-ed/7.5)[..., None]*rim[0]*0.9
        hcol = hcol2
    ax_, ay_ = H['anchor']
    cr, sr = math.cos(roll)*hs, math.sin(roll)*hs
    M = np.array([[cr, -sr, (hu-x0c)-(cr*ax_-sr*ay_)], [sr, cr, (hv-y0c)-(sr*ax_+cr*ay_)]], np.float64)
    if back: # mirror horizontally about the anchor
        M = np.array([[-cr, -sr, (hu-x0c)-(-cr*ax_-sr*ay_)], [-sr, cr, (hv-y0c)-(-sr*ax_+cr*ay_)]], np.float64)
    prem = np.dstack([hcol*hal[..., None], hal])
    hw = cv2.warpAffine(prem, M, (w, h), flags=cv2.INTER_AREA if hs < 0.6 else cv2.INTER_LINEAR, borderValue=0)
    if hs < 0.5: hw = cv2.GaussianBlur(hw, (0, 0), 0.3/hs*0.5) if False else hw
    hc, ha = hw[..., :3], hw[..., 3]
    # contact shadow of the head on the shoulders/neck
    sh = cv2.GaussianBlur(ha, (0, 0), max(1.0, 5*s_px))
    shm = np.clip(np.roll(sh, int(6*s_px), axis=0)*0.55, 0, 0.6)
    col_s = col_s*(1-shm[..., None]*(1-front)[..., None])
    out = col_s*(1-ha[..., None])+hc
    oa = a_s*(1-ha)+ha
    fm = (front*a_s)[..., None]
    out = out*(1-fm)+col_s*front[..., None]
    oa = np.maximum(oa, front*a_s)
    fu = (P['toeL'][0]+P['toeR'][0]+P['ankL'][0]+P['ankR'][0])/4; 
    ground = cam.project(np.array([X0, 0, Z0]))
    return dict(x0=x0c, y0=y0c, rgb=out, alpha=oa, ground=(ground[0], ground[1]), z=zc, spx=s_px,
                head=(hu, hv, hs), height=J['pel'][1])

def comp(dst, layer):
    """composite premultiplied layer dict onto float frame dst (in place)"""
    x0, y0 = layer['x0'], layer['y0']; h, w = layer['alpha'].shape
    X0, Y0 = max(0, x0), max(0, y0); X1, Y1 = min(dst.shape[1], x0+w), min(dst.shape[0], y0+h)
    if X1 <= X0 or Y1 <= Y0: return
    c = layer['rgb'][Y0-y0:Y1-y0, X0-x0:X1-x0]; a = layer['alpha'][Y0-y0:Y1-y0, X0-x0:X1-x0, None]
    dst[Y0:Y1, X0:X1] = dst[Y0:Y1, X0:X1]*(1-a)+c

HELL_LIGHT = dict(lights=[((0, -1, -0.35), rgb(1.45, .60, .17), 'wrap', 1.7), ((0.3, 0.2, 1.0), rgb(.9, .35, .08), 'lambert', 1.0)],
                  amb=rgb(.17, .08, .06), rim=(rgb(1.7, .78, .22), 1.3, None), head_gain=1.0)
HOPE_LIGHT = dict(lights=[((0.0, 1.0, -0.25), rgb(.62, .66, .74), 'wrap', 1.5), ((-0.45, 0.6, 0.75), rgb(1.6, 1.2, .75), 'lambert', 1.0), ((0.25, 0.1, -1.0), rgb(.30, .26, .22), 'wrap', 1.0)],
                  amb=rgb(.22, .25, .31), rim=(rgb(2.6, 1.9, 1.0), 1.2, (-0.45, -0.9)), head_gain=1.05)

if __name__ == '__main__':
    import time
    Wd, Hd = 1080, 1920
    for nm, light, bg in (('hell', HELL_LIGHT, (0.05, 0.03, 0.06)), ('hope', HOPE_LIGHT, (0.75, 0.62, 0.45))):
        fr = np.zeros((Hd, Wd, 3), np.float32); fr[:] = bg
        cam = Camera(1500, 540, 900, 62, 0.02)
        t0 = time.time()
        specs = [(-52, 420, 0.4, 'pump', ('scream', 5)), (-17, 330, 2.0, 'flail', ('scream', 4)), (20, 400, 3.5, 'pump', ('scream', 3)), (55, 345, 5.0, 'flail', ('scream', 5))]
        if nm == 'hope': specs = [(-52, 420, 0.0, 'cheer', ('joy', 3)), (-17, 330, 0, 'fists', ('joy', 2)), (20, 400, 0, 'idle', ('awe', 2)), (55, 345, 0, 'cheer', ('joy', 3))]
        layers = []
        for i, (x, z, ph, arms, ex) in enumerate(specs):
            st = dict(pos=(x, z), phase=ph, run=1.0 if nm == 'hell' else 0.0, arms=(arms, arms, 0), expr=ex, t=1.0, head_roll=0.08*math.sin(ph), jump=0 if nm == 'hell' else (14 if i == 0 else 0))
            layers.append(draw_character(i, st, cam, light, fr.shape))
        for L in sorted(layers, key=lambda L: -L['z']): comp(fr, L)
        print(nm, 'time', round(time.time()-t0, 2))
        cv2.imwrite(f'chars_{nm}.jpg', np.clip(fr*255, 0, 255).astype(np.uint8)[300:1700], [cv2.IMWRITE_JPEG_QUALITY, 92])
