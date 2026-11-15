"""Shared rendering utilities: noise, perspective floor, fire, particles, post, text."""
import numpy as np, cv2, math
from PIL import Image, ImageDraw, ImageFont
W, H = 1080, 1920
FPS = 24
BAR = 112

def sm(a, b, x):
    t = min(1.0, max(0.0, (x-a)/(b-a))) if b != a else float(x >= b); return t*t*(3-2*t)
def smv(a, b, x):
    t = np.clip((x-a)/(b-a), 0, 1); return t*t*(3-2*t)
def lerp(a, b, t): return a+(b-a)*t
def bgr(r, g, b): return np.array([b, g, r], np.float32)

# ------------------------------------------------------------------ noise
def fft_noise(n, beta=2.0, seed=0, lo=1, hi=None):
    rng = np.random.default_rng(seed)
    fx = np.fft.fftfreq(n)[:, None]*n; fy = np.fft.rfftfreq(n)[None, :]*n
    k = np.sqrt(fx*fx+fy*fy); k[0, 0] = 1
    amp = 1/k**(beta/2); amp[k < lo] = 0
    if hi: amp[k > hi] = 0
    amp[0, 0] = 0
    f = amp*np.exp(1j*rng.uniform(0, 2*np.pi, k.shape))
    x = np.fft.irfft2(f, s=(n, n))
    x = (x-x.min())/(x.max()-x.min())
    return x.astype(np.float32)
def pad_tile(t): return np.pad(t, ((0, 1), (0, 1)), mode='wrap')
class Tile:
    def __init__(self, n, beta, seed, lo=1, hi=None):
        self.n = n; self.t = pad_tile(fft_noise(n, beta, seed, lo, hi))
    def sample(self, mx, my):
        mx, my = np.broadcast_arrays(np.asarray(mx, np.float32), np.asarray(my, np.float32))
        return cv2.remap(self.t, np.ascontiguousarray(np.mod(mx, self.n), np.float32), np.ascontiguousarray(np.mod(my, self.n), np.float32), cv2.INTER_LINEAR)
NA = Tile(512, 2.4, 11); NB = Tile(512, 1.7, 23); NC = Tile(512, 3.0, 37); ND = Tile(256, 2.0, 5)
def noise1d(n, beta=2.0, seed=0, lo=1):
    rng = np.random.default_rng(seed)
    k = np.fft.rfftfreq(n)*n; k[0] = 1
    amp = 1/k**(beta/2); amp[k < lo] = 0; amp[0] = 0
    x = np.fft.irfft(amp*np.exp(1j*rng.uniform(0, 2*np.pi, k.shape)), n)
    return ((x-x.min())/(x.max()-x.min())).astype(np.float32)
def hash01(i, s=0.0):
    x = math.sin(i*127.1+s*311.7)*43758.5453; return x-math.floor(x)
def vnoise(t, seed=0):
    """smooth 1D value noise in -1..1"""
    i = math.floor(t); f = t-i; f = f*f*(3-2*f)
    return lerp(hash01(i, seed), hash01(i+1, seed), f)*2-1

# ------------------------------------------------------------------ perspective ground
def ground_coords(cam, y0, y1):
    v = np.arange(y0, y1, dtype=np.float32)[:, None]+0.5; u = np.arange(W, dtype=np.float32)[None, :]+0.5
    c, s = math.cos(cam.pitch), math.sin(cam.pitch)
    ry = c*(cam.cy-v)/cam.f-s
    rz = s*(cam.cy-v)/cam.f+c
    t = -cam.h/np.minimum(ry, -1e-5)
    X = t*(u-cam.cx)/cam.f
    Z = t*rz
    return X, Z, t
def floor_start(cam):
    if cam.pitch > 1.2: return 0
    return int(max(0, min(H, math.ceil(cam.horizon())+1)))

def sample_ground(pyr, cam, y0, y1, tile_w, tile_h, scroll, flip_v=True, xoff=0.0):
    """pyr: list of float32/uint8 images (level0 TW x TH, each next half). returns float32 (rows, W, 3), X, Z"""
    X, Z, t = ground_coords(cam, y0, y1)
    TH, TW = pyr[0].shape[:2]
    tu = TW*(0.5+(X+xoff)/tile_w)
    tv = TH*(0.5-(Z+scroll)/tile_h) if flip_v else TH*((Z+scroll)/tile_h)
    tu = TW-np.abs(np.mod(tu, 2*TW)-TW)
    tv = TH-np.abs(np.mod(tv, 2*TH)-TH)
    tv = np.broadcast_to(tv, tu.shape)
    lat = (t[:, 0]/cam.f)*TW/tile_w
    zz = Z[:, 0]; dz = np.abs(np.gradient(zz)) if len(zz) > 1 else np.zeros_like(zz)
    ver = dz*TH/tile_h
    lod = np.log2(np.maximum(np.maximum(lat, ver), 1e-3))
    lod = np.clip(lod, 0, len(pyr)-1)
    out = np.zeros((y1-y0, W, 3), np.float32)
    for L in range(len(pyr)):
        wgt = np.clip(1-np.abs(lod-L), 0, 1)
        rows = np.nonzero(wgt > 0)[0]
        if len(rows) == 0: continue
        r0, r1 = rows[0], rows[-1]+1
        sc = 1.0/(2**L)
        mx = np.ascontiguousarray(tu[r0:r1]*sc-0.5, np.float32); my = np.ascontiguousarray(tv[r0:r1]*sc-0.5, np.float32)
        smp = cv2.remap(pyr[L], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        out[r0:r1] += smp.astype(np.float32)*wgt[r0:r1, None, None]
    return out, X, Z

def pyramid(img, levels=5):
    p = [img]
    for _ in range(levels-1):
        p.append(cv2.resize(p[-1], (max(1, p[-1].shape[1]//2), max(1, p[-1].shape[0]//2)), interpolation=cv2.INTER_AREA))
    return p

# ------------------------------------------------------------------ fire
def fire_ramp(I):
    """intensity (0..1) -> BGR HDR fire colour (dark red -> orange -> yellow -> white core)"""
    I = np.clip(I, 0, 1.0)
    r = np.clip(I*2.4, 0, 1.0)**0.85*(0.9+0.9*I)
    g = np.clip(I*1.75-0.30, 0, 1.2)**1.35*(0.85+0.7*I)
    b = np.clip(I*1.6-0.95, 0, 1.0)**1.8*1.3
    return np.dstack([b, g, r]).astype(np.float32)

def fire_field(w, h, t, mask, seed=0.0, sx=1.0, sy=1.0, speed=1.0, soft=0.34, warp=42.0):
    """flame intensity (h, w), y=0 at the top.  mask (h, w) 0..1 says how much fire is allowed at each point:
       where it is high the flame is solid, where it falls off only noise peaks survive -> tongues and licks."""
    xs = np.arange(w, dtype=np.float32)[None, :]; ys = np.arange(h, dtype=np.float32)[:, None]
    u = xs*(256.0/w)*sx+seed*91.0+np.zeros_like(ys)
    v = ys*(256.0/h)*sy+np.zeros_like(xs)
    wx = (NB.sample(u*0.6+40, v*0.6+t*95*speed)-0.5)*warp
    wy = (NC.sample(u*0.5+9, v*0.5+t*70*speed)-0.5)*warp*0.6
    n1 = NA.sample(u+wx, v+wy+t*150*speed+seed*17)
    n2 = NB.sample(u*2.2+wx*1.4, v*1.9+t*250*speed+seed*5)
    n3 = ND.sample(u*4.6+wx*2.0, v*3.4+t*380*speed)
    n = 0.54*n1+0.34*n2+0.12*n3
    n = np.clip((n-0.22)/0.56, 0, 1)
    I = np.clip((n-(1.0-mask))/soft, 0, 1)
    return cv2.GaussianBlur(I, (0, 0), 0.8)

# ------------------------------------------------------------------ particles
class Embers:
    """closed-form 3D ember particles living in a looping volume in front of the camera"""
    def __init__(self, n=420, seed=1, xr=420, zr=(40, 1500), yr=(0, 330), rise=(25, 95)):
        r = np.random.default_rng(seed)
        self.n = n; self.xr = xr; self.zr = zr; self.yr = yr
        self.x0 = r.uniform(-xr, xr, n); self.z0 = r.uniform(zr[0], zr[1], n)
        self.life = r.uniform(1.6, 4.5, n); self.ph = r.uniform(0, 1, n)
        self.vy = r.uniform(rise[0], rise[1], n); self.wob = r.uniform(4, 22, n); self.wf = r.uniform(1.2, 4.0, n)
        self.size = r.uniform(0.5, 1.6, n)**1.6; self.heat = r.uniform(0.5, 1.0, n)
    def state(self, t, zflow=0.0, wind=0.0):
        a = (t/self.life+self.ph)
        age = (a-np.floor(a))
        cyc = np.floor(a)
        jx = np.sin(cyc*12.9898+self.ph*78.233)*43758.5453; jx = jx-np.floor(jx)
        x = (self.x0+(jx-0.5)*200+self.wob*np.sin(self.wf*t+self.ph*40)+wind*age*self.life)
        y = self.yr[0]+self.vy*age*self.life
        span = self.zr[1]-self.zr[0]
        z = self.zr[0]+np.mod(self.z0-self.zr[0]+zflow, span)
        fade = np.clip(age*8, 0, 1)*np.clip((1-age)*2.2, 0, 1)
        return x, y, z, fade
    def draw(self, layer, cam, t, zflow=0.0, wind=0.0, dt=1/48, gain=1.0, zmin=0, zmax=1e9, flow_speed=0.0, color=(0.18, 0.62, 1.0)):
        x, y, z, fade = self.state(t, zflow, wind)
        x2, y2, z2, _ = self.state(t-dt, zflow-flow_speed*dt, wind)
        for i in range(self.n):
            if not (zmin <= z[i] < zmax) or fade[i] <= 0.02: continue
            u, v, zc = cam.project((x[i], y[i], z[i]))
            if zc < 12: continue
            u2, v2, zc2 = cam.project((x2[i], y2[i], z2[i]))
            if zc2 < 12: continue
            if not (-50 < u < W+50 and -50 < v < H+50): continue
            rad = self.size[i]*cam.f/zc*1.25
            k = fade[i]*self.heat[i]*gain
            if rad > 7:          # near: soft bokeh disc
                kk = k*0.30*min(1.0, 9.0/rad)
                cv2.circle(layer, (int(u), int(v)), int(min(rad, 46)), (color[0]*kk*1.6, color[1]*kk*1.2, color[2]*kk), -1, cv2.LINE_AA)
            else:
                th = max(1, int(round(rad*1.6)))
                c = (color[0]*k*2.2, color[1]*k*2.0, color[2]*k*2.4)
                cv2.line(layer, (int(u2), int(v2)), (int(u), int(v)), c, th, cv2.LINE_AA)

# ------------------------------------------------------------------ post
_GRAIN = None
def grain(frame_idx, amount=0.03):
    global _GRAIN
    if _GRAIN is None:
        r = np.random.default_rng(99); _GRAIN = []
        for k in range(5):
            g = r.normal(0, 1, (H//2+8, W//2+8)).astype(np.float32)
            g2 = cv2.resize(g, (W+16, H+16), interpolation=cv2.INTER_CUBIC)
            f = r.normal(0, 1, (H+16, W+16)).astype(np.float32)
            _GRAIN.append((0.75*g2+0.55*cv2.GaussianBlur(f, (0, 0), 0.7)))
    g = _GRAIN[frame_idx % 5]
    ox, oy = (frame_idx*7) % 16, (frame_idx*13) % 16
    g = g[oy:oy+H, ox:ox+W]
    if (frame_idx//5) % 2: g = g[:, ::-1]
    return g*amount

def bloom(img, thr=0.80, strength=0.85, wide=1.0, tint=None):
    small = cv2.resize(img, (W//4, H//4), interpolation=cv2.INTER_AREA)
    hi = np.maximum(small-thr, 0)
    b1 = cv2.GaussianBlur(hi, (0, 0), 2.5)
    b2 = cv2.GaussianBlur(hi, (0, 0), 9)
    s16 = cv2.resize(hi, (W//16, H//16), interpolation=cv2.INTER_AREA)
    b3 = cv2.resize(cv2.GaussianBlur(s16, (0, 0), 7), (W//4, H//4), interpolation=cv2.INTER_LINEAR)
    g = b1*0.55+b2*0.40+b3*0.45*wide
    if tint is not None: g = g*tint
    img += cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)*strength
    return img

def streak_flare(img, thr=1.25, strength=0.5, tint=(1.0, 0.75, 0.55)):
    small = cv2.resize(img, (W//8, H//8), interpolation=cv2.INTER_AREA)
    hi = np.maximum(small.max(2)-thr, 0)
    st = cv2.GaussianBlur(hi, (0, 0), sigmaX=34, sigmaY=0.8)+cv2.GaussianBlur(hi, (0, 0), sigmaX=12, sigmaY=0.6)
    st = cv2.resize(st, (W, H), interpolation=cv2.INTER_LINEAR)
    img += st[..., None]*np.asarray(tint, np.float32)*strength
    return img

_VIG = {}
def vignette(strength=0.55, power=2.2, cy=0.5):
    key = (strength, power, cy)
    if key not in _VIG:
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((x-W/2)/(W*0.78))**2+((y-H*cy)/(H*0.62))**2)
        _VIG[key] = (1-strength*np.clip(r, 0, 1.4)**power).astype(np.float32)[..., None]
    return _VIG[key]

def make_lut(exposure=1.0, contrast=1.10, pivot=0.38, lift=(0, 0, 0), gamma=(1, 1, 1), gain=(1, 1, 1), shoulder=0.72, toe=0.0, n=4096, xmax=4.0):
    """per-channel tone curves; channel order BGR. returns (3, n) float32 LUT for inputs 0..xmax"""
    x = np.linspace(0, xmax, n, dtype=np.float64)*exposure
    out = []
    for c in range(3):
        y = x*gain[c]
        y = np.where(y < shoulder, y, shoulder+(1-shoulder)*(1-np.exp(-(y-shoulder)/(1-shoulder))))
        y = np.clip(y, 0, 1)
        y = pivot*(y/pivot)**contrast*(y < pivot)+(1-(1-pivot)*((1-y)/(1-pivot))**contrast)*(y >= pivot)
        y = np.clip(y, 0, 1)**(1.0/gamma[c])
        y = y*(1-lift[c])+lift[c]
        out.append(y)
    return np.array(out, np.float32), xmax
def apply_lut(img, lutx, sat=1.0):
    lut, xmax = lutx
    if sat != 1.0:
        L = (0.114*img[..., 0]+0.587*img[..., 1]+0.299*img[..., 2])[..., None]
        img = L+(img-L)*sat
    n = lut.shape[1]
    q = np.clip(img*((n-1)/xmax), 0, n-1).astype(np.uint16)
    out = np.empty_like(img)
    for c in range(3): out[..., c] = lut[c][q[..., c]]
    return out

def letterbox(img, bar=BAR):
    img[:bar] = 0; img[H-bar:] = 0
    return img

def finish(img, frame_idx, grain_amt=0.020):
    """img float 0..1 BGR after grade -> uint8 with grain + bars"""
    L = img.mean(2, keepdims=True)
    g = grain(frame_idx, grain_amt)[..., None]
    img = img+g*(0.35+0.65*np.clip(1.2-L, 0.2, 1))
    letterbox(img)
    return np.clip(img*255+0.5, 0, 255).astype(np.uint8)

def chroma_ab(img, amt=1.0015):
    """tiny radial chromatic aberration"""
    M = cv2.getRotationMatrix2D((W/2, H/2), 0, amt)
    img[..., 2] = cv2.warpAffine(img[..., 2], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    M = cv2.getRotationMatrix2D((W/2, H/2), 0, 2-amt)
    img[..., 0] = cv2.warpAffine(img[..., 0], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return img

def shake2d(img, dx, dy, rot, zoom=1.03):
    M = cv2.getRotationMatrix2D((W/2, H/2), rot, zoom); M[0, 2] += dx; M[1, 2] += dy
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def zoom_blur(img, center, amount=0.03, steps=5, mask=None):
    acc = img.copy(); n = 1
    for k in range(1, steps):
        s = 1+amount*k/(steps-1)
        M = np.array([[s, 0, center[0]*(1-s)], [0, s, center[1]*(1-s)]], np.float32)
        acc += cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE); n += 1
    acc /= n
    if mask is not None: return img*(1-mask)+acc*mask
    return acc

# ------------------------------------------------------------------ text
FONTS = 'fonts/'
_FC = {}
def font(name, size, weight=None):
    key = (name, size, weight)
    if key not in _FC:
        f = ImageFont.truetype(FONTS+name, size)
        if weight is not None:
            try: f.set_variation_by_axes([weight])
            except Exception: pass
        _FC[key] = f
    return _FC[key]
_GC = {}
def glyph(ch, name, size, weight=None):
    key = (ch, name, size, weight)
    if key not in _GC:
        f = font(name, size, weight)
        asc, desc = f.getmetrics()
        adv = f.getlength(ch)
        pad = int(size*0.25)+4
        im = Image.new('L', (int(adv)+2*pad, asc+desc+2*pad), 0)
        ImageDraw.Draw(im).text((pad, pad), ch, font=f, fill=255)
        _GC[key] = (np.asarray(im, np.float32)/255.0, adv, pad, asc)
    return _GC[key]
def text_mask(text, name, size, weight=None, tracking=0.0, space=0.32):
    """returns alpha (float 0..1) for a single line with given tracking (em), tight canvas with margin"""
    gl = []; x = 0.0
    for ch in text:
        if ch == ' ':
            x += size*space+tracking*size; continue
        a, adv, pad, asc = glyph(ch, name, size, weight)
        gl.append((a, x, pad)); x += adv+tracking*size
    width = x-tracking*size
    m = int(size*0.6)+8
    hh = gl[0][0].shape[0]
    out = np.zeros((hh+2*m-2*gl[0][2], int(math.ceil(width))+2*m), np.float32)
    for a, gx, pad in gl:
        fx = gx+m-pad; ix = int(math.floor(fx)); frac = fx-ix
        if frac > 1e-3:
            M = np.float32([[1, 0, frac], [0, 1, 0]]); a = cv2.warpAffine(a, M, (a.shape[1]+1, a.shape[0]), flags=cv2.INTER_LINEAR)
        y0 = m-pad; h_, w_ = a.shape
        ys0 = max(0, y0); xs0 = max(0, ix)
        sub = out[ys0:y0+h_, xs0:ix+w_]
        sub[:] = np.maximum(sub, a[ys0-y0:ys0-y0+sub.shape[0], xs0-ix:xs0-ix+sub.shape[1]])
    return out

def place(dst, layer_rgb, layer_a, cx, cy, scale=1.0, additive=False, opacity=1.0):
    """composite a (premultiplied if additive False: straight rgb + alpha) layer centred at (cx, cy)"""
    h, w = layer_a.shape
    if scale != 1.0:
        nw, nh = max(1, int(round(w*scale))), max(1, int(round(h*scale)))
        layer_rgb = cv2.resize(layer_rgb, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        layer_a = cv2.resize(layer_a, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        h, w = nh, nw
    x0 = int(round(cx-w/2)); y0 = int(round(cy-h/2))
    X0, Y0 = max(0, x0), max(0, y0); X1, Y1 = min(W, x0+w), min(H, y0+h)
    if X1 <= X0 or Y1 <= Y0: return
    c = layer_rgb[Y0-y0:Y1-y0, X0-x0:X1-x0]; a = layer_a[Y0-y0:Y1-y0, X0-x0:X1-x0, None]*opacity
    if additive: dst[Y0:Y1, X0:X1] += c*a
    else: dst[Y0:Y1, X0:X1] = dst[Y0:Y1, X0:X1]*(1-a)+c*a

def bevel(a, sigma=5.0, light=(-0.5, -0.8), depth=1.0):
    hgt = cv2.GaussianBlur(a, (0, 0), sigma)
    gx = cv2.Sobel(hgt, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(hgt, cv2.CV_32F, 0, 1, ksize=3)
    l = np.array(light, np.float32); l /= np.linalg.norm(l)
    return np.clip(-(gx*l[0]+gy*l[1])*sigma*depth, -1, 1)
