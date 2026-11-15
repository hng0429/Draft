"""Landmark-driven expression warps for the head sprites (jaw open / brows / smile)."""
import cv2, numpy as np
from scipy.spatial import Delaunay
FH = 300.0
UP_OUT=[61,185,40,39,37,0,267,269,270,409,291]; LO_OUT=[61,146,91,181,84,17,314,405,321,375,291]
UP_IN=[78,191,80,81,82,13,312,311,310,415,308]; LO_IN=[78,95,88,178,87,14,317,402,318,324,308]
BROW_L=[70,63,105,66,107,55,65,52,53,46]; BROW_R=[300,293,334,296,336,285,295,282,283,276]
LID_UP=[246,161,160,159,158,157,173,466,388,387,386,385,384,398]
LID_LO=[7,163,144,145,153,154,155,249,390,373,374,380,381,382]
OVAL=[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]

class Head:
    def __init__(self, path):
        d = np.load(path)
        self.rgb = d['rgb'].astype(np.float32); self.alpha = d['alpha'].astype(np.float32)/255
        self.lm = d['lm'][:468,:2].copy(); self.lm3 = d['lm'][:468].copy()
        self.h, self.w = self.alpha.shape
        lm = self.lm
        self.xc = (lm[13,0]+lm[14,0])/2; self.ym = (lm[61,1]+lm[291,1])/2; self.yc = lm[152,1]
        self.hw = abs(lm[291,0]-lm[61,0])/2
        self.R = (abs(lm[397,0]-self.xc)+abs(lm[172,0]-self.xc))/2
        self.cen = lm[OVAL].mean(0)
        # lip line: mid curve between inner contours as function of x
        up = lm[UP_IN]; lo = lm[LO_IN]; mid = (up+lo)/2; o = np.argsort(mid[:,0])
        self.lipx, self.lipy = mid[o,0], mid[o,1]
        ringB = self.cen + (lm[OVAL]-self.cen)*1.16
        ringC = self.cen + (lm[OVAL]-self.cen)*1.55
        ringC[:,0] = np.clip(ringC[:,0], 4, self.w-5); ringC[:,1] = np.clip(ringC[:,1], 4, self.h-5)
        bx = np.linspace(0, self.w-1, 7); by = np.linspace(0, self.h-1, 8)
        border = np.array([(x,0) for x in bx]+[(x,self.h-1) for x in bx]+[(0,y) for y in by[1:-1]]+[(self.w-1,y) for y in by[1:-1]],float)
        self.src = np.vstack([lm, ringB, ringC, border])
        self.nB = len(ringB); self.nfix = len(ringC)+len(border)
        self.tri = Delaunay(self.src).simplices
        # skin / lip colour samples
        a = self.rgb
        def med(p, r=6):
            x,y = int(p[0]), int(p[1]); return np.median(a[y-r:y+r, x-r:x+r].reshape(-1,3), axis=0)
        self.skin = (med(lm[205])+med(lm[425]))/2      # cheeks
        self.lipcol = med((lm[17]+lm[14])/2, 3)
        self.is_lip_up = set(UP_OUT+UP_IN); self.is_lip_lo = set(LO_OUT+LO_IN)

    def _weights(self, P, explicit=None):
        """jaw-drop weight for points P (N,2); explicit: list of 'up'/'lo'/None per point"""
        x, y = P[:,0], P[:,1]
        t = np.abs(x-self.xc)/self.hw
        u = (y-self.ym)/(self.yc-self.ym)
        yl = np.interp(x, self.lipx, self.lipy)
        w = np.zeros(len(P))
        inside = t <= 1.0
        upper = inside & (y < yl-0.5)
        lower = inside & ~upper
        if explicit is not None:
            for i, e in enumerate(explicit):
                if e == 'up': upper[i], lower[i] = True, False
                elif e == 'lo': upper[i], lower[i] = False, True
        tt = np.clip(t, 0, 1)
        w_up = 0.35*tt**4*np.clip(1+u/0.45, 0, 1)
        w_lip = 0.35+0.65*np.cos(np.pi*tt/2)**2
        w_lo = w_lip+(1-w_lip)*np.clip(u, 0, 1)**0.6
        w_lat = np.where(u < 0, 0.35*np.clip(1+u/0.45, 0, 1), 0.35+0.65*np.clip(u, 0, 1)**0.6)
        g = 1-0.75*np.clip((np.abs(x-self.xc)-self.hw)/max(self.R-self.hw, 1), 0, 1.3)**2
        w = np.where(upper, w_up, np.where(lower, w_lo, w_lat*np.clip(g, 0.1, 1)))
        return w

    def targets(self, jaw=0.0, brow=0.0, smile=0.0, lid=0.0):
        lm = self.lm; n = len(lm)
        pts = np.vstack([lm, self.src[n:n+self.nB]])
        explicit = [None]*len(pts)
        for i in self.is_lip_up: explicit[i] = 'up'
        for i in self.is_lip_lo: explicit[i] = 'lo'
        for i in (61, 291, 78, 308): explicit[i] = 'lo'
        a = jaw*0.205*FH
        w = self._weights(pts, explicit)
        d = np.zeros_like(pts)
        d[:,1] += a*w
        # upper lip lifts a little, mouth corners pull in when wide open
        for i in self.is_lip_up:
            if i not in (61,291,78,308): d[i,1] -= 0.06*a*(1-min(1,abs(lm[i,0]-self.xc)/self.hw)**2)
        for cidx in (61, 291):
            c = lm[cidx]; s = np.sign(c[0]-self.xc)
            f = np.exp(-((pts-c)**2).sum(1)/(2*(0.55*self.hw)**2))
            d[:,0] += -s*0.11*a*f
            # smile: corners out + up, cheeks lift
            d[:,0] += s*0.040*FH*smile*f; d[:,1] += -0.034*FH*smile*f
        if smile:
            for cidx in (205, 425):
                f = np.exp(-((pts-lm[cidx])**2).sum(1)/(2*(0.11*FH)**2)); d[:,1] += -0.014*FH*smile*f
            for i in LID_LO: d[i,1] -= 0.006*FH*smile
        # slight narrowing of lower face when jaw drops
        u = np.clip((pts[:,1]-self.ym)/(self.yc-self.ym), 0, 1.2)
        d[:,0] += -np.sign(pts[:,0]-self.xc)*0.045*a*u*np.clip(np.abs(pts[:,0]-self.xc)/self.R, 0, 1)
        # brows (fear: up, inner ends more)
        if brow:
            b = 0.034*FH*brow
            for grp, inner in ((BROW_L, (107, 55)), (BROW_R, (336, 285))):
                dd = np.sqrt(((pts[:,None,:]-lm[grp][None])**2).sum(2)).min(1)
                f = np.exp(-dd**2/(2*(0.06*FH)**2)); d[:,1] -= b*f
                for i in inner: d[i,1] -= 0.35*b; d[i,0] += np.sign(self.xc-lm[i,0])*0.2*b
        if lid:
            for i in LID_UP: d[i,1] -= 0.013*FH*lid
            for i in LID_LO: d[i,1] += 0.004*FH*lid
        dst = self.src.copy(); dst[:len(pts)] = pts+d
        return dst

    def render(self, jaw=0.0, brow=0.0, smile=0.0, lid=0.0, extra=None):
        """returns rgb(float32 0..255), alpha(0..1) and optionally warped extra maps"""
        dst = self.targets(jaw, brow, smile, lid)
        mapx = np.tile(np.arange(self.w, dtype=np.float32), (self.h, 1)); mapy = np.tile(np.arange(self.h, dtype=np.float32)[:,None], (1, self.w))
        for tr in self.tri:
            s = self.src[tr].astype(np.float32); t = dst[tr].astype(np.float32)
            if np.abs(s-t).max() < 1e-3: continue
            M = cv2.getAffineTransform(t, s)
            x0, y0 = np.floor(t.min(0)).astype(int); x1, y1 = np.ceil(t.max(0)).astype(int)+1
            x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, self.w), min(y1, self.h)
            if x1 <= x0 or y1 <= y0: continue
            m = np.zeros((y1-y0, x1-x0), np.uint8); cv2.fillConvexPoly(m, np.round((t-[x0, y0])).astype(np.int32), 1)
            ys, xs = np.nonzero(m)
            X = xs+x0; Y = ys+y0
            mapx[Y, X] = M[0,0]*X+M[0,1]*Y+M[0,2]; mapy[Y, X] = M[1,0]*X+M[1,1]*Y+M[1,2]
        rgb = cv2.remap(self.rgb, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        alpha = cv2.remap(self.alpha, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        outs = [cv2.remap(e, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for e in (extra or [])]
        rgb = np.clip(rgb, 0, 255)
        if jaw > 0.02: rgb = self._mouth(rgb, dst)
        return (rgb, np.clip(alpha, 0, 1), outs) if extra is not None else (rgb, np.clip(alpha, 0, 1))

    def _mouth(self, rgb, dst):
        S = 4  # supersample factor for the mouth patch
        up = dst[UP_IN]; lo = dst[LO_IN]
        poly = np.vstack([up, lo[::-1][1:-1]])
        x0, y0 = np.floor(poly.min(0)).astype(int)-6; x1, y1 = np.ceil(poly.max(0)).astype(int)+6
        W, H = (x1-x0)*S, (y1-y0)*S
        def tf(p): return (np.asarray(p)-[x0, y0])*S
        opening = (lo[:,1]-up[:,1]).max(); width = up[-1,0]-up[0,0]
        mask = np.zeros((H, W), np.uint8); cv2.fillPoly(mask, [np.round(tf(poly)).astype(np.int32)], 255, cv2.LINE_AA)
        layer = np.zeros((H, W, 3), np.float32)
        yy = np.linspace(0, 1, H)[:,None]
        back = np.array([9, 5, 16], np.float32)
        layer[:] = back
        # tongue
        tc = tf([(up[0,0]+up[-1,0])/2, lo[:,1].max()-0.20*opening])
        tl = np.zeros((H, W), np.float32); cv2.ellipse(tl, tuple(np.round(tc).astype(int)), (int(0.31*width*S), int(max(2, 0.26*opening*S))), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        tl = cv2.GaussianBlur(tl, (0, 0), 0.05*width*S)
        tong = np.clip(self.lipcol*np.array([0.50, 0.40, 0.70]), 0, 255)
        layer = layer*(1-tl[...,None]*0.8)+tong*tl[...,None]*0.8
        # upper teeth: band below the upper inner lip
        ux = np.linspace(up[0,0], up[-1,0], 60); uy = np.interp(ux, up[:,0], up[:,1])
        tnorm = (ux-ux.mean())/(width/2)
        th = min(0.34*opening, 0.070*FH)*(1-0.30*tnorm**2)
        vis = np.abs(tnorm) < 0.86
        bright = np.clip(self.skin*0.35+np.array([150, 160, 165])*0.72, 0, 255)
        if th.max() > 1.5:
            top = np.stack([ux[vis], uy[vis]-2], 1); bot = np.stack([ux[vis], uy[vis]+th[vis]], 1)
            tp = np.vstack([top, bot[::-1]])
            tm = np.zeros((H, W), np.uint8); cv2.fillPoly(tm, [np.round(tf(tp)).astype(np.int32)], 255, cv2.LINE_AA)
            tmf = tm.astype(np.float32)/255
            # shade: darker to the sides and near the lip
            xs = (np.arange(W)/S+x0-ux.mean())/(width/2)
            side = np.clip(1-0.55*np.abs(xs)**2.2, 0.25, 1)[None,:]
            tcol = bright[None,None,:]*side[...,None]*np.ones((H,1,1),np.float32)
            # gaps between teeth
            gap = np.ones((H, W), np.float32)
            for g in (0, 0.27, -0.27, 0.50, -0.50, 0.69, -0.69, 0.83, -0.83):
                gx = int(round(((ux.mean()+g*width/2)-x0)*S)); cv2.line(gap, (gx, 0), (gx, H), 0.5, max(1, int(0.4*S)), cv2.LINE_AA)
            gap = cv2.GaussianBlur(gap, (0, 0), 0.35*S)
            tcol = tcol*gap[...,None]
            layer = layer*(1-tmf[...,None])+tcol*tmf[...,None]
        # lower teeth (mostly hidden)
        lx = np.linspace(lo[0,0], lo[-1,0], 60); ly = np.interp(lx, lo[:,0], lo[:,1]); ln = (lx-lx.mean())/(width/2)
        lth = min(0.13*opening, 0.03*FH)*(1-0.5*ln**2); lv = np.abs(ln) < 0.7
        if lth.max() > 1.2:
            lp = np.vstack([np.stack([lx[lv], ly[lv]-lth[lv]], 1), np.stack([lx[lv], ly[lv]+2], 1)[::-1]])
            lmk = np.zeros((H, W), np.uint8); cv2.fillPoly(lmk, [np.round(tf(lp)).astype(np.int32)], 255, cv2.LINE_AA)
            lmf = cv2.GaussianBlur(lmk.astype(np.float32)/255, (0, 0), 0.5*S)
            layer = layer*(1-lmf[...,None]*0.85)+(bright*0.62)*lmf[...,None]*0.85
        # inner shadow near the lips
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)/S
        sh = np.clip(dist/max(2.0, 0.16*opening), 0, 1); sh = 0.45+0.55*sh*sh*(3-2*sh)
        layer *= sh[...,None]
        layer = cv2.GaussianBlur(layer, (0, 0), 0.55*S)
        small = cv2.resize(layer, (x1-x0, y1-y0), interpolation=cv2.INTER_AREA)
        msm = cv2.resize(mask, (x1-x0, y1-y0), interpolation=cv2.INTER_AREA).astype(np.float32)/255
        msm = cv2.GaussianBlur(msm, (0, 0), 0.8)[...,None]
        out = rgb.copy(); reg = out[y0:y1, x0:x1]
        out[y0:y1, x0:x1] = reg*(1-msm)+small*msm
        return out

if __name__ == '__main__':
    rows = []
    for i in range(4):
        h = Head(f'head_{i}.npz'); t = []
        for kw in (dict(), dict(jaw=0.35, brow=0.5, lid=0.5), dict(jaw=0.7, brow=0.9, lid=1), dict(jaw=1.0, brow=1.0, lid=1), dict(jaw=0.0, smile=1), dict(jaw=0.45, smile=1.0)):
            rgb, al = h.render(**kw)
            im = (rgb*al[...,None]+np.array((60, 45, 40))*(1-al[...,None])).astype(np.uint8)
            t.append(im[150:720, 100:620])
        rows.append(np.hstack(t))
    cv2.imwrite('expr.jpg', cv2.resize(np.vstack(rows), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    cv2.imwrite('expr_big.jpg', np.vstack(rows)[:570*2, 520:520*4])
