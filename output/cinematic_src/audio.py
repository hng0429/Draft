"""Synthesised soundtrack: fire ambience, trailer hits, action percussion, riser, then a warm major-key resolution."""
import numpy as np, scipy.signal as sg, scipy.io.wavfile as wf
SR = 48000; DUR = 18.8; N = int(SR*DUR)
rng = np.random.default_rng(7)
T1, T2, T3, T4, T5 = 0.0, 3.6, 7.4, 10.0, 13.6      # shot starts
mix = np.zeros((N+SR*4, 2), np.float64)             # dry bus
rev = np.zeros((N+SR*4, 2), np.float64)             # reverb send

def lp(x, fc, o=2): return sg.sosfilt(sg.butter(o, min(fc, SR*0.45)/(SR/2), 'low', output='sos'), x, axis=0)
def hp(x, fc, o=2): return sg.sosfilt(sg.butter(o, fc/(SR/2), 'high', output='sos'), x, axis=0)
def bp(x, f1, f2, o=2): return sg.sosfilt(sg.butter(o, [f1/(SR/2), min(f2, SR*0.45)/(SR/2)], 'band', output='sos'), x, axis=0)
def tt(d): return np.arange(int(d*SR))/SR
def env(d, a=0.01, r=0.1, curve=1.0):
    t = tt(d); e = np.minimum(1, t/max(a, 1e-4))*np.minimum(1, (d-t)/max(r, 1e-4)); return np.clip(e, 0, 1)**curve
def put(sig, t0, gain=1.0, pan=0.0, send=0.0):
    """sig mono or stereo; pan -1..1"""
    i = int(t0*SR)
    if sig.ndim == 1:
        l = np.cos((pan+1)*np.pi/4); r = np.sin((pan+1)*np.pi/4); sig = np.stack([sig*l, sig*r], 1)*1.4142
    n = min(len(sig), len(mix)-i)
    if n <= 0: return
    mix[i:i+n] += sig[:n]*gain
    if send: rev[i:i+n] += sig[:n]*gain*send
def saw(f, d, hmax=7000, phase=None):
    t = tt(d); K = int(max(1, min(48, hmax//f))); out = np.zeros_like(t)
    for k in range(1, K+1):
        out += np.sin(2*np.pi*k*f*t+rng.uniform(0, 6.28))/k
    return out*0.6
def pad(notes, d, a=0.5, r=1.0, fc=1800, det=7.0, vib=0.0):
    out = np.zeros((int(d*SR), 2))
    for f in notes:
        for j, c in enumerate((-det, 0.0, det)):
            ff = f*2**(c/1200.0)
            s = saw(ff, d, hmax=5000)
            if vib:
                t = tt(d); s = s*(1+vib*np.sin(2*np.pi*(4.6+0.3*j)*t+j))
            p = (-0.7, 0.0, 0.7)[j]
            out[:, 0] += s*np.cos((p+1)*np.pi/4); out[:, 1] += s*np.sin((p+1)*np.pi/4)
    out = lp(out, fc, 2)
    return out*env(d, a, r)[:, None]/max(1, len(notes))**0.5
def bell(f, d=2.2, bright=1.0):
    t = tt(d); out = np.zeros_like(t)
    for ratio, amp, dec in ((1.0, 1.0, 1.0), (2.0, 0.45*bright, 0.7), (3.01, 0.22*bright, 0.45), (4.2, 0.12*bright, 0.3), (5.43, 0.06*bright, 0.22)):
        out += amp*np.sin(2*np.pi*f*ratio*t)*np.exp(-t/(0.55*dec*d/2.2))
    return out*np.minimum(1, t/0.004)
def drum(f0=120, f1=48, d=0.5, tau=0.035, click=0.5, dec=0.16):
    t = tt(d); f = f1+(f0-f1)*np.exp(-t/tau)
    ph = 2*np.pi*np.cumsum(f)/SR
    s = (np.sin(ph)+0.45*np.sin(2*ph)+0.25*np.sin(3.02*ph)+0.15*np.sin(4.1*ph))*np.exp(-t/dec)
    c = bp(rng.normal(0, 1, len(t)), 250, 3200)*np.exp(-t/0.022)*click*2.2
    return np.tanh((s+c)*1.5)
def noise(d): return rng.normal(0, 1, int(d*SR))

# ---------------------------------------------------------------- fire ambience (hell: 0 .. 10s)
d = T4+0.3
roar = lp(np.cumsum(rng.normal(0, 1, (int(d*SR), 2)), axis=0)*0.002, 320, 2); roar = hp(roar, 30)
am = 0.75+0.25*lp(rng.normal(0, 1, (int(d*SR), 1)), 3.0)*40
roar *= np.clip(am, 0.3, 1.6)
cr = np.zeros((int(d*SR), 2))
nimp = int(d*34)
for _ in range(nimp):
    i = rng.integers(0, len(cr)-2000); L = int(rng.uniform(0.002, 0.014)*SR)
    b = rng.normal(0, 1, L)*np.exp(-np.arange(L)/(L*0.3))*rng.uniform(0.1, 1.0)**2.2
    p = rng.uniform(-0.9, 0.9)
    cr[i:i+L, 0] += b*np.cos((p+1)*np.pi/4); cr[i:i+L, 1] += b*np.sin((p+1)*np.pi/4)
cr = bp(cr, 1400, 7500, 2)
fenv = np.minimum(1, tt(d)/1.2)*np.clip((T4+0.05-tt(d))/0.6, 0, 1)
fenv2 = fenv*(1-0.6*np.clip((tt(d)-7.6)/2.0, 0, 1))
put(roar/np.abs(roar).max()*fenv[:, None], 0, 0.20)
put(cr/np.abs(cr).max()*fenv2[:, None], 0, 0.62, send=0.10)
wh = bp(rng.normal(0, 1, (int(d*SR), 2)), 280, 1800, 2)*np.clip(0.6+0.4*lp(rng.normal(0, 1, (int(d*SR), 1)), 1.5)*60, 0.2, 1.5)
put(wh/np.abs(wh).max()*fenv[:, None], 0, 0.17, send=0.1)

# ---------------------------------------------------------------- dark drone (0 .. 10s)
dr = pad([36.71, 73.42, 110.0], d, a=1.6, r=0.5, fc=260, det=9)+0.55*pad([146.83, 220.0, 293.66], d, a=2.0, r=0.5, fc=900, det=10)
dr2 = pad([77.78], d, a=3.0, r=0.5, fc=220, det=5)*0.35         # uneasy minor 2nd
swell = (0.55+0.45*np.clip(tt(d)/T4, 0, 1))[:, None]
put((dr+dr2)*swell, 0, 0.55, send=0.15)

# ---------------------------------------------------------------- braam on the title (1.95)
def braam(d=2.6, base=73.42):
    s = pad([base, base*2, base*1.498, base*0.5, base*4], d, a=0.02, r=1.6, fc=5000, det=14)
    t = tt(d)[:, None]
    out = np.zeros_like(s)
    cuts = [3800, 2400, 1500, 900, 520]
    pos = np.clip(t/(d*0.8), 0, 1)*(len(cuts)-1)
    for k, c in enumerate(cuts):
        w = np.clip(1-np.abs(pos-k), 0, 1); out += lp(s, c, 2)*w
    out = np.tanh(out*3.0)*np.exp(-t/1.1)
    sub = np.sin(2*np.pi*np.cumsum(38+42*np.exp(-tt(d)/0.25))/SR)*np.exp(-tt(d)/0.9)
    return out+0.9*sub[:, None]
b = braam()
put(b/np.abs(b).max(), 1.93, 0.62, send=0.35)
put(drum(150, 40, 1.2, 0.06, 0.8, 0.45), 1.93, 0.55, send=0.3)

# ---------------------------------------------------------------- action section (S2 + S3)
STEP = 0.1
def hit(t0, g=1.0, kind='lo'):
    if kind == 'lo': put(drum(125, 46, 0.6, 0.04, 0.5, 0.20), t0, 0.50*g, pan=0.0, send=0.22)
    elif kind == 'mid': put(drum(190, 80, 0.35, 0.03, 0.6, 0.11), t0, 0.30*g, pan=rng.uniform(-0.5, 0.5), send=0.25)
    else: put(drum(260, 120, 0.25, 0.02, 0.9, 0.07), t0, 0.20*g, pan=rng.uniform(-0.6, 0.6), send=0.25)
KICK = [1, 0, 0, 1, 0, 0, 1, 0]; TOM = [0, 0, 1, 0, 1, 0, 0, 1]
SLOW0, SLOW1 = T2+1.02, T2+1.90
k = 0; t = T2
put(drum(170, 36, 1.6, 0.08, 1.0, 0.6), T2-0.01, 0.80, send=0.4)           # cut impact
while t < T4-0.05:
    inside = SLOW0 <= t < SLOW1
    if not inside:
        prog = (t-T2)/(T4-T2)
        if KICK[k % 8]: hit(t, 0.85+0.3*prog, 'lo')
        if TOM[k % 8]: hit(t, 0.7+0.5*prog, 'mid')
        if t > T3+1.3: hit(t, 0.3+1.0*(t-T3-1.3)/1.3, 'hi')               # roll into the light
        # low string chug
        f = [73.42, 73.42, 87.31, 73.42, 73.42, 98.0, 87.31, 73.42][k % 8]
        s = lp(saw(f, 0.11, 4000)*0.6+saw(f*2, 0.11, 4000)+saw(f*4, 0.11, 5000)*0.5, 1900)*env(0.11, 0.004, 0.05)
        put(s, t, 0.26*(1.0 if KICK[k % 8] else 0.6), pan=-0.2, send=0.15)
    k += 1; t += STEP
# geyser explosion + slow motion "air"
ex = lp(noise(2.4), 900)*np.exp(-tt(2.4)/0.7)+0.8*np.sin(2*np.pi*np.cumsum(30+40*np.exp(-tt(2.4)/0.3))/SR)*np.exp(-tt(2.4)/0.8)
put(np.tanh(ex/np.abs(ex).max()*2.2), T2+0.97, 0.62, send=0.4)
sl = pad([73.42, 110.0, 146.83], SLOW1-SLOW0+0.3, a=0.25, r=0.3, fc=500, det=10)
put(sl, SLOW0-0.05, 0.5, send=0.4)
rv = lp(noise(0.5), 2500)*np.linspace(0, 1, int(0.5*SR))**3                # suck-in back to speed
put(rv, SLOW1-0.5, 0.20, send=0.2)
put(drum(170, 40, 1.0, 0.06, 1.0, 0.4), SLOW1, 0.65, send=0.35)

# riser from mid S2 to the white-out
rd = T4-(T2+2.2)
r0 = noise(rd)
cuts = [300, 600, 1200, 2400, 4800, 8000]
pos = (tt(rd)/rd)**1.3*(len(cuts)-1); rs = np.zeros(int(rd*SR))
for kk, c in enumerate(cuts):
    rs += bp(r0, c*0.7, c*1.4, 2)*np.clip(1-np.abs(pos-kk), 0, 1)
rs *= (tt(rd)/rd)**2.2
gl = np.sin(2*np.pi*np.cumsum(110*2**(3.0*(tt(rd)/rd)**1.5))/SR)*(tt(rd)/rd)**2*(1+0.3*np.sin(2*np.pi*7*tt(rd)))
put(rs/np.abs(rs).max(), T2+2.2, 0.36, send=0.3)
put(lp(gl, 6000), T2+2.2, 0.15, send=0.4)

# ---------------------------------------------------------------- transition into the light (reverse swell -> impact at T4)
D3, A3, D4, Fs4, A4, D5, E5, Fs5, A5, D6, E6, Fs6 = 146.83, 220.0, 293.66, 369.99, 440.0, 587.33, 659.26, 739.99, 880.0, 1174.66, 1318.51, 1479.98
G3, B3, G4, B4, Cs4, E4, Cs5 = 196.0, 246.94, 392.0, 493.88, 277.18, 329.63, 554.37
sw = sum(bell(f, 3.0, 0.8) for f in (D4, A4, D5, Fs5, A5))
ir0 = rng.normal(0, 1, int(1.8*SR))*np.exp(-tt(1.8)/0.5)
sw = sg.fftconvolve(sw, ir0)[:int(2.6*SR)]; sw = sw[::-1]/np.abs(sw).max()
sw *= np.minimum(1, tt(2.6)[:len(sw)]/0.3)
put(sw, T4-len(sw)/SR, 0.34, send=0.2)
put(drum(140, 34, 2.2, 0.09, 0.6, 0.9), T4, 0.60, send=0.5)                 # soft deep bloom

# ---------------------------------------------------------------- hope: warm chords
I_ = [D3, A3, D4, Fs4, A4]; IV = [G3, B3, D4, G4, B4]; V_ = [A3, Cs4, E4, A4, Cs5]; If = [D3, A3, D4, Fs4, A4, D5]
def chord(notes, t0, d, g=1.0, fc=2200):
    put(pad(notes, d+1.2, a=0.45, r=1.2, fc=fc, det=6, vib=0.03), t0, 0.50*g, send=0.45)
    put(pad([notes[0]/2], d+1.0, a=0.3, r=1.0, fc=400, det=3), t0, 0.50*g)                    # bass
chord(I_, T4+0.02, 2.4, 0.85, 1800)
chord(IV, T4+2.4, 0.7, 0.9, 2100)
chord(V_, T4+3.0, 0.6, 1.0, 2500)
chord(If, T5, 2.3, 1.15, 3000)
chord(IV, T5+2.3, 1.0, 1.0, 2600)
chord(If, T5+3.3, 2.2, 1.0, 2800)
put(drum(110, 44, 1.6, 0.07, 0.3, 0.7), T5, 0.42, send=0.5)                  # timpani-like arrival
# high shimmer (octave strings) on the final section
sh = pad([D5, A5, D6], 5.2, a=1.2, r=1.6, fc=5000, det=9, vib=0.05)
put(hp(sh, 400), T5, 0.20, send=0.6)
# arpeggio bells in S4
seq4 = [D5, Fs5, A5, D6, A5, Fs5, D5, A5, G4*2, B4*2, D6, B4*2, A5, Cs5*2, E6, A5]
for j in range(12):
    f = seq4[j % len(seq4)]
    put(bell(f, 1.8, 0.7), T4+0.25+j*0.28, 0.085, pan=0.5*np.sin(j*1.3), send=0.6)
# melody in S5
for dt, f, g in ((0.0, A5, 1.0), (0.55, D6, 1.0), (1.10, Fs6, 1.05), (1.95, E6, 0.9), (2.5, D6, 1.0), (3.3, A5, 0.85), (3.85, D6, 0.8)):
    put(bell(f, 2.6, 0.9), T5+0.12+dt, 0.15*g, pan=0.3*np.sin(dt*2), send=0.7)
    put(bell(f/2, 2.6, 0.5), T5+0.12+dt, 0.07*g, pan=-0.3*np.sin(dt*2), send=0.7)
# birds + air
def chirp():
    d = rng.uniform(0.06, 0.13); t = tt(d); f0 = rng.uniform(2600, 4200); f1 = f0*rng.uniform(1.15, 1.6)
    f = f0+(f1-f0)*np.sin(np.pi*t/d)**1.5
    return np.sin(2*np.pi*np.cumsum(f)/SR)*np.sin(np.pi*t/d)**2*(1+0.4*np.sin(2*np.pi*38*t))
tb = T4+0.5
while tb < DUR-0.8:
    nn = rng.integers(2, 6); p = rng.uniform(-0.8, 0.8); g = rng.uniform(0.02, 0.05)
    for q in range(nn): put(chirp(), tb+q*rng.uniform(0.09, 0.16), g, pan=p, send=0.5)
    tb += rng.uniform(0.5, 1.4)
air = hp(lp(rng.normal(0, 1, (int((DUR-T4)*SR), 2)), 5000), 500)*env(DUR-T4, 1.0, 1.0)[:, None]
put(air/np.abs(air).max(), T4, 0.035)

# ---------------------------------------------------------------- reverb + master
L = int(3.2*SR); tI = np.arange(L)/SR
ir = np.stack([rng.normal(0, 1, L), rng.normal(0, 1, L)], 1)
ir = lp(ir, 6500)*np.exp(-tI/0.42)[:, None]*0.6+lp(ir, 1800)*np.exp(-tI/0.85)[:, None]
ir[:int(0.012*SR)] = 0
ir /= np.sqrt((ir**2).sum(0)).mean()
wet = np.stack([sg.fftconvolve(rev[:, c], ir[:, c])[:len(mix)] for c in (0, 1)], 1)
out = mix+wet*0.55
out = out[:N]
out = hp(out, 24, 2)
fade = np.ones(N); fo = int(1.5*SR); fade[-fo:] = np.cos(np.linspace(0, np.pi/2, fo))**2
fi = int(0.25*SR); fade[:fi] *= np.linspace(0, 1, fi)
out *= fade[:, None]
# loudness: aim for about -15 LUFS (approximated with K-weighting-free RMS of the loudest half) with gentle peak control
out = out/np.abs(out).max()
def soft(x, th=0.80): 
    y = x.copy(); m = np.abs(x) > th
    y[m] = np.sign(x[m])*(th+(1-th)*np.tanh((np.abs(x[m])-th)/(1-th)))
    return y
import subprocess, re
def lufs(sig):
    wf.write('_tmp.wav', SR, (np.clip(sig, -1, 1)*32767).astype(np.int16))
    r = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-i', '_tmp.wav', '-af', 'ebur128', '-f', 'null', '-'], capture_output=True, text=True).stderr
    return float(re.findall(r'I:\s*(-?[0-9.]+) LUFS', r)[-1])
g = 10**((-15.0-lufs(out))/20)
out = soft(out*g)*0.97
print('gain', round(g, 3), 'final LUFS', lufs(out))
wf.write('score.wav', SR, (np.clip(out, -1, 1)*32767).astype(np.int16))
seg = lambda a, b: 20*np.log10(np.sqrt((out[int(a*SR):int(b*SR)]**2).mean())+1e-9)
print('peak', np.abs(out).max().round(3), 'rms dB by shot', [round(seg(a, b), 1) for a, b in ((0, T2), (T2, T3), (T3, T4), (T4, T5), (T5, DUR))])
