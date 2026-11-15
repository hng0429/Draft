import cv2, numpy as np, json
SRC='/root/.claude/uploads/5f9885ee-a0d7-5eb5-8859-692f5b1d6250/c7ec5198-image.jpg'
img=cv2.imread(SRC); H,W=img.shape[:2]
L={int(k):np.array(v) for k,v in json.load(open('landmarks.json')).items()}
KEEP={
 0:[(0,0),(556,0),(556,225),(548,240),(550,260),(550,280),(547,300),(540,320),(533,340),(528,360),(523,380),(518,400),(516,420),(518,430),(520,450),(521,470),(520,490),(518,500),(514,520),(511,540),(508,560),(503,580),(493,600),(485,610),(472,620),(455,637),(440,650),(420,662),(395,669),(365,667),(335,660),(305,650),(275,637),(245,620),(215,602),(190,585),(165,566),(140,547),(118,528),(100,512),(80,520),(0,520)],
 1:[(554,30),(554,270),(549,290),(546,305),(546,325),(550,340),(556,352),(554,380),(555,410),(562,440),(572,460),(586,480),(600,497),(620,507),(640,515),(660,519),(680,520),(700,517),(720,511),(740,503),(757,498),(768,486),(778,472),(787,467),(800,470),(878,470),(878,30)],
 2:[(878,95),(878,468),(905,470),(930,476),(946,474),(950,452),(958,451),(970,458),(990,465),(1010,469),(1030,469),(1050,464),(1068,455),(1082,444),(1084,462),(1088,489),(1276,489),(1276,95)],
 3:[(1344,374),(1344,552),(1329,598),(1319,638),(1314,660),(1313,700),(1312,760),(1317,777),(1319,838),(1352,838),(1370,850),(1395,862),(1420,870),(1450,875),(1475,876),(1500,870),(1520,858),(1538,842),(1552,822),(1562,800),(1569,781),(1584,779),(1584,735),(1600,728),(1601,620),(1601,606),(1700,606),(1700,374)],
}
# --- fix: paint out fingertip overlapping head 3's hair (fill with nearby hair colour)
hair=np.median(img[590:615,1580:1598].reshape(-1,3),axis=0)
fm=np.zeros((H,W),np.uint8)
cv2.fillPoly(fm,[np.array([(1573,641),(1577,627),(1587,619),(1606,616),(1606,672),(1588,670),(1578,660)],np.int32)],255,cv2.LINE_AA)
fm=cv2.GaussianBlur(fm,(0,0),1.5).astype(np.float32)[...,None]/255
rng=np.random.default_rng(1); streak=cv2.GaussianBlur(rng.normal(0,1,(H,W)).astype(np.float32),(0,0),sigmaX=0.8,sigmaY=5)*40
fill=np.clip(hair[None,None,:]+streak[...,None],0,255)
img=(img*(1-fm)+fill*fm).astype(np.uint8)
SS=3  # supersample for polygon AA
FH=300; CW,CH=720,840; CHIN=(360,620)
for i in range(4):
    X0,Y0,X1,Y1=json.load(open(f'crop_{i}.json'))
    m=cv2.imread(f'matte_{i}.png',0).astype(np.float32)/255
    full=np.zeros((H,W),np.float32); full[Y0:Y1,X0:X1]=m
    pm=np.zeros((H*SS//1,W*SS//1),np.uint8) if False else None
    poly=np.array(KEEP[i],np.float32)
    k=np.zeros((H,W),np.uint8); cv2.fillPoly(k,[np.round(poly*16).astype(np.int32)],255,cv2.LINE_AA,shift=4)
    k=cv2.GaussianBlur(k,(0,0),1.0).astype(np.float32)/255
    a=full*k
    # largest component
    n,lab,st,_=cv2.connectedComponentsWithStats((a>0.5).astype(np.uint8))
    big=1+np.argmax(st[1:,4]); comp=cv2.dilate((lab==big).astype(np.uint8),np.ones((25,25),np.uint8))
    a*=comp
    # foreground colour estimate for soft edge pixels (de-spill)
    solid=(a>0.92).astype(np.float32)
    col=img.astype(np.float32)
    num=cv2.GaussianBlur(col*solid[...,None],(0,0),4); den=cv2.GaussianBlur(solid,(0,0),4)[...,None]
    fgc=num/np.maximum(den,1e-4)
    w=np.clip((0.92-a)/0.5,0,1)[...,None]*(den>0.02)
    col=col*(1-w)+fgc*w
    # similarity transform: eyes level, face height FH, chin at CHIN
    p=L[i]
    ang=np.arctan2(p[473,1]-p[468,1],p[473,0]-p[468,0])
    if abs(ang)>np.pi/2: ang-=np.sign(ang)*np.pi
    fh=np.linalg.norm(p[10,:2]-p[152,:2]); s=FH/fh
    c,sn=np.cos(-ang)*s,np.sin(-ang)*s
    M=np.array([[c,-sn,0],[sn,c,0]],np.float64)
    M[:,2]=np.array(CHIN)-M[:,:2]@p[152,:2]
    rgb=cv2.warpAffine(col,M,(CW,CH),flags=cv2.INTER_CUBIC if s>1 else cv2.INTER_AREA,borderValue=0)
    al=cv2.warpAffine(a,M,(CW,CH),flags=cv2.INTER_LINEAR,borderValue=0)
    if s>1.05:  # gentle sharpening for upscaled heads
        bl=cv2.GaussianBlur(rgb,(0,0),1.6); rgb=np.clip(rgb+0.5*(rgb-bl),0,255)
    lm=np.zeros((478,3)); lm[:,:2]=(M[:,:2]@p[:,:2].T).T+M[:,2]; lm[:,2]=p[:,2]*s
    np.savez_compressed(f'head_{i}.npz',rgb=np.clip(rgb,0,255).astype(np.uint8),alpha=np.clip(al*255,0,255).astype(np.uint8),lm=lm,scale=s,angle=ang)
    print(i,'scale',round(s,3),'roll deg',round(np.degrees(ang),1),'bbox',np.where(al>0.5)[1].min(),np.where(al>0.5)[0].min(),np.where(al>0.5)[1].max(),np.where(al>0.5)[0].max())
# preview on dark + bright
rows=[]
for bg in ((20,12,30),(235,225,205)):
    t=[]
    for i in range(4):
        d=np.load(f'head_{i}.npz'); a=d['alpha'][...,None]/255.
        t.append((d['rgb']*a+np.array(bg)*(1-a)).astype(np.uint8))
    rows.append(np.hstack(t))
cv2.imwrite('heads_v2.jpg',cv2.resize(np.vstack(rows),None,fx=0.66,fy=0.66,interpolation=cv2.INTER_AREA))
