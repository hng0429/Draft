import cv2, numpy as np, mediapipe as mp, sys, json
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
img = cv2.imread(sys.argv[1]); H,W = img.shape[:2]
opts = vision.FaceLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path='models/face_landmarker.task'),
    num_faces=6, min_face_detection_confidence=0.3, output_face_blendshapes=True, output_facial_transformation_matrixes=True)
det = vision.FaceLandmarker.create_from_options(opts)
def run(im):
    mpimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    return det.detect(mpimg)
res = run(img)
print('faces found (full image):', len(res.face_landmarks))
faces=[]
for f in res.face_landmarks:
    pts=np.array([[p.x*W,p.y*H,p.z*W] for p in f]); faces.append(pts)
    print(' bbox', pts[:,0].min().round(), pts[:,1].min().round(), pts[:,0].max().round(), pts[:,1].max().round())
# refine per-face on an upscaled crop for precision
boxes=[(0,40,575,730),(570,140,860,585),(880,200,1160,530),(1295,525,1600,895)]
out={}
vis=img.copy()
for i,(x0,y0,x1,y1) in enumerate(boxes):
    pad=int(0.35*max(x1-x0,y1-y0)); X0,Y0=max(0,x0-pad),max(0,y0-pad); X1,Y1=min(W,x1+pad),min(H,y1+pad)
    crop=img[Y0:Y1,X0:X1]; s=1024/max(crop.shape[:2]); cu=cv2.resize(crop,None,fx=s,fy=s,interpolation=cv2.INTER_CUBIC)
    r=run(cu)
    if not r.face_landmarks:
        print('face',i,'not found in crop'); continue
    # pick face with largest bbox
    best=max(r.face_landmarks,key=lambda f:(max(p.x for p in f)-min(p.x for p in f)))
    pts=np.array([[p.x*cu.shape[1]/s+X0,p.y*cu.shape[0]/s+Y0,p.z*cu.shape[1]/s] for p in best])
    out[i]=pts.tolist()
    for p in pts: cv2.circle(vis,(int(p[0]),int(p[1])),1,(0,255,0),-1)
    bs=r.face_blendshapes[r.face_landmarks.index(best)]
    print('face',i,'ok', len(pts), 'jawOpen', [round(b.score,2) for b in bs if b.category_name=='jawOpen'])
json.dump(out,open('landmarks.json','w'))
cv2.imwrite('lm_vis.jpg',vis)
