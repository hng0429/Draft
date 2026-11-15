import cv2, numpy as np, onnxruntime as ort, json, sys, time
img = cv2.imread(sys.argv[1]); H,W = img.shape[:2]; i = int(sys.argv[2])
L = {int(k):np.array(v) for k,v in json.load(open('landmarks.json')).items()}
so = ort.SessionOptions(); so.intra_op_num_threads = 4; so.enable_cpu_mem_arena = False; so.enable_mem_pattern = False
sess = ort.InferenceSession('models/birefnet_portrait.onnx', so, providers=['CPUExecutionProvider'])
inp = sess.get_inputs()[0]
def matte(bgr):
    x = cv2.resize(bgr,(1024,1024),interpolation=cv2.INTER_LANCZOS4)[...,::-1].astype(np.float32)/255
    x = ((x-np.array([0.485,0.456,0.406]))/np.array([0.229,0.224,0.225])).transpose(2,0,1)[None].astype(np.float32)
    y = sess.run(None,{inp.name:x})[0]; y = 1/(1+np.exp(-y[0,0]))
    return cv2.resize(y,(bgr.shape[1],bgr.shape[0]),interpolation=cv2.INTER_LINEAR)
t=time.time()
if i<0:
    m = matte(img); cv2.imwrite('matte_full.png',(m*255).astype(np.uint8))
else:
    pts=L[i]; fh = np.linalg.norm(pts[10,:2]-pts[152,:2]); c = (pts[10,:2]+pts[152,:2])/2 - np.array([0,0.12*fh]); half = 1.35*fh
    x0,y0,x1,y1 = [int(v) for v in (c[0]-half,c[1]-half,c[0]+half,c[1]+half)]
    X0,Y0,X1,Y1 = max(0,x0),max(0,y0),min(W,x1),min(H,y1)
    crop = img[Y0:Y1,X0:X1]; side = max(crop.shape[:2]); pad = np.zeros((side,side,3),np.uint8); pad[:crop.shape[0],:crop.shape[1]] = crop
    mm = matte(pad)[:crop.shape[0],:crop.shape[1]]
    cv2.imwrite(f'matte_{i}.png',(mm*255).astype(np.uint8)); cv2.imwrite(f'crop_{i}.png',crop)
    json.dump([X0,Y0,X1,Y1],open(f'crop_{i}.json','w'))
    print('face',i,(X0,Y0,X1,Y1),'fh',round(fh),round(time.time()-t,1),'s')
