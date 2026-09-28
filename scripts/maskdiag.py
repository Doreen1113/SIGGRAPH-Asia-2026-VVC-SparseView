import zipfile, numpy as np, cv2, torch, torch.nn.functional as F, os, sys
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
V='/work/doreen071/vvc'; S=sys.argv[1]
seg=deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).cuda().eval().half()
mean_=torch.tensor([0.485,0.456,0.406]).cuda().view(1,3,1,1); std_=torch.tensor([0.229,0.224,0.225]).cuda().view(1,3,1,1)
z=zipfile.ZipFile(f'{V}/submissions/HB3_p085_nbx15s.zip')
t=lambda a: torch.from_numpy(a[...,::-1].copy()).permute(2,0,1)[None].cuda().float()/255
res={}
with torch.inference_mode():
  for c in ['004_1_seq0','006_1_seq0','007_0_seq0','009_0_seq0','011_0_seq0']:
    views=sorted(os.listdir(f'{V}/submissions/S5P_ens/renders/{c}')); miss=[]; cov=[]; k=0
    for v in views:
      fs=sorted(os.listdir(f'{V}/submissions/S5P_ens/renders/{c}/{v}'))
      for f in fs[::12]:
        R=t(cv2.imread(f'{V}/submissions/S5P_ens/renders/{c}/{v}/{f}')); B=t(cv2.imread(f'{V}/submissions/NBX15_pAV/renders/{c}/{v}/{f}'))
        C=t(cv2.imdecode(np.frombuffer(z.read(f'sparseViewTrack/renders/{c}/{v}/{f}'),np.uint8),1))
        pm=(seg(((C-mean_)/std_).half())['out'].float().softmax(1)[0,15]>0.5).float()
        ev=((R-B).abs().mean(1)[0]>0.12).float()
        ev=F.max_pool2d(-F.max_pool2d(-ev[None,None],5,1,2),5,1,2)[0,0]   # opening: drop specks
        pmd=F.max_pool2d(pm[None,None],25,1,12)[0,0]
        if ev.sum()>500:
          miss.append(((ev>0)&(pmd==0)).sum().item()/ev.sum().item()); cov.append(pm.mean().item())
        if c=='011_0_seq0' and k<3 and ev.sum()>500:
          m=((ev>0)&(pmd==0)).cpu().numpy()
          im=(C[0].permute(1,2,0).cpu().numpy()[...,::-1]*255).astype(np.uint8).copy()
          im[m]=(0,0,255); cnt,_=cv2.findContours((pm.cpu().numpy()>0).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
          cv2.drawContours(im,cnt,-1,(0,255,0),3)
          ys,xs=np.where((ev.cpu().numpy()>0)|(pm.cpu().numpy()>0))
          if len(ys):
            y0,y1,x0,x1=max(ys.min()-80,0),min(ys.max()+80,im.shape[0]),max(xs.min()-80,0),min(xs.max()+80,im.shape[1])
            cv2.imwrite(f'{S}/diag011_{v}_{f[:-4]}.jpg', cv2.resize(im[y0:y1,x0:x1],None,fx=0.5,fy=0.5))
          k+=1
    res[c]=(np.mean(miss),np.mean(cov),len(miss))
for c,(m,cv_,n) in res.items(): print(f'{c}: 人物證據像素中「沒被遮罩蓋到」的比例 {m*100:.1f}%   遮罩覆蓋 {cv_*100:.2f}%   (n={n})')
