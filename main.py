# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 13:43:48 2021

@author: Petkovic Uros 2020/3027
"""
#%% Ucitavanje potrebnih biblioteka i konstanti

import os
import numpy as np
import math
import cv2
from matplotlib import pyplot as plt

CYLINDER=False #Biramo hocemo li planarne ili cilindricne panorame
ALGORITAM='SIFT'   #SURF,SIFT,ORB
PLOT=False
FEATURE_THRESHOLD=0.01
DESCRIPTOR_SIZE=5
MATCHING_Y_RANGE=100

RANSAC_K=1000
RANSAC_THRESHOLD_DISTANCE=8

ALPHA_BLEND_WINDOW=20

FEATURE_CUT_X_EDGE=10
FEATURE_CUT_Y_EDGE=50

#%% Funkcije potrebne za izvrsavanje programa

def cylindrical_system(img,focal_distance):
    height,width,_=img.shape
    cylinder=np.zeros(shape=img.shape, dtype=np.uint8) 
    for y in range(-int(height/2), int(height/2)):
        for x in range(-int(width/2), int(width/2)):
            cylinder_x=focal_distance*math.atan(x/focal_distance)
            cylinder_y=focal_distance*y/math.sqrt(x**2+focal_distance**2)  
            cylinder_x = round(cylinder_x + width/2)
            cylinder_y = round(cylinder_y + height/2)
            if cylinder_x >= 0 and cylinder_x < width and cylinder_y >= 0 and cylinder_y < height:
                cylinder[cylinder_y][cylinder_x] = img[y+int(height/2)][x+int(width/2)]
    # Kropujem crnu pozadinu
    _, thresh=cv2.threshold(cv2.cvtColor(cylinder,cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    contours=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h=cv2.boundingRect(contours[0])
        
    return cylinder[y:y+h, x:x+w]

def matched_plot(img1,img2,parovi):
    _,offset,_=img1.shape
    plot_img=np.concatenate((img1,img2),axis=1)
    plt.figure(figsize=(10,10))
    plt.imshow(convertResult(plot_img))
    plt.title('Uparena obelezja dveju slika')
    for i in range(len(parovi)):
        plt.scatter(x=parovi[i][0][1], y=parovi[i][0][0], c='r')
        plt.plot([parovi[i][0][1], offset+parovi[i][1][1]], [parovi[i][0][0], parovi[i][1][0]], 'y-', lw=1)
        plt.scatter(x=offset+parovi[i][1][1], y=parovi[i][1][0], c='b')
    plt.show()
    
def RANSAC(parovi,prev_shift):
    parovi=np.asarray(parovi)  
    if len(parovi)>RANSAC_K:
        random_constant=True
    else:
        random_constant=False
    best_shift=[]
    if random_constant:
        K=RANSAC_K
    else:
        K=len(parovi)
    max_inliner=0
    for k in range(K):
        # Random izabrani par poklapajucih obelezja
        if random_constant:
            id=int(np.random.random_sample()*len(parovi))
        else:
            id=k
        sample=parovi[id]
        # Podesavanje uvijenog modela
        shift=sample[1]-sample[0]
        # Racunanje inliner tacaka
        shifted=parovi[:,1]-shift
        razlika=parovi[:,0]-shifted    
        inliner=0
        for diff in razlika:
            if np.sqrt((diff**2).sum())<RANSAC_THRESHOLD_DISTANCE:
                inliner=inliner+1   
        if inliner>max_inliner:
            max_inliner=inliner
            best_shift=shift

    if prev_shift[1]*best_shift[1]<0:
        print('Najbolje pomeranje',best_shift)
        raise ValueError('Smer pomeranja nije isti kao prethodni.')

    return best_shift

def cropping(img):
    _,threshold=cv2.threshold(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),1,255,cv2.THRESH_BINARY)
    upper,lower=[-1, -1]
    black_threshold=img.shape[1]//100
    for y in range(threshold.shape[0]):
        if len(np.where(threshold[y]==0)[0])<black_threshold:
            upper=y
            break   
    for y in range(threshold.shape[0]-1,0,-1):
        if len(np.where(threshold[y]==0)[0])<black_threshold:
            lower=y
            break

    return img[upper:lower, :]

def align_image(img,shifts):
    sum_y,sum_x =np.sum(shifts,axis=0)
    y_shift=np.abs(sum_y)
    col_shift=None

    # Isti znak
    if sum_x*sum_y>0:
        col_shift=np.linspace(y_shift,0,num=img.shape[1],dtype=np.uint16)
    else:
        col_shift=np.linspace(0,y_shift,num=img.shape[1],dtype=np.uint16)
    aligned_image=img.copy()
    for x in range(img.shape[1]):
        aligned_image[:,x]=np.roll(img[:,x],col_shift[x],axis=0)

    return aligned_image

def alpha_blend(row1,row2,porub,window,direction='left'):
    if direction=='right':
        row1,row2=row2,row1
    new_row=np.zeros(shape=row1.shape,dtype=np.uint8)
    for x in range(len(row1)):
        color1=row1[x]
        color2=row2[x]
        if x<porub-window:
            new_row[x]=color2
        elif x>porub+window:
            new_row[x]=color1
        else:
            ratio=(x-porub+window)/(window*2)
            new_row[x]=(1-ratio)*color2+ratio*color1

    return new_row

def stitching_images(img1,img2,shift,blending=True):
    padding = [(shift[0], 0) if shift[0]>0 else (0, -shift[0]),(shift[1], 0) if shift[1]>0 else (0, -shift[1]),(0, 0)]
    shifted_img1=np.lib.pad(img1,padding,'constant',constant_values=0)
    # Isecanje nepotrebnoh regiona sa slike
    split=img2.shape[1]+abs(shift[1])
    if shift[1]>0:
        splited=shifted_img1[:,split:]
    else:
        splited=shifted_img1[:, :-split]
    if shift[1]>0:
       shifted_img1 = shifted_img1[:, :split]
    else:
        shifted_img1=shifted_img1[:, -split:]

    h1,w1,_=shifted_img1.shape
    h2,w2,_=img2.shape
    inv_shift=[h1-h2,w1-w2]
    inv_padding=[(inv_shift[0], 0) if shift[0] < 0 else (0, inv_shift[0]),(inv_shift[1], 0) if shift[1] < 0 else (0, inv_shift[1]),(0, 0)]
    shifted_img2=np.lib.pad(img2,inv_padding,'constant',constant_values=0)
    if shift[1]>0:
        direction='left'
    else:
        direction='right'
        
    if blending:
        porub=shifted_img1.shape[1]//2
        shifted_img=[alpha_blend(shifted_img1[y],shifted_img2[y],porub,ALPHA_BLEND_WINDOW,direction) for y in range(h1)]
        shifted_img=np.asarray(shifted_img)
        if shift[1]>0:
            shifted_img=np.concatenate((shifted_img,splited),axis=1)
        else:
            shifted_img=np.concatenate((splited, shifted_img), axis=1)
        shifted_img1=shifted_img
    else:
        raise ValueError('Nije upotrebljeno blendovanje"')

    return shifted_img1

def R_parameter(xx_row, yy_row, xy_row,k):
    R_matrix=np.zeros(shape=xx_row.shape,dtype=np.float32)
    for x in range(len(xx_row)):
        det_M=xx_row[x]*yy_row[x]-xy_row[x]**2
        trace_M=xx_row[x]+yy_row[x]
        R=det_M-k*trace_M**2
        R_matrix[x]=R

    return R_matrix

def harris_dots(img,k=0.04,block_size=2):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray=np.float32(gray)/255
    height,width,_=img.shape
    dx=cv2.Sobel(gray, -1, 1, 0)
    dy=cv2.Sobel(gray, -1, 0, 1)
    Ixx=dx*dx
    Iyy=dy*dy
    Ixy=dx*dy
    cov_xx=cv2.boxFilter(Ixx,-1,(block_size, block_size),normalize=False)
    cov_yy=cv2.boxFilter(Iyy,-1,(block_size, block_size), normalize=False)
    cov_xy=cv2.boxFilter(Ixy,-1,(block_size, block_size), normalize=False)
    harris_response=[R_parameter(cov_xx[y],cov_yy[y],cov_xy[y],k) for y in range(height)]
          
    return np.asarray(harris_response)

def matching_features(descriptor1,descriptor2,feature_position1,feature_position2,y_range=10):
    TASKS_NUM = 32
    partition_descriptors=np.array_split(descriptor1, TASKS_NUM)
    partition_positions=np.array_split(feature_position1, TASKS_NUM)
    results=[compute_match(partition_descriptors[i],descriptor2,partition_positions[i],feature_position2, y_range) for i in range(TASKS_NUM)]   
    parovi_poklapanja=[]
    for res in results:
        if len(res)>0:
            parovi_poklapanja+=res

    return parovi_poklapanja

def compute_match(descriptor1,descriptor2,feature_position1,feature_position2,y_range=10):
    parovi_poklapanja=[]
    parovi_poklapanja_rank=[]
    
    for i in range(len(descriptor1)):
        distances=[]
        y=feature_position1[i][0]
        for j in range(len(descriptor2)):
            diff=float('Inf')
            
            # Uporedjuju se samo ona obelezja koja imaju slicnu y-osu 
            if y-y_range<=feature_position2[j][0]<=y+y_range:
                diff=descriptor1[i]-descriptor2[j]
                diff=(diff**2).sum()
            distances+=[diff]
        sorted_index=np.argpartition(distances, 1)
        local_optimal=distances[sorted_index[0]]
        local_optimal2=distances[sorted_index[1]]
        if local_optimal>local_optimal2:
            local_optimal,local_optimal2=local_optimal2,local_optimal
        
        if local_optimal/local_optimal2<=0.5:
            upareni_index=np.where(distances==local_optimal)[0][0]
            par=[feature_position1[i],feature_position2[upareni_index]]
            parovi_poklapanja+=[par]
            parovi_poklapanja_rank+=[local_optimal]

    # Refine
    sortirani_rank_id=np.argsort(parovi_poklapanja_rank)
    sortirani_parovi_poklapanja=np.asarray(parovi_poklapanja)
    sortirani_parovi_poklapanja=sortirani_parovi_poklapanja[sortirani_rank_id]

    refined_parovi=[]
    for item in sortirani_parovi_poklapanja:
        duplicated=False
        for refined_item in refined_parovi:
            if refined_item[1]==list(item[1]):
                duplicated=True
                break
        if not duplicated:
            refined_parovi+=[item.tolist()]
            
    return refined_parovi

def extract_features(img,harris_response,threshold=0.01,kernel=3):
    height,width=harris_response.shape

    # Redukovanje
    features=np.zeros(shape=(height, width),dtype=np.uint8)
    features[harris_response>threshold*harris_response.max()]=255

    # Isecanje ivica slike
    features[:FEATURE_CUT_Y_EDGE, :] = 0  
    features[-FEATURE_CUT_Y_EDGE:, :] = 0
    features[:, -FEATURE_CUT_X_EDGE:] = 0
    features[:, :FEATURE_CUT_X_EDGE] = 0
    
    # Redukovanje obelezja lokalnim maksimumom
    window=3
    for y in range(0, height-10, window):
        for x in range(0, width-10, window):
            if features[y:y+window, x:x+window].sum()==0:
                continue
            block=harris_response[y:y+window, x:x+window]
            max_y,max_x = np.unravel_index(np.argmax(block),(window,window))
            features[y:y+window, x:x+window]=0
            features[y+max_y][x+max_x]=255
    feature_positions=[]
    feature_descriptions=np.zeros(shape=(1,kernel**2),dtype=np.float32)
    half_k = kernel//2
    for y in range(half_k,height-half_k):
        for x in range(half_k,width-half_k):
            if features[y][x]==255:
                feature_positions+=[[y,x]]
                desc=harris_response[y-half_k:y+half_k+1,x-half_k:x+half_k+1]
                feature_descriptions=np.append(feature_descriptions,[desc.flatten()],axis=0)
                
    return feature_descriptions[1:],feature_positions

def drawKeypoints(img, kp):
    img1=img
    cv2.drawKeypoints(img,kp,img1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img1

def extract_features2(image, opt="SIFT"):
    # Prebacivanje u sivu sliku
    grayImage=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    if opt=="SURF":
        md=cv2.xfeatures2d.SURF_create()
    if opt=="ORB":
        md=cv2.ORB_create(nfeatures=3000)
    if opt=="SIFT":
        md=cv2.xfeatures2d.SIFT_create(nfeatures=3000)
    # Find interest points and Computing features.
    keypoints,features=md.detectAndCompute(grayImage,None)
    # Konvertovanje u brojeve
    # keypoints=np.float32(keypoints)
    features=np.float32(features)
    return features,keypoints

def convertResult(img):
    img = np.array(img,dtype=float)/float(255)
    img = img[:,:,::-1]
    return img
#%% Glavni program

# Folder u kome se nalaze slike
folder="D:/MASTER RAD UROS PETKOVIC/Kalemegdan/"

# Citanje slika i ziznih daljina iz tekstualnog fajla
imgnames=[]
focal_distance=[]
f=open(os.path.join(folder, 'imglist.txt'))
for line in f:
    if(line[0]=='#'):
        continue
    (imgname,f,*rest)=line.split()
    imgnames+=[imgname]
    focal_distance+=[float(f)]
img_list = [cv2.imread(os.path.join(folder, i), 1) for i in imgnames]

if CYLINDER:
    
    # Ako radimo cilindricnu panoramu, vrsimo cilindricnu projekciju
    print('Prebacivanje slika u cilindricnu projekciju')
    print('Ovim postupkom prenosimo slike u cilindricni sistem')
    img_list=[cylindrical_system(img_list[i],focal_distance[i]) for i in range(0,len(img_list))]
    print('Kraj prebacivanja u cilindricne koordinate')

#%%
_,img_width,_=img_list[0].shape
spojena_slika=img_list[0].copy()
shifts=[[0, 0]]
features=[[], []]
for i in range(1,len(img_list)):
    print('Obrada '+str(i)+'. i '+str(i+1)+'. slike od '+str(len(img_list))+' slike.')
    img1=img_list[i-1]
    img2=img_list[i]
    if ALGORITAM=='HARRIS':
        print('Pronalazenje obelezja leve slike:')
        descriptors1,positions1=features
        if (len(descriptors1)==0):
            harris_response1=harris_dots(img1)
            descriptors1, positions1 =extract_features(img1,harris_response1,kernel=DESCRIPTOR_SIZE,threshold=FEATURE_THRESHOLD)
        print(str(len(descriptors1))+' obelezja leve slike je pronadjeno.')
        print('Pronalazenje obelezja desne slike:')
        harris_response2=harris_dots(img2)
        descriptors2, positions2=extract_features(img2,harris_response2,kernel=DESCRIPTOR_SIZE,threshold=FEATURE_THRESHOLD)
        print(str(len(descriptors2))+' obelezja desne slike je pronadjeno.')
        features=[descriptors2, positions2]
        if PLOT:
            cv2.imshow('Harisove tacke leve slike', harris_response1)
            cv2.imshow('Harisove tacke desne slike', harris_response2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            positions11=np.asarray(positions1)
            positions22=np.asarray(positions2)
            positions111=np.zeros(shape=(len(positions11[:,0]),len(positions11[:,1])))
            positions222=np.zeros(shape=(len(positions22[:,0]),len(positions22[:,1])))
            positions111[:,0]=positions11[:,1]
            positions111[:,1]=positions11[:,0]
            positions222[:,0]=positions22[:,1]
            positions222[:,1]=positions22[:,0]
            plt.figure(figsize=(10,10))
            plt.imshow(convertResult(img1))
            plt.scatter(positions111[:,0],positions111[:,1],c='r',s=5)
            plt.title('Pronadjena obelezja na levoj slici pomocu Harisovih tacaka')
            plt.show()
            plt.figure(figsize=(10,10))
            plt.imshow(convertResult(img2))
            plt.scatter(positions222[:,0],positions222[:,1],c='r',s=5)
            plt.title('Pronadjena obelezja na desnoj slici pomocu Harisovih tacaka')
            plt.show()
        print('Poklapanje obelezja dveju slika:')
        parovi_poklapanja=matching_features(descriptors1,descriptors2,positions1,positions2,y_range=MATCHING_Y_RANGE)
        print(str(len(parovi_poklapanja)) +' parova poklapanja obelezja dveju slika je pronadjeno.')
        if PLOT:
            matched_plot(img1,img2,parovi_poklapanja)
        print('RANSAC algoritam za uklanjanje outlier-a')
        shift =RANSAC(parovi_poklapanja,shifts[-1])
        shifts+=[shift]
        print('Najbolje namestanje: ',shift)
        print('Spajanje slika:')
        spojena_slika=stitching_images(spojena_slika,img2,shift,blending=True)
        cv2.imwrite('D:/MASTER RAD UROS PETKOVIC/'+ str(i) +'.jpg',spojena_slika)
        if PLOT:
            cv2.imshow('Spojene slike',spojena_slika)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        img11=img1.copy()
        img22=img2.copy()
        print('Pronalazenje obelezja leve slike:')
        obelezja1,pozicije1=extract_features2(img11,ALGORITAM)
        print(str(len(obelezja1))+' obelezja leve slike je pronadjeno.')
        print('Pronalazenje obelezja desne slike:')
        obelezja2,pozicije2=extract_features2(img22,ALGORITAM)
        print(str(len(obelezja2))+' obelezja desne slike je pronadjeno.')
        img1kp=drawKeypoints(img11,pozicije1)
        img2kp=drawKeypoints(img22,pozicije2)
        imgkp=np.concatenate((img1kp,img2kp), axis=1)
        plt.figure(figsize=(10,10))
        plt.imshow(convertResult(imgkp))
        plt.title('Pronadjena obelezja na slikama metodom '+str(ALGORITAM))
        plt.show()
        pozicije1=cv2.KeyPoint_convert(pozicije1)
        pozicije2=cv2.KeyPoint_convert(pozicije2)
        temp=np.zeros(shape=(pozicije1.shape[0],2))
        temp[:,0]=pozicije1[:,1]
        temp[:,1]=pozicije1[:,0]
        pozicije1=temp
        temp=np.zeros(shape=(pozicije2.shape[0],2))
        temp[:,0]=pozicije2[:,1]
        temp[:,1]=pozicije2[:,0]
        pozicije2=temp
        print('Poklapanje obelezja dveju slika:')
        parovi_poklapanja=matching_features(obelezja1,obelezja2,pozicije1,pozicije2,y_range=MATCHING_Y_RANGE)
        print(str(len(parovi_poklapanja)) +' parova poklapanja obelezja dveju slika je pronadjeno.')
        if PLOT:
            matched_plot(img1,img2,parovi_poklapanja)
        print('RANSAC algoritam za uklanjanje outlier-a')
        shift =RANSAC(parovi_poklapanja,shifts[-1])
        shift=np.array(shift,dtype='int')
        shifts+=[shift]
        print('Najbolje namestanje: ',shift)
        print('Spajanje slika:')
        spojena_slika=stitching_images(spojena_slika,img2,shift,blending=True)
        cv2.imwrite('D:/MASTER RAD UROS PETKOVIC/'+ str(i) +'.jpg',spojena_slika)
        if PLOT:
            cv2.imshow('Spojene slike',spojena_slika)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
print('Poravnanje slika na kraju:')
aligned_img=align_image(spojena_slika,shifts)
cv2.imwrite('D:/MASTER RAD UROS PETKOVIC/aligned_img.jpg', aligned_img)
if PLOT:
    cv2.imshow('Poravnata dobijena panorama',aligned_img)
print('Kropovanje slike')
cropped_img=cropping(aligned_img)
cv2.imwrite('D:/MASTER RAD UROS PETKOVIC/cropped_img.jpg',cropped_img)
if PLOT:
    cv2.imshow('Konacna dobijena poravnata isecena panorama',cropped_img)


#%% Ugradjenom OpenCV funkcijom
    
stitcher=cv2.createStitcher(try_use_gpu=True)
ret,panorama=stitcher.stitch(img_list)
cv2.imwrite('D:/MASTER RAD UROS PETKOVIC/ugradjenapanorama.jpg',panorama)
cv2.imshow('Panorama dobijena OpenCV ugradjenom funkcijom',panorama)  


