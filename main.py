#coding:utf-8
from numpy import *
import cv2
def loadImageSet():
    FaceMat = mat(zeros((7*15,137*147)))
    j =0
    for f in range(15):
        if f+1<10:
            dir = "./dataset/00"+str(f+1)+"/"
        else:
            dir = "./dataset/0"+str(f+1)+"/"
        for i in range(7):
            try:
                img = cv2.imread(dir + "0"+str(i+1)+".jpg",0)
                print  dir + "0"+str(i+1)+".jpg"
            except:
                print 'load %s failed'%i
            FaceMat[j,:] = mat(img).flatten()
            j += 1
    return FaceMat

def ReconginitionVector(selecthr = 0.8):
    # 读入所有图像，存进FaceMat矩阵里面
    FaceMat = loadImageSet().T
    # 输出我们要的平均脸
    avgImg = mean(FaceMat,1)
    avgshow = avgImg.copy()
    cv2.imwrite("./avg.jpg",avgshow.reshape((147, 137)))
    # 计算特征脸
    diffTrain = FaceMat-avgImg
    for i in range(15):
        for j in range(7):
            eface = diffTrain[:,i*7+j].copy()
            cv2.imwrite("./eigenface/"+str(i+1)+"/0"+str(j+1)+".jpg",eface.reshape(147,137))
    # 计算协方差矩阵
    eigvals,eigVects = linalg.eig(mat(diffTrain.T*diffTrain))
    eigSortIndex = argsort(-eigvals)
    for i in xrange(shape(FaceMat)[1]):
        if (eigvals[eigSortIndex[:i]]/eigvals.sum()).sum() >= selecthr:
            eigSortIndex = eigSortIndex[:i]
            break
    covVects = diffTrain * eigVects[:,eigSortIndex] # covVects is the eigenvector of covariance matrix
    # avgImg 是均值图像，covVects是协方差矩阵的特征向量，diffTrain是偏差矩阵
    return avgImg,covVects,diffTrain

def judgeFace(judgeImg,FaceVector,avgImg,diffTrain,fi,st):
    diff = judgeImg.T - avgImg
    ot = diff.copy()
    #print "./eigenface/"+str(fi+1)+"/"+st+".jpg"
    cv2.imwrite("./eigenface/"+str(fi+1)+"/"+st+".jpg",ot.reshape(147,137))
    weiVec = FaceVector.T* diff
    res = 0
    resVal = inf
    for i in range(7*15):
        TrainVec = FaceVector.T*diffTrain[:,i]
        if  (array(weiVec-TrainVec)**2).sum() < resVal:
            res =  i
            resVal = (array(weiVec-TrainVec)**2).sum()
    return res//7 +1

if __name__ == '__main__':
    avgImg,FaceVector,diffTrain = ReconginitionVector(selecthr = 0.9)
    nameList = ['08','09','10','11']
    count = 0
    for te in range(15):
        for i in range(len(nameList)):
            if te+1<10:
                loadname = "./dataset/00"+str(te+1)+"/"+nameList[i]+".jpg"
            else :
                loadname = "./dataset/0"+str(te+1)+"/"+nameList[i]+".jpg"
            print loadname
            #从每个文件夹里选取测试集人像照片
            judgeImg = cv2.imread(loadname,0)
            if judgeFace(mat(judgeImg).flatten(),FaceVector,avgImg,diffTrain,te,nameList[i]) == te+1:
                #如果返回值和人像对应的文件夹id相同，则说明匹配正确
                count = count + 1
                print "Correct!\n"
            else:
                print "Incorrect!\n"

    print "accuracy:",count,"of",60,"are correct!"