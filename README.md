# OpenCV-license-plate-recognition
# 一、說明
本專題主要使用 OpenCV 透過自行訓練並建立 Haar 特徵分類器「偵測車牌」模型，先偵測框選車牌及原始圖片轉換尺寸及偵測、然後將擷取車牌號碼產生圖形、再來是去除畸零地把邊緣的輪廓、雜訊、黑色部分去除、最後將用光學辨識軟體 OCR 辨識出車牌號碼，就會出現以下結果及結論。
# 二、相關文章
OpenCV最為人稱道為「人臉辨識」。使用OpenCV提供的Haar特徵分類器人臉模型，即可輕鬆偵測人臉位置。是否有辦法偵測其他物件嗎？如果要偵測前請先建立Haar特徵分類器模型，然後再建立Haar特徵分類器偵測物件。

目前許多停車場已使用自動車牌辨識系統經營以節省人力成本，我們希望一步一步地探究影像辨識的技術並模擬建立停車場自動車牌辨識系統，從無到有自行訓練並建立 Haar 特徵分類器「偵測車牌」模型，先偵測框選車牌，接著將用光學辨識軟體 OCR 辨識出車牌號碼。

Haar特徵是用來描繪一張圖片。Haar特徵是一個矩形區域，可進行選轉、平移、縮放等，有15個類型。

![image](https://github.com/LonelyCaesar/OpenCV-license-plate-recognition/assets/101235367/3321f368-9a6c-48b8-899a-056494d02a2c)

Haar特徵分類器可以幫我們在圖片或照片中偵測某特定物件是否存在，並可得知該物件的座標位置。這個套特定物件可以是人臉、交通標誌、動物等使用Haar特徵分類器模型分析。
# 三、實作
首先將要辨識的車牌圖片檔案更名為車牌號碼，使用OCR辨識車牌號碼後就可以將辨識結果與檔名名稱直接做比對，能不能辨識正確。

註：執行套件pip install pillow、pip install opencv-python。

### 1.	原始圖片轉換尺寸及偵測：
將所有數位相機拍攝或下載圖片尺寸轉換為300x225像素圖形，也使用<haar_carplate.xml>模型做偵測。
### 程式碼：
``` Python
import PIL
from PIL import Image
import glob
import shutil, os
from time import sleep
import cv2
import glob

def emptydir(dirname):         #清空資料夾
    if os.path.isdir(dirname): #資料夾存在就刪除
        shutil.rmtree(dirname)
        sleep(2)       #需延遲,否則會出錯
    os.mkdir(dirname)  #建立資料夾

def dirResize(src, dst):
    myfiles = glob.glob(src + '/*.JPG') #讀取資料夾全部jpg檔案
    emptydir(dst)
    print(src + ' 資料夾：')
    print('開始轉換圖形尺寸！')
    for f in myfiles:
        fname = f.split("\\")[-1]
        img = Image.open(f)
        img_new = img.resize((300, 225), PIL.Image.LANCZOS)  #尺寸300x225
        img_new.save(dst + '/' + fname)
    print('轉換圖形尺寸完成！\n')
files = glob.glob("predictPlate/*.jpg")
dirResize('predictPlate_sr', 'predictPlate')

for file in files:
    print('圖片檔案：' + file)
    img = cv2.imread(file)
    detector = cv2.CascadeClassifier('haar_carplate.xml')
    signs = detector.detectMultiScale(img, minSize=(76, 20), scaleFactor=1.1, minNeighbors=4)
    if len(signs) > 0 :
        for (x, y, w, h) in signs:          
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)  
            print(signs)
    else:
        print('沒有偵測到車牌！')
    
    cv2.imshow('Frame', img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if key == 113 or key==81:  #按q鍵結束
        break
```
### 執行結果：
![image](https://github.com/LonelyCaesar/OpenCV-license-plate-recognition/assets/101235367/e21ee51c-fabe-4183-bf84-ef569764cdcc)

### 2.	擷取車牌號碼圖形：
使用Haar特徵分類器<haar_carplate.xml>模型框選出車牌號碼，並將車牌號碼圖形擷取下來。
### 程式碼：

