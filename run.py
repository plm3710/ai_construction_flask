#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt


# In[2]:


from flask import Flask, redirect, request, render_template
from werkzeug.utils import secure_filename 
from matplotlib.pyplot import imread
from PIL import Image
import numpy as np
from urllib import parse

from tensorflow.keras.models import load_model
model = load_model("model/cnn_224_model1_epoch-0016_acc-0.9553.h5")

#결과 값
construction_list = ['안전난간', '작업발판', '개구부']
preventive_dic = {"안전난간": {"element":"안전난간",
                               "hazardousType":"안전난간 미설치로 인한 추락 위험",
                               "measures":"안전난간 안전사항",
                               "rule":"산업안전보건 기준에 관한 규칙 제13조(안전난간의 구조 및 설치요건)",
                               "measuresDetail":"상부 난간대 중간 난간대 발끝막이판 및 난간기둥으로 구성할 것;안전난간은 구조적으로 가장 취약한 지점에서 가장 취약한 방향으로 작용하는 100킬로그램 이상의 하중에 견딜 수 있는 튼튼한 구조일 것;상부 난간대는 바닥면·발판 또는 경사로의 표면으로부터 90cm 이상 지점에 설치할 것;난간 설치가 힘든 부분은 철골용 포스트 및 로프를 사용하여 추락위험방지 조치할 것"},
                  "작업발판": {"element":"작업발판",
                               "hazardousType":"작업발판 설치 상태 불량으로 인한 추락 위험",
                               "measures":"작업발판 안전사항",
                               "rule":"산업안전보건기준에 관한 규칙 제9조(작업발판 등)",
                               "measuresDetail":"발판재료는 작업할 때의 하중을 견딜 수 있도록 견고한 것으로 할 것;작업발판 폭은 40cm이상 발판재료 간의 틈은 3cm이하로 할 것;추락의 위험이 있는 장소에는 안전난간을 설치할 것;작업발판의 지지물은 하중에 의하여 파괴될 우려가 없는 것을 사용할 것;작업발판재료는 뒤집히거나 떨어지지 않도록 둘 이상의 지지물에 연결하거나 고정시킬 것;작업발판을 작업에 따라 이동시킬 경우에는 위험 방지에 필요한 조치를 할 것"},
                  "개구부" : {"element":"개구부",
                               "hazardousType":"개구부 덮개 미설치로 인한 추락 위험",
                               "measures":"안전난간 안전사항",
                               "rule":"산업안전보건기준에 관한 규칙 제43조(개구부 등의 방호 조치)",
                               "measuresDetail":"덮개를 설치하는 경우 뒤집히거나 떨어지지 않도록 설치하고 어두운 장소에서도 알아볼 수 있도록 개구부임을 표시할 것;근로자 추락위험 장소 안전난간 울타리 수직형 추락방망 또는 덮개 등의 방호 조치 충분한 강도를 가진 구조로 튼튼하게 설치할 것"}}






"""Flask"""
app = Flask(__name__)
@app.route("/construction",methods=['GET','POST'])
def construction():
    
    if request.method=="POST":
        f = request.files["file"] #이미지 파일 전송 받기
        fname = 'temp_image/'+str(secure_filename(f.filename)) 
        print(fname)
        f.save(fname)#이미지 저장
        
    img = imread(fname)
    img = np.array(Image.fromarray(img).resize((224,224))) #사이즈 변경
    img = img.reshape(-1, img.shape[0], img.shape[1], img.shape[2]) #4차원으로 변경
    pre = np.argmax(model.predict(img)) #예측결과 확인

    result = preventive_dic[construction_list[pre]]
    
    #사진 파일 이름 추가
    result['img_fname'] = fname
    
    #print("결과 : {}".format(construction_list[pre]))
    #print("실행 완료")
    
    #json 형식이나 쿼리스트링으로 전송시 문자열로 전송하기 위해 내용 처리
    #result = str(result).replace("'","").replace('"','').replace("{","").replace("}","").replace("[","").replace("]","").replace(":",",")
    
    #return redirect("http://192.168.0.111:5500/construction_result.html?result="+str(result))
    print(result)
    return result

if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('0.0.0.0', 9002, app)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




