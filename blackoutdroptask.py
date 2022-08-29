# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np

# matplotlib 폰트설정
plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 25
plt.rcParams['figure.figsize'] = (20,10)

color_dict = ['#0000FF','#FF0000']

plt.rcParams['font.size'] = 22
plt.tight_layout()

## 데이터
# age_category = ['10대 이하', '20대', '30대', '40대', '50대', '60대 이상'] ## 연령 카테고리, x축 눈금에 표시될 라벨
index=['QTO','Full-offloading', 'Non-offloading', 'Q-Learning', 'DACO', 'HGOS']
data = np.array([853762,	855180,	0,	848068,	156312,	403210])
# num_patient = [3,3,5,7,8,9] ## 방문환자수
 
## 시각화
# plt.figure(figsize=(10,10)) ## Figure 생성 사이즈는 10 by 10
colors = ['blue','green','black','red','yellow','purple']
xtick_label_position = list(range(len(index))) ## x축 눈금 라벨이 표시될 x좌표
plt.xticks(xtick_label_position, index) ## x축 눈금 라벨 출력
 
plt.bar(xtick_label_position, data, color=colors, edgecolor="black") ## 바 차트 출력
 
# plt.title('지난달 연령별 방문환자 수',fontsize=20) ## 타이틀 출력
# plt.xlabel('연령') ## x축 라벨 출력
# leg = plt.legend().remove()
# plt.yticks([i for i in range(0,int(max(S_data))+5000, 5000)],[format(i,",") for i in range(0,int(max(S_data))+5000, 5000)])
plt.yticks([i for i in range(0, 1200000, 200000)], [format(i, ",") for i in range(0, 1200000, 200000)])
plt.ylabel('Amount of blackout drop task', labelpad=10) ## y축 라벨 출력
plt.show()
plt.savefig('blackoutdroptask.png',bbox_inches='tight')
plt.savefig('blackoutdroptask.pdf',bbox_inches='tight')