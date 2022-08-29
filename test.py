import pandas as pd
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt

# matplotlib 폰트설정
plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 25
plt.rcParams['figure.figsize'] = (24, 10)

color_dict = ['#0000FF', '#FF0000']

plt.rcParams['font.size'] = 22
colors = ['blue', 'green', 'orange', 'red', 'yellow', 'purple']

X = ["5 devices", "10 devices", "15 devices"]
# 데이터
# age_category = ['10대 이하', '20대', '30대', '40대', '50대', '60대 이상'] ## 연령 카테고리, x축 눈금에 표시될 라벨
index = ['QTO', 'Full-offloading',
    'Non-offloading', 'Q-Learning', 'DACO', 'HGOS']
droptask = np.array([19,	0,	432000,	2,	29601,	0])
blackouttask = np.array([853762,	855180,	0,	848068,	156312,	403210])
forblock = np.array([0,0,0,0,0,0])
forslash = np.array([0,0,0,0,0,0])

def plot_clustered_stacked(dfall, labels=None, title="",  H="///", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      edgecolor="black",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      color=['white','black'],
                      **kwargs)  # make bar plots

    k = 0
    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part     
                # rect.set_color(colors[int(i/n_col)])
                rect.set_width(1 / float(n_df + 1))
                
                if k >= 30:
                    rect.set_edgecolor('white')
                    rect.set_hatch("\\\\\\") #edited part    
                elif k < 30 and k >23 :
                    rect.set_hatch("\\\\\\") #edited part    
                elif k <= 23 and k > 17:
                    rect.set_edgecolor('white')
                    rect.set_hatch("///") #edited part     
                rect.set_linewidth(0.5)
                k+=1

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_title(title)
    

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        # n.append(axe.bar(0, 0, color="#3B75AF", hatch=H * i))
        n.append(axe.bar(0, 0, color="white", hatch=H * i, edgecolor="black"))
        # n.append(axe.bar(0, 0, color="gray", hatch=H * i))
    n[-1] = axe.bar(0, 0, color="white", hatch='\\\\\\', edgecolor="black")

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[0.675, 0.85])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[0.78, 0.63]) 
    axe.add_artist(l1)
    return axe

# blackouttask = np.array([[70697.5,	427590,	49448,	226672.8,	151327.5,	201605],
#                          [141554,	855180,	98901,	452480.8,	234468,	403210],
#                          [212092.5,	1282770,	0,	680018.4,	453982.5,	604815]])

droptask = np.array([[9358,	18688, 28074],
                    [0 , 0 , 0],
                    [216000, 432000, 648000],
                    [9386, 18973, 28158],
                    [15600, 29601, 46800],
                    [0, 0, 0]])

data1 = np.array([[70697.5, 9358],
                  [427590, 0],
                  [49448, 216000],
                  [226672.8, 9386],
                  [151327.5, 15600],
                  [201605,0]])
data2 = np.array([[141554, 18688],
                  [855180, 0],
                  [98901,432000],
                  [452480.8,18973],
                  [234468, 29601],
                  [403210,0]])
data3 = np.array([[212092.5, 28074],
                  [1282770, 0],
                  [148344, 648000],
                  [680018.4,28158],
                  [453982.5,46800],
                  [604815,0]])
# blackouttask = blackouttask1.T

data11 = pd.DataFrame(data=data1, index=index, columns=('Because of blackout', 'Because of latency'))
data22 = pd.DataFrame(data=data2, index= index, columns=('Because of blackout', 'Because of latency'))
data33 = pd.DataFrame(data=data3, index= index, columns=('Because of blackout', 'Because of latency'))
# Then, just call :
plot_clustered_stacked([data11, data22, data33],X)
plt.yticks([i for i in range(0, 1600000, 200000)], [format(i, ",") for i in range(0, 1600000, 200000)])
plt.ylabel('Total drop task', labelpad=10)  # y축 라벨 출력
plt.show()