import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from statistics import mode

def margin(x,coefs):
    return np.dot(x,coefs)/np.linalg.norm(coefs[:-1])

models=[[-1.5,1,2.5],[0.5,1,-5.5],[4,1,-12]]
neg_pos=[[2,1],[3,1],[3,2]]
def visualize(models):
    xs,ys=np.meshgrid(np.arange(0,6.1,0.005),np.arange(0,6.1,0.005))
    data=[[x,y,1] for x,y in zip(xs.reshape(-1),ys.reshape(-1))]
    margins=np.array([[margin(x,model) for x in data] for i,model in enumerate(models)])
    binary_predictions=np.transpose(np.array([[neg_pos[model][int(m>=0)] for m in margins[model]] for model in range(3)]))
    final_predictions=np.zeros(len(data))
    margins=np.transpose(margins)
    for i,point in enumerate(binary_predictions):
        if len(np.unique(point))==2:
            final_predictions[i]=mode(point)
        else:
            confidences=[margins[i][0]+margins[i][1],margins[i][2]-margins[i][0],-margins[i][1]-margins[i][2]]
            final_predictions[i]=np.argmax(confidences)+1
    plt.contourf(xs,ys,final_predictions.reshape(xs.shape),levels=2,colors=['purple','pink','orange'],alpha=0.5)
    class1=mpatches.Patch(color='purple',alpha=0.5,label='class 1')
    class2=mpatches.Patch(color='pink',alpha=0.5,label='class 2')
    class3=mpatches.Patch(color='orange',alpha=0.5,label='class 3')
    plt.legend(handles=[class1,class2,class3])
    plt.contour(xs,ys,np.array([margin(x,models[0])>=0 for x in data]).reshape(xs.shape),levels=1,colors=['k'])
    plt.contour(xs,ys,np.array([margin(x,models[1])>=0 for x in data]).reshape(xs.shape),levels=1,colors=['k'])
    plt.contour(xs,ys,np.array([margin(x,models[2])>=0 for x in data]).reshape(xs.shape),levels=1,colors=['k'])
    plt.savefig("ovo.png",dpi=1000)

def margin(x,coefs):
    return np.dot(x,coefs)/np.linalg.norm(coefs[:-1])

models=[[0.5,1,-5.5],[-1.5,1,2.5],[4,1,-12]]
neg_pos=[[2,1],[3,1],[3,2]]
signs=[1,-1,-1]
def visualize_2(models):
    xs,ys=np.meshgrid(np.arange(0,6.1,0.005),np.arange(0,6.1,0.005))
    data=[[x,y,1] for x,y in zip(xs.reshape(-1),ys.reshape(-1))]
    margins=np.array([[signs[i]*margin(x,model) for x in data] for i,model in enumerate(models)])
    margins=np.transpose(margins)
    final_predictions=np.array([np.argmax(point)+1 for point in margins])
    plt.contourf(xs,ys,final_predictions.reshape(xs.shape),levels=2,colors=['purple','pink','orange'],alpha=0.5)
    class1=mpatches.Patch(color='purple',alpha=0.5,label='class 1')
    class2=mpatches.Patch(color='pink',alpha=0.5,label='class 2')
    class3=mpatches.Patch(color='orange',alpha=0.5,label='class 3')
    plt.legend(handles=[class1,class2,class3])
    plt.contour(xs,ys,np.array([margin(x,models[0])>=0 for x in data]).reshape(xs.shape),levels=1,colors=['k'])
    plt.contour(xs,ys,np.array([margin(x,models[1])>=0 for x in data]).reshape(xs.shape),levels=1,colors=['k'])
    plt.contour(xs,ys,np.array([margin(x,models[2])>=0 for x in data]).reshape(xs.shape),levels=1,colors=['k'])
    plt.savefig("ova.png",dpi=1000)