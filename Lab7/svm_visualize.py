import numpy as np
import matplotlib.pyplot as plt
import math

def visualize(model,data=None,target=None,aux_models=[],colors=['blue','red'],axis=None):
    try:
        x_min,x_max,y_min,y_max=min(data[:,0]),max(data[:,0]),min(data[:,1]),max(data[:,1])
        data_color=np.array([colors[1] if p>0 else colors[0] for p in target])
    except:
        x_min,x_max,y_min,y_max=-0.5,1.5,-0.5,1.5
    xs,ys=np.meshgrid(np.arange(0.8*x_min,1.2*x_max,0.005*(x_max-x_min)),np.arange(0.8*y_min,1.2*y_max,0.005*(y_max-y_min)))
    predictions=model([[x,y] for x,y in zip(xs.reshape(-1),ys.reshape(-1))])
    predictions=predictions.reshape(xs.shape)
    if axis:
        axis.contourf(xs,ys,predictions.reshape(xs.shape),levels=1,colors=colors,alpha=0.2)
        for aux_model in aux_models:
            aux_predictions=aux_model([[x,y] for x,y in zip(xs.reshape(-1),ys.reshape(-1))])
            axis.contour(xs,ys,aux_predictions.reshape(xs.shape),levels=1,colors=['grey'],linewidths=0.5)
        axis.contour(xs,ys,predictions.reshape(xs.shape),levels=1,colors=['k'],linewidths=2)
        try:
            axis.scatter(data[:,0],data[:,1],color=data_color)
        except:
            pass
        axis.set_xlim((0.8*x_min,1.2*x_max))
        axis.set_ylim((0.8*y_min,1.2*y_max))
    else:
        plt.contourf(xs,ys,predictions.reshape(xs.shape),levels=1,colors=colors,alpha=0.2)
        for aux_model in aux_models:
            aux_predictions=aux_model([[x,y] for x,y in zip(xs.reshape(-1),ys.reshape(-1))])
            plt.contour(xs,ys,aux_predictions.reshape(xs.shape),levels=1,colors=['grey'],linewidths=0.5)
        plt.contour(xs,ys,predictions.reshape(xs.shape),levels=1,colors=['k'],linewidths=2)
        try:
            plt.scatter(data[:,0],data[:,1],color=data_color)
        except:
            pass
        plt.xlim((0.8*x_min,1.2*x_max))
        plt.ylim((0.8*y_min,1.2*y_max))