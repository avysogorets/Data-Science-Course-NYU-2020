import numpy as np
import math

def generate(pts_per_class,sections):
    centers_1=[[-0.5,0.3],[0.1,0.5],[0.4,0.1]]
    X_1=np.array([np.random.multivariate_normal(mean=center,cov=0.01*np.eye(2),size=pts_per_class) for center in centers_1])
    centers_2=[[math.cos(k*math.pi/(2*sections-2)),np.sqrt(1-math.cos(k*math.pi/(2*sections-2))**2)] for k in range(0,sections+1)]
    X_2=np.array([np.random.multivariate_normal(mean=[math.cos(k*math.pi/(2*sections-2)),np.sqrt(1-math.cos(k*math.pi/(2*sections-2))**2)],cov=0.007*np.eye(2),size=pts_per_class) for k in range(0,sections+1)])
    data=np.concatenate([X_1.reshape((X_1.shape[0]*X_1.shape[1],X_1.shape[2])),X_2.reshape((X_2.shape[0]*X_2.shape[1],X_2.shape[2]))])
    data_y=np.array([[i]*pts_per_class for i in range(sections+4)]).reshape(-1)
    return data,data_y,np.squeeze(np.array([centers_1+centers_2]))

def visualize(model,axis,color=True):
    colors=['pink','orange','paleturquoise','dodgerblue','green','grey','navy','purple','c','violet','peru']
    xs,ys=np.meshgrid(np.arange(-0.75,1.25,0.001),np.arange(-0.5,1.5,0.001))
    predictions=model.predict([[x,y] for x,y in zip(xs.reshape(-1),ys.reshape(-1))])
    if color:
      axis.contourf(xs,ys,predictions.reshape(xs.shape),levels=len(np.unique(predictions)),colors=colors,alpha=0.5)
    axis.contour(xs,ys,predictions.reshape(xs.shape),levels=len(np.unique(predictions)),colors='k')