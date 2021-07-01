import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression():
    def __init__(self,alpha,max_iter,C,tolerance=1e-5):
        self.lam=1/C
        self.alpha=alpha
        self.tolerance=tolerance
        self.max_iter=max_iter
    def gradient(self,X,y):
        
        # TO DO
        
        # Solution:
        term_1=sigmoid(np.matmul(self.theta,np.transpose(X)))
        term_2=np.matmul(y,X)
        return np.matmul(term_1,X)-term_2+2*self.lam*self.theta
    def fit(self,X,y):
        y=y.reshape((1,X.shape[0]))
        X=np.hstack([X,np.ones((X.shape[0],1))])
        self.theta=np.random.normal(loc=0,scale=1,size=X.shape[1]).reshape((1,X.shape[1]))
        progress=[0,compute_loss(X,y[0],self.theta[0],self.lam)]
        while self.max_iter>=0 and abs(progress[-1]-progress[-2])>self.tolerance:
            self.theta-=self.alpha*self.gradient(X,y)
            if self.max_iter%500==0:
                progress.append(compute_loss(X,y[0],self.theta[0],self.lam))
            self.max_iter-=1
        return progress[1:]
    def margin(self,X):
        X=np.hstack([X,np.ones((X.shape[0],1))])
        return np.squeeze(np.matmul(self.theta,np.transpose(X)))
    def predict(self,X,thresh):
        X=np.hstack([X,np.ones((X.shape[0],1))])
        return [1 if sigmoid(np.dot(self.theta,X_i))>thresh else 0 for X_i in X]
    def proba(self,X):
        X=np.hstack([X,np.ones((X.shape[0],1))])
        return np.squeeze(np.array([sigmoid(np.dot(self.theta,X_i)) for X_i in X]))

def compute_loss(X,y,theta,lam):
    
    # TO DO
    
    # Solution:
    return -sum([y_i*np.log(sigmoid(np.dot(theta,X_i)))+(1-y_i)*np.log(1-sigmoid(np.dot(theta,X_i))) for X_i,y_i in zip(X,y)])+lam*np.matmul(theta,np.transpose(theta))

def sigmoid(t):
    return 1./(1+np.exp(-t))

def confusion_matrix(predictions,labels):
    
    # TO DO

    # Solution:
    cm=np.zeros((len(np.unique(labels)),len(np.unique(labels))))
    for i,pred in enumerate(predictions):
        cm[pred,int(labels[i])]+=1
    return np.array(cm,dtype=np.int)

def precision(predictions,labels,pos_class=1):
    
    # TO DO
    
    # Solution:
    cm=confusion_matrix(predictions,labels)
    return cm[pos_class,pos_class]/sum(cm[pos_class])

def recall(predictions,labels,pos_class=1):
        
    # TO DO
    
    # Solution:
    cm=confusion_matrix(predictions,labels)
    return cm[pos_class,pos_class]/sum(cm[:,pos_class])

def f1_score(predictions,labels):
        
    # TO DO
    
    # Solution:
    r=recall(predictions,labels)
    p=precision(predictions,labels)
    return p*r/(p+r)

def roc(model,XX,yy,axis,inverted=False):
    
    # TO DO
    
    # Solution:
    auc=[]
    sensitivity,specificity=[],[]
    sensitivity.append(1)
    specificity.append(0)
    auc=0
    predictions=model.proba(XX)
    yy=yy[np.argsort(predictions)]
    predictions=sorted(predictions)
    for j,prediction in enumerate(predictions):
        if yy[j]>0:
            specificity.append(specificity[-1])
            sensitivity.append(sensitivity[-1]-1/sum([y>0 for y in yy]))
        else:
            sensitivity.append(sensitivity[-1])
            specificity.append(specificity[-1]+1/sum([y==0 for y in yy]))
            auc+=(specificity[-1]-specificity[-2])*sensitivity[-1]
    if not inverted:
        axis.plot([1-specif for specif in specificity],sensitivity,color='orange',linewidth=3,alpha=1)
        axis.set_xlabel("1-specificity (FPR)")
        axis.plot([0,1],[0,1],linestyle='dashed',color='k')
    else:
        axis.plot(specificity,sensitivity,color='pink',linewidth=3,alpha=1)
        axis.set_xlabel("specificity (TNR)")
        axis.plot([0,1],[1,0],linestyle='dashed',color='k')
    axis.set_ylabel("sensitivity (TPR)")
    axis.grid()