from sklearn.metrics import confusion_matrix
def businessgain(ytest,ypredict):
    TP = confusion_matrix(ytest,ypredict)[1,1]
    FP = confusion_matrix(ytest,ypredict)[0,1]
    FN = confusion_matrix(ytest,ypredict)[1,0]
    return (1876.7*(TP-FN)-8*20*(FP+TP))/((1876.7-8*20)*(TP+FN)) #8-hour investigation at $20/h 
