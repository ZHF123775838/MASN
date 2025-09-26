def calc(TN, FP, FN, TP):
    recall = TP / (TP + FN)  # recall

    Precision = TP / (TP + FP)
   
    F1 = (2 * TP) / (2 * TP + FP + FN)
    
    return print(f'Recall:{recall:.4f} Precision:{Precision:.4f} F1:{F1:.4f}')

# conf_matrix of cnn xgb svm
conf_matrix_list = [[36646 ,354, 135, 3954],[36597 ,403, 107, 3982],[36378 ,622, 273, 3816]]
method=['DCNN','XGB','SVM']

for i in range(3):
    print(f'The result of {method[i]} Normal:')
    tp,fn,fp,tn=conf_matrix_list[i]
    calc(tn, fp, fn, tp) 
    
    tn, fp, fn, tp = conf_matrix_list[i]
    print(f'The result of {method[i]} Dos::')
    calc(tn, fp, fn, tp)
       