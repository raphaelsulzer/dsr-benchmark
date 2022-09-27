import numpy as np

pre = np.array([0.8,0.7,0.9,0.1])
rec = np.array([0.05,0.5,1.0,0.9])



def compute_f1(precision,recall):

    f1s = []
    for i in range(len(precision)):
        f1 = (2*precision[i]*recall[i])/(precision[i]+recall[i])
        f1s.append(f1)
    return np.array(f1s)

print(compute_f1([pre.mean()],[rec.mean()]))
print(compute_f1(pre,rec).mean())


