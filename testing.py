import pandas as pd
import math



def entropy(C):

    val = C.unique()
    entropy = 0
    for i in val:
        Pr = C.value_counts()[i]/C.count()
        entropy = (Pr * math.log(Pr))
        return -entropy

'''
def entropy_Aj(D,C):

    val = D.unique()

    entropy = 0
    for i in val:
        Pr = C.value_counts()[i]/C.count()
        entropy = (Pr * math.log(Pr))
        return -entropy


def Gain(D,A):
    return entropy(D) - entropy_Aj(D,A)

function selectSplittingAttribute(A,D,threshold);
information gain p0 := enthropy(D); 
for each Ai ∈ Ado p[Ai] := enthropyAi (D);
   Gain[Ai] = p0−p[Ai]; 
//compute info gain endfor 
best := arg(findMax(Gain[])); 
if Gain[best] >threshold then return best else return NULL;
'''


def selectSplittingAttribute(df,target, threshold):
    C = df[target]
    D = df.loc[:,df.columns != target]
    D0 = entropy(C)
    Gain = {}
    TypeDict = {}

    for curr in D.columns:
       A=D[curr].unique()
       type = ''
       if len(A) > 5 and isinstance(A[0], (int, float)):
           type = 'numeric'
       else:
           type = 'categorical'
       TypeDict.update({curr:type})
       if type == 'numeric':
           (split,gain)=findBestSplit(D[curr], C)
           Gain.update ({curr:gain})
       else:  
            length = len(D)
            entAj = 0
            for i in A:
                Cj = C[D[curr] == i]
                S=len(Cj)
                entAj = entAj + S/length *entropy(Cj)
            Gain.update({curr:D0-entAj})
    if max(Gain.values()) >  threshold:
        best = max(Gain, key=Gain.get)
        return (best, TypeDict[best])
    else:
        return (None, None)
    
def buildTree(df, target, threshold):
    print(df)
    (best_split, type) = selectSplittingAttribute(df,target,threshold)
    if best_split == None:
        print(f"Prediction: {df[target].mode }")
    else:
        splitD = list()
        if type == "numeric":
           (split,gain) = findBestSplit(df[best_split],df[target])
           edges = [f"{split} <=", f"{split} >"]
           splitD = [df[df[best_split]<=split].loc[:,df.columns != best_split], df[df[best_split]>split].loc[:,df.columns != best_split]]
        else:
            edges = df[best_split].unique()

            for i in edges:
                splitD.append(df[df[best_split]==i].loc[:,df.columns != best_split])
            
        for data in splitD:
            buildTree(data, target, threshold)


def findBestSplit(A, C):
    A= A.sort_values()
    D0 = entropy(C)
    maxgain = -1
    length = len(A)
    bestsplit =()
    findBestSplit
    for i in range(1, length):
        S1 = C[A.iloc[:i].index]
        S2 = C[A.iloc[i:].index]
        curr_gain = D0-(len(S1)/length*entropy(S1) + len(S2)/length*entropy(S2))
        if(curr_gain > maxgain):
            maxgain  = curr_gain
            bestsplit=(A.iloc[i],curr_gain)
    return bestsplit




def main():
    df = pd.read_csv('iris.data.csv')
    target = 'species'
    threshold = 0.3
    buildTree(df, target, threshold)

       


main()




