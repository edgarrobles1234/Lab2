import pandas as pd
import math


def entropy(C):
    if len(C.unique()) <= 1:
        return 0  # If only one unique value, entropy is 0
    
    val = C.unique()
    entropy = 0
    for i in val:
        Pr = C.value_counts()[i] / C.count()
        entropy -= Pr * math.log(Pr, 2)  # Use log base 2 for entropy
    return entropy


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


def buildTree(df, target, threshold, depth=0):
    (best_split, type) = selectSplittingAttribute(df,target,threshold)

    if best_split is None:
        leaf_value = df[target].mode()[0]  # Most common class as the leaf node
        print(leaf_value)
        #print("  " * depth + f"Leaf: {leaf_value}")
        return leaf_value

    #print("  " * depth + f"Split on {best_split}")

    tree = {best_split: {}}
    
    
    
    if type == "numeric":
        (split,gain) = findBestSplit(df[best_split],df[target])
        subset1 = df[df[best_split] <= split].drop(columns=[best_split])
        tree[best_split]["split 1"] = buildTree(subset1, target, threshold, depth + 2)
        subset2 = df[df[best_split] > split].drop(columns=[best_split])
        tree[best_split]["split 2"] = buildTree(subset2, target, threshold, depth + 2)
    else:
        edges = df[best_split].unique()
        for edge in edges:
            #print("  " * (depth + 1) + f"Edge: {edge}")
            subset = df[df[best_split] == edge].drop(columns=[best_split])
            tree[best_split][edge] = buildTree(subset, target, threshold, depth + 2)

    return tree





def main():
    
    df = pd.read_csv("iris.data.csv")
    target = 'species'
    threshold = 0.1
    tree = buildTree(df, target, threshold)
    #prediction = predict(tree, single_row)
    #print(prediction)



main()