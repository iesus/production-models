def preIncre(name, local={}):
    #Equivalent to ++name
    if name in local:
        local[name]+=1
        return local[name]
    globals()[name]+=1
    return globals()[name]

def xplusplus(name, local={}):
    #Equivalent to name++
    if name in local:
        local[name]+=1
        return local[name]-1
    globals()[name]+=1
    return globals()[name]-1