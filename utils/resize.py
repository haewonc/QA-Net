import torch 

def to_spatial(x, dim):
    if dim==4:
        # (B, T, X, Y) -> (BT, 1, X, Y)
        B, T, X, Y = x.size()
        x = x.view(B*T, 1, X, Y) 
    if dim==5:
        # (B, T, C, X, Y) -> (BT, C, X, Y)
        B, T, C, X, Y = x.size()
        x = x.view(B*T, C, X, Y)
    return x

def from_spatial(x, T):
    # (BT, C, X, Y) -> (B, T, C, X, Y)
    B, C, X, Y = x.size()
    x = x.view(B//T, T, -1, X, Y)
    return x 

def spatial_expand(x, T):
    # (BT, C, X, Y) -> (B, C, T, X, Y)
    x = from_spatial(x, T)
    x = x.transpose(2, 1)
    return x

def spatial_squeeze(x):
    # (B, C, T, X, Y) -> (BT, C, X, Y)
    B, C, T, X, Y = x.size()
    x = x.transpose(2, 1)
    x = x.reshape(B*T, C, X, Y)
    return x

def spatial_to_mha(x, T):
    B, C, X, Y = x.size()
    B = B//T
    x = spatial_expand(x, T)
    x = torch.reshape(x.permute(2,0,3,4,1),[T, B*X*Y, C])
    return x    

def temporal_to_spatial(x, C):
    # (BC, T, X, Y) -> (BT, C, X, Y)
    B, T, X, Y = x.size()
    B = B//C 
    x = x.reshape(B, C, T, X, Y)
    x = x.transpose(2, 1)
    x = x.reshape(B*T, C, X, Y)
    return x

def mha_to_spatial(x, S):
    # (T, BXY, C) -> (BT, C, X, Y)
    T, B, C = x.size()
    B = B//(S*S)
    X = S
    Y = S
    x = torch.reshape(x, [T, B, X, Y, C]).permute(1,4,0,2,3)
    x = spatial_squeeze(x)
    return x 