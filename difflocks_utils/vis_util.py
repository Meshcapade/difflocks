import torch
from torch.autograd import Function


class PCA(Function):
    @staticmethod
    def forward(ctx, sv): #sv corresponds to the slices values, it has dimensions nr_positions x val_full_dim

        # http://agnesmustar.com/2017/11/01/principal-component-analysis-pca-implemented-pytorch/


        X=sv.detach().cpu()#we switch to cpu of memory issues when doing svd on really big imaes
        k=3
        # print("x is ", X.shape)
        X_mean = torch.mean(X,0)
        # print("x_mean is ", X_mean.shape)
        X = X - X_mean.expand_as(X)

        # U,S,V = torch.svd(torch.t(X))
        U,S,V = torch.pca_lowrank( torch.t(X) )
        C = torch.mm(X,U[:,:k])
        # print("C has shape ", C.shape)
        # print("C min and max is ", C.min(), " ", C.max() )
        C-=C.min()
        C/=C.max()
        # print("after normalization C min and max is ", C.min(), " ", C.max() )

        return C
    

#img supposed to be N,C,H,W
def img_2_pca(img):
    assert img.shape[0]==1
    img_c=img.shape[1]
    img_h=img.shape[2]
    img_w=img.shape[3]
    vals = img.permute(0,2,3,1) #N,H,W,C
    c=PCA.apply(vals.view(-1,img_c))
    c=c.view(1, img_h, img_w, 3).permute(0,3,1,2) #N,H,W,C to N,C,H,W
    return c

