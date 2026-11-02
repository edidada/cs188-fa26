from torch.nn.functional import mse_loss, cross_entropy

def regression_loss(y_pred, y):
    return mse_loss(y_pred, y).reshape(1)



def digitclassifier_loss(y_pred, y):
    return cross_entropy(y_pred, y).reshape(1)


def languageid_loss(y_pred, y):
    return cross_entropy(y_pred, y).reshape(1)


def digitconvolution_Loss(y_pred, y):
    return cross_entropy(y_pred, y).reshape(1)
    
