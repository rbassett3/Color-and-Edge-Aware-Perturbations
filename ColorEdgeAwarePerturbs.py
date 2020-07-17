#Authors: Mitchell Graves and Robert Bassett
import torch

def pert_lab(image, label, grad_fun, num_iter, eps, weight=None):
    '''image is in Lab space
    grad_fun is a function which generates a gradient for a Lab-space image
    eps is either a sequence of length num_iter or a constant
    num_iter is the number of iterations to be performed
    weight is a vector which determines the constaint in each pixel as eps*weight[i].
    should be of length image.flatten()/3, one for each pixel. I imagine this is for the sobel filter.
    It can also be a scalar, but this is dumb and you should just put the weight into epsilon
    outputs delta. A vector of shape image such that image + delta is the perturbed image.'''
    if weight is None:
        weight = 1
    delta = torch.zeros(image.shape)
    for i in range(num_iter):
        grad = grad_fun(image + delta)
        delta += weight*eps*grad/torch.norm(grad, p=2, dim=1) #can be parallelized across pixels
    return delta

def pert_rgb(image, label, model, num_iter, eps, targeted = False, weight=None, do_imagenet_scale=True):
    '''
    image is in RGB space and in [0,1]
    model maps rgb image to the logits
    eps is either a sequence of length num_iter or a constant
    num_iter is the number of iterations to be performed
    weight is a vector which determines the constaint in each pixel as eps*weight[i].
    should be of length image.flatten()/3, one for each pixel. This can be use for an edge filter.
    It can also be a scalar, but this is dumb and you should just put the weight into epsilon
    outputs delta. A vector of shape image such that image + delta is the perturbed image.'''
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    img_lab = rgb2lab(image)
    if targeted == False:
        grad_fun = lambda img: grad_lab2lab(model, img, label, do_imagenet_scale=do_imagenet_scale)
    else:
        grad_fun = lambda img: -grad_lab2lab(model, img, label, do_imagenet_scale=do_imagenet_scale)
    delta_lab = pert_lab(img_lab, label, grad_fun, num_iter, eps, weight=weight)
    pert_img = torch.clamp(lab2rgb(img_lab + delta_lab), 0, 1)
    return(pert_img)

def grad_lab2lab(model, input_img, label, do_imagenet_scale=True):
    '''img assumed to be in [0,1].
    If the model uses the typical scaling of imagenet used in pytorch set do_imagenet_scale=True.
    See https://pytorch.org/docs/stable/torchvision/models.html'''
    from torch.nn import CrossEntropyLoss
    loss = CrossEntropyLoss()

    model.eval() #make sure the model isn't in training mode
    #make sure the shapes work are 4D
    if torch.is_tensor(label) is False:
        label = torch.tensor(label)
    if len(label.shape) == 0: #it's a scalar
        label=label.unsqueeze(0)
    img = input_img
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    img.requires_grad=True
    rgb_img = lab2rgb(img)
    if do_imagenet_scale:
        scaled_img = imagenet_transform(rgb_img)
    else:
        scaled_img = rgb_img
    out = loss(model(scaled_img), label)
    #out = loss(model(rgb_img), label)
    out.backward()
    return(img.grad)

def imagenet_transform(img):
    means = torch.tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1)
    sds = torch.tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1)
    if img.is_cuda:
        means = means.cuda()
        sds = sds.cuda()
    return (img - means)/sds

#source for these functions below (MIT License)
#https://github.com/richzhang/colorization-pytorch
def rgb2xyz(rgb): # rgb from [0,1]
    # xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
        # [0.212671, 0.715160, 0.072169],
        # [0.019334, 0.119193, 0.950227]])

    mask = (rgb > .04045).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (((rgb+.055)/1.055)**2.4)*mask + rgb/12.92*(1-mask)

    x = .412453*rgb[:,0,:,:]+.357580*rgb[:,1,:,:]+.180423*rgb[:,2,:,:]
    y = .212671*rgb[:,0,:,:]+.715160*rgb[:,1,:,:]+.072169*rgb[:,2,:,:]
    z = .019334*rgb[:,0,:,:]+.119193*rgb[:,1,:,:]+.950227*rgb[:,2,:,:]
    out = torch.cat((x[:,None,:,:],y[:,None,:,:],z[:,None,:,:]),dim=1)
    return out

def xyz2lab(xyz):
    # 0.95047, 1., 1.08883 # white
    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    if(xyz.is_cuda):
        sc = sc.cuda()
    xyz_scale = xyz/sc
    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if(xyz_scale.is_cuda):
        mask = mask.cuda()
    xyz_int = xyz_scale**(1/3.)*mask + (7.787*xyz_scale + 16./116.)*(1-mask)
    L = 116.*xyz_int[:,1,:,:]-16.
    a = 500.*(xyz_int[:,0,:,:]-xyz_int[:,1,:,:])
    b = 200.*(xyz_int[:,1,:,:]-xyz_int[:,2,:,:])
    out = torch.cat((L[:,None,:,:],a[:,None,:,:],b[:,None,:,:]),dim=1)
    return out

def xyz2rgb(xyz):
    # array([[ 3.24048134, -1.53715152, -0.49853633],
    #        [-0.96925495,  1.87599   ,  0.04155593],
    #        [ 0.05564664, -0.20404134,  1.05731107]])
    r = 3.24048134*xyz[:,0,:,:]-1.53715152*xyz[:,1,:,:]-0.49853633*xyz[:,2,:,:]
    g = -0.96925495*xyz[:,0,:,:]+1.87599*xyz[:,1,:,:]+.04155593*xyz[:,2,:,:]
    b = .05564664*xyz[:,0,:,:]-.20404134*xyz[:,1,:,:]+1.05731107*xyz[:,2,:,:]
    rgb = torch.cat((r[:,None,:,:],g[:,None,:,:],b[:,None,:,:]),dim=1)
    rgb = torch.max(rgb,torch.zeros_like(rgb)) # sometimes reaches a small negative number, which causes NaNs
    mask = (rgb > .0031308).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()
    rgb = (1.055*(rgb**(1./2.4)) - 0.055)*mask + 12.92*rgb*(1-mask)
    return rgb

def lab2xyz(lab):
    y_int = (lab[:,0,:,:]+16.)/116.
    x_int = (lab[:,1,:,:]/500.) + y_int
    z_int = y_int - (lab[:,2,:,:]/200.)
    if(z_int.is_cuda):
        z_int = torch.max(torch.Tensor((0,)).cuda(), z_int)
    else:
        z_int = torch.max(torch.Tensor((0,)), z_int)
    out = torch.cat((x_int[:,None,:,:],y_int[:,None,:,:],z_int[:,None,:,:]),dim=1)
    mask = (out > .2068966).type(torch.FloatTensor)
    if(out.is_cuda):
        mask = mask.cuda()
    out = (out**3.)*mask + (out - 16./116.)/7.787*(1-mask)
    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    sc = sc.to(out.device)
    out = out*sc
    return out

def rgb2lab(rgb):
    return xyz2lab(rgb2xyz(rgb))

def lab2rgb(lab):
    return xyz2rgb(lab2xyz(lab))

def get_probs(model, img):
    '''Given an image that has not had the imagenet_transorm done,
     obtain the vector probabilities for each label in the ImageNet Dataset'''
    return(torch.softmax(model(imagenet_transform(img)), 1))
