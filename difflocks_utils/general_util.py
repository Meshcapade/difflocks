import torch
import numpy as np
import sys
from functools import reduce
from torch.nn.modules.module import _addindent
import scipy.signal
import cv2
import torch.nn.functional as F
import random
from typing import Optional
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
def map_range_val( input_val, input_start, input_end,  output_start,  output_end):
    # input_clamped=torch.clamp(input_val, input_start, input_end)
    input_clamped=max(input_start, min(input_end, input_val))
    # input_clamped=torch.clamp(input_val, input_start, input_end)
    return output_start + ((output_end - output_start) / (input_end - input_start)) * (input_clamped - input_start)

def batched_index_select(t, dim, inds):
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy) # b x e x f
    return out

def tridiagonal_solve(b, A_upper, A_diagonal, A_lower):
    """Solves a tridiagonal system Ax = b.

    The arguments A_upper, A_digonal, A_lower correspond to the three diagonals of A. Letting U = A_upper, D=A_digonal
    and L = A_lower, and assuming for simplicity that there are no batch dimensions, then the matrix A is assumed to be
    of size (k, k), with entries:

    D[0] U[0]
    L[0] D[1] U[1]
         L[1] D[2] U[2]                     0
              L[2] D[3] U[3]
                  .    .    .
                       .      .      .
                           .        .        .
                        L[k - 3] D[k - 2] U[k - 2]
           0                     L[k - 2] D[k - 1] U[k - 1]
                                          L[k - 1]   D[k]

    Arguments:
        b: A tensor of shape (..., k), where '...' is zero or more batch dimensions
        A_upper: A tensor of shape (..., k - 1).
        A_diagonal: A tensor of shape (..., k).
        A_lower: A tensor of shape (..., k - 1).

    Returns:
        A tensor of shape (..., k), corresponding to the x solving Ax = b

    Warning:
        This implementation isn't super fast. You probably want to cache the result, if possible.
    """

    # This implementation is very much written for clarity rather than speed.

    A_upper, _ = torch.broadcast_tensors(A_upper[:, None, :], b[..., :-1])
    A_lower, _ = torch.broadcast_tensors(A_lower[:, None, :], b[..., :-1])
    A_diagonal, b = torch.broadcast_tensors(A_diagonal[:, None, :], b)

    channels = b.size(-1)

    new_b = np.empty(channels, dtype=object)
    new_A_diagonal = np.empty(channels, dtype=object)
    outs = np.empty(channels, dtype=object)

    new_b[0] = b[..., 0]
    new_A_diagonal[0] = A_diagonal[..., 0]
    for i in range(1, channels):
        w = A_lower[..., i - 1] / new_A_diagonal[i - 1]
        new_A_diagonal[i] = A_diagonal[..., i] - w * A_upper[..., i - 1]
        new_b[i] = b[..., i] - w * new_b[i - 1]

    outs[channels - 1] = new_b[channels - 1] / new_A_diagonal[channels - 1]
    for i in range(channels - 2, -1, -1):
        outs[i] = (new_b[i] - A_upper[..., i] * outs[i + 1]) / new_A_diagonal[i]

    return torch.stack(outs.tolist(), dim=-1)


def FDT(strand):
    codes = []
    N = strand.shape[0]
    count = 0

    for j in range(3):
        code = []
        t = np.linspace(0, 1, N, endpoint=False)
        signal = strand[:, j]

        # calculate DFT
        X = np.fft.fft(signal)

        # calculate magnitude and phase
        F_A = np.abs(X)
        # phase_spectrum = np.angle(X)
        # F_cos = X.real/F_A
        # F_sin = X.imag/F_A
        F_cos = X.real
        F_sin = X.imag
        # 频率索引

        # frequencies = np.fft.fftfreq(N, d=t[1] - t[0])
        # reconstructed_fft_result = F_A * np.exp(1j * phase_spectrum)
        # reconstructed_fft_result = np.fft.ifft(reconstructed_fft_result)

        count += 1
        code.append(F_A[:N//2+1])
        code.append(F_cos[:N//2+1])
        code.append(F_sin[:N//2+1])
        code = np.concatenate(code,0)[...,None]
        codes.append(code)
    codes = np.concatenate(codes,1)
    return codes


def inverse_FDT(code):

    N = code.shape[0]//3

    strand = []
    for i in range(3):
        signal = code[:,i]
        F_A = signal[:N]
        F_cos = signal[N:2*N]
        F_sin = signal[2*N:]


        # phase = F_cos * F_A + 1j*F_sin * F_A
        phase = F_cos + 1j*F_sin
        phase = np.angle(phase)
        phase = np.concatenate([phase,-1 * phase[1:-1][::-1]],0)


        F_A = np.concatenate([F_A,F_A[1:-1][::-1]],0)
        reconstructed_fft_result = F_A* np.exp(1j * phase)
        reconstructed_fft_result = np.fft.ifft(reconstructed_fft_result)
        strand.append(np.real(reconstructed_fft_result)[:,None])


    strand = np.concatenate(strand,1)

    return strand

def strands_from_signal(signal):
    strands = []
    signal = signal.detach().cpu().numpy()
    for s in signal:
        strand = inverse_FDT(s)
        strands.append(strand[None,...])
    strands = np.concatenate(strands,0)
    strands = torch.from_numpy(strands).type(torch.float32).cuda()
    return strands
    # strands = []
    # signals = signals.cpu().numpy()
    # for s in signals:
    #     strand = inverse_FDT(s)
    #     strands.append(strand[None,...])
    # strands = np.concatenate(strands, 0)
    # strands = torch.from_numpy(strands).type(torch.float32).cuda()
    # return strands


def draw_facepose(image, lmks,eps=1):
    H,W = image.shape[2:]
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    count =0
    for lmk in lmks:
        x, y = int(lmk[0]),int(lmk[1])
        if (x > eps and y > eps) and (x < W and y < H):
            # print(x,y)
            count+=1
            cv2.circle(canvas, (x, y), 3*H//800, (255, 255, 255), thickness=-1)


    out = torch.from_numpy(canvas)/255
    out = out.permute(2,0,1)[None,...].to(image.device)
    out = torch.where(out==0,image,out)
    return out


def to_tensor(x,from_numpy=True,dtype='float32',device='cuda:0'):
    if from_numpy:
        x = torch.from_numpy(x)

    if dtype=='float32':
        x = x.type(torch.float32)
    elif dtype =='long':
        x = x.type(torch.long)
    x = x.to(device)
    return x

def summary(model,file=sys.stderr):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            # print("name is ", name)
            if p is not None:
                # print("parameter with shape ", p.shape)
                # print("parameter has dim ", p.dim())
                if p.dim()==0: #is just a scalar parameter
                    total_params+=1
                else:
                    total_params += reduce(lambda x, y: x * y, p.shape)
                # if(p.grad==None):
                #     print("p has no grad", name)
                # else:
                #     print("p has gradnorm ", name ,p.grad.norm() )

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        if file is sys.stderr:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
            for name, p in model._parameters.items():
                if hasattr(p, 'grad'):
                    if(p.grad==None):
                        print("p has no grad", name)
                        main_str+="p no grad"
                    else:
                        # print("p has gradnorm ", name ,p.grad.norm() )
                        main_str+= "\n" + name + " p has grad norm, min, max" + str(p.grad.norm()) + " " + str(p.grad.min()) + " " + str(p.grad.max())
                        main_str+= "\n" + name + " p has grad type" + str(p.grad.type()) 

                        #check for nans
                        if torch.isnan(p.grad).any():
                            print("NAN detected in grad of ", name)
                            print("main_str is ", main_str)
                            exit(1)

                #show also the parameter itself, an not only the gradient
                if (p is not None):
                    if (p.numel()!=0):
                        main_str+= "\n" + name + " Param, min, max"  + str(p.min()) + " " + str(p.max())

                        #check for nans
                        if torch.isnan(p).any():
                            print("NAN detected in param ", name)
                            print("main_str is ", main_str)
                            exit(1)

        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        print(string, file=file)
    return count

def get_window(win_type: str, win_length: int):
    """Return a window function.

    Args:
        win_type (str): Window type. Can either be one of the window function provided in PyTorch
            ['hann_window', 'bartlett_window', 'blackman_window', 'hamming_window', 'kaiser_window']
            or any of the windows provided by [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.get_window.html).
        win_length (int): Window length

    Returns:
        win: The window as a 1D torch tensor
    """

    try:
        win = getattr(torch, win_type)(win_length)
    except:
        win = torch.from_numpy(scipy.signal.windows.get_window(win_type, win_length))

    return win


# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def strands_from_signal_torch(signal,norm='backward'):
    N = signal.shape[1]//3
    F_A = signal[:,:N]
    F_cos = signal[:,N:2 * N]
    F_sin = signal[:,2 * N:]

    # phase = F_cos * F_A + 1j*F_sin * F_A
    phase = F_cos + 1j * F_sin
    phase = torch.angle(phase)
    reconstructed_fft_result = F_A * torch.exp(1j * phase)
    reconstructed_fft_result = torch.fft.irfft(reconstructed_fft_result,dim=-2,norm=norm)
    return reconstructed_fft_result


def strands_from_signal_torch1(signal,input_type,decode_type,norm='ortho'):

    if input_type=='fft':
        N = signal.shape[1]//2
        F_cos = signal[:,: N]
        F_sin = signal[:,N:2 * N]

        # phase = F_cos * F_A + 1j*F_sin * F_A
        phase = F_cos + 1j * F_sin
        F_A = torch.abs(phase)

        phase = torch.angle(phase)
        reconstructed_fft_result = F_A * torch.exp(1j * phase)
        reconstructed_fft_result = torch.fft.irfft(reconstructed_fft_result,dim=-2,norm=norm)
    elif input_type=='chunked_fft':

        N = signal.shape[1]//3
        reconstructed_fft_result =[]
        for i in range(3):
            sub_signal = signal[:,N*i:(i+1)*N]

            F_cos = sub_signal[:, : N//2]
            F_sin = sub_signal[:, N//2: N]

            phase = F_cos + 1j * F_sin
            F_A = torch.abs(phase)
            phase = torch.angle(phase)
            reconstructed_fft= F_A * torch.exp(1j * phase)
            reconstructed_fft = torch.fft.irfft(reconstructed_fft, dim=-2, norm=norm) #B,nr_points//3,3
            reconstructed_fft_result.append(reconstructed_fft)
        reconstructed_fft_result = torch.cat(reconstructed_fft_result,1)

    if decode_type=='dir':
        reconstructed_fft_result = torch.cumsum(reconstructed_fft_result,dim=1)


    return reconstructed_fft_result

def strands_from_signal_torch2(signal,norm='ortho'):
    N = signal.shape[1]//2
    F_cos = signal[:,: N]
    F_sin = signal[:,N:2 * N]

    # phase = F_cos * F_A + 1j*F_sin * F_A
    phase = F_cos + 1j * F_sin
    F_A = torch.abs(phase)

    phase = torch.angle(phase)
    reconstructed_fft_result = F_A * torch.exp(1j * phase)
    reconstructed_fft_result = torch.fft.irfft(reconstructed_fft_result,dim=-2,norm=norm)
    return reconstructed_fft_result


def compute_crop_size(mask):
    H,W = mask.shape[:2]
    index = np.nonzero(mask)
    left = np.min(index[1])
    right = np.max(index[1])
    top = np.min(index[0])
    bottom = np.max(index[0])
    horizontal = H - (right-left)
    vertical = W - (bottom-top)
    crop_size = max(horizontal,vertical)
    crop_size = random.randint(0,crop_size)
    ratio_left = left/(horizontal)

    ratio_top = top/(vertical)

    crop_left = int(ratio_left*crop_size)
    crop_right = W - crop_size + crop_left
    crop_top = int(ratio_top*crop_size)

    crop_bottom = H -crop_size + crop_top
    return crop_left,crop_right,crop_top,crop_bottom


def dilate_erode_mask(mask):
    mask = cv2.resize(mask, (512, 512))
    rand_dilate_erod = random.random()
    if rand_dilate_erod < 0.5:  # dilate
        kernel = np.ones((3, 3), np.uint8)
        iterations = random.randint(1, 5)
        dilated_image = cv2.dilate(mask, kernel, iterations=iterations)
        # mask = dilated_image
        random_x = random.randint(100, 400)
        if random.random() < 0.5:
            ###only dilate part of mask, avoid the entire mask shape not changing but only getting bigger
            mask[:, random_x:min(random_x + 256, 511)] = dilated_image[:, random_x:min(random_x + 256, 511)]
        else:
            mask[random_x:min(random_x + 256, 511), :] = dilated_image[random_x:min(random_x + 256, 511), :]
    elif rand_dilate_erod < 0.8: #erode
        kernel = np.ones((3, 3), np.uint8)
        iterations = random.randint(1, 2)
        eroded_image = cv2.erode(mask, kernel, iterations=iterations)
        random_x = random.randint(100, 400)
        if random.random() < 0.5:
            mask[:, random_x:min(random_x + 256, 511)] = eroded_image[:, random_x:min(random_x + 256, 511)]
        else:
            mask[random_x:min(random_x + 256, 511), :] = eroded_image[random_x:min(random_x + 256, 511), :]
    return mask


#from pytorch3d
def random_quaternions(
    n: int, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    o = torch.randn((n, 4), dtype=dtype, device=torch.device("cuda"))
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o





def random_rotations(
    n: int, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Generate random rotations as 3x3 rotation matrices.

    Args:
        n: Number of rotation matrices in a batch to return.
        dtype: Type to return.
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type.

    Returns:
        Rotation matrices as tensor of shape (n, 3, 3).
    """
    quaternions = random_quaternions(n, dtype=dtype)
    return quaternion_to_matrix(quaternions)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)



class CustomCrop:
    def __init__(self, crop_size,top_ratio, left_ratio):
        self.top_ratio = top_ratio
        self.left_ratio = left_ratio
        self.crop_size = crop_size

    def __call__(self, tensor):
        _, h, w = tensor.shape

        top_ratio = random.uniform(0.2,self.top_ratio)
        left_ratio = random.uniform(0.3,0.7)

        crop_size = random.randint(self.crop_size,h)

        top = int((h-crop_size) * top_ratio)
        left = int((w-crop_size) * left_ratio)

        tensor_cropped = F.crop(tensor, top, left, self.crop_size, self.crop_size)

        return tensor_cropped

class HorizontalFlip(object):
    def __call__(self, img):
        return F.hflip(img)

def get_transform(size=(512,512),apply_agmentationt=False,normalization=False,flip=False):
    color_jitter = transforms.ColorJitter(
        brightness=(0.8, 1.2),
        contrast=(0.8,1.2),
        saturation=(0.7,1.3),
        hue=0.05
    )
    transform = []
    if apply_agmentationt:
        transform.append(color_jitter)

    transform.append(transforms.ToTensor())

    if flip:
        transform.append(HorizontalFlip())

    if apply_agmentationt:
        transform.append(CustomCrop(crop_size=600, top_ratio=0.4, left_ratio=0.5))

    transform.append(transforms.Resize(size))
    if apply_agmentationt:
        transform.append(transforms.RandomRotation(degrees=30))

    if normalization:
        transform.append( transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ))


    transform = transforms.Compose(transform)
    return transform
