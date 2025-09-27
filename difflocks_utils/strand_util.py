import struct
import numpy as np
# import splines
import torch
from modules.difflocks.difflocks_utils.general_util import tridiagonal_solve
import trimesh
from scipy.interpolate import splprep, splev
from modules.difflocks.difflocks_utils.general_util import get_window

def load_strand(file, return_strands=False,interp=False,num_interp=100):
    with open(file, mode='rb')as f:
        num_strand = f.read(4)

        (num_strand,) = struct.unpack('I', num_strand)
        point_count = f.read(4)

        (point_count,) = struct.unpack('I', point_count)

        # print("num_strand:",num_strand)
        segments = f.read(2 * num_strand)
        segments = struct.unpack('H' * num_strand, segments)
        segments = list(segments)

        num_points = sum(segments)

        points = f.read(4 * num_points * 3)
        points = struct.unpack('f' * num_points * 3, points)


    f.close()
    points = list(points)

    points = np.array(points)
    points = np.reshape(points, (-1, 3))
    new_p = []
    if return_strands:
        beg = 0
        strands = []
        oris = []
        for seg in segments:
            end = beg + seg
            strand = points[beg:end]
            if interp:
                try:
                    if strand.shape[0]>3:
                        new_points = B_spline_interpolate(strand,num_interp)
                        strand = np.stack(new_points,1)
                        new_p.append(strand)
                    else:
                        beg+=seg
                        continue
                    # new_p.append(strand)
                except:
                    beg+=seg
                    continue


            strands.append(strand)
            dir = np.concatenate([strand[1:] - strand[:-1], strand[-1:] - strand[-2:-1]], 0)
            dir = dir/np.linalg.norm(dir,2,-1,keepdims=True)
            oris.append(dir)
            beg += seg
        if interp:
            points = np.concatenate(new_p,0)
        return segments, points, strands, oris
    else:
        return segments, points


def save_hair_strands(path,strands):
    segments = [strands[i].shape[0] for i in range(len(strands))]
    hair_count=len(segments)
    point_count=sum(segments)
    points = np.concatenate(strands,0)

    with open(path, 'wb')as f:
        f.write(struct.pack('I', hair_count))
        f.write(struct.pack('I', point_count))
        for num_every_strand in segments:
            f.write(struct.pack('H', num_every_strand ))

        for vec in points:
            f.write(struct.pack('f', vec[0]))
            f.write(struct.pack('f', vec[1]))
            f.write(struct.pack('f', vec[2]))
    f.close()


def get_strands(points):
    strands = []
    count = 0
    total = points.shape[0]
    oris = []
    while True:
        strand = []
        base_dist = points[count+1] - points[count]
        base_dist = np.linalg.norm(base_dist,2)
        if base_dist==0:
            print('base:0')
            base_dist=0.0025
        for i in range(256):
            strand.append(points[count:count+1])
            if count== total-1:
                count+=1
                break
            dist = points[count+1] - points[count]
            count += 1
            if np.linalg.norm(dist,2)> base_dist*6:
                break
        strand = np.concatenate(strand,0)
        ori = np.concatenate([strand[1:]-strand[:-1],strand[-1:]-strand[-2:-1]],0)
        strands.append(strand)
        oris.append(ori)
        if count==total:
            break
    oris = np.concatenate(oris,0)
    return strands,oris



def load_strands_from_ply(file,move_strands_to_usc=True):
    lineset = trimesh.load(file)
    points = np.asarray(lineset.vertices)

    if move_strands_to_usc:
        points *= 1.05
        points += np.array([0, 1.437, 0.009])

    if points.shape[0]%256==0:
        strands = points.reshape(-1, 256, 3)
        oris = strands[:, 1:] - strands[:, :-1]
        oris = np.concatenate([oris, strands[:, -1:] - strands[:, -2:-1]], 1)
        check = False
    else:
        strands,oris = get_strands(points)
        check =True

    return points, strands, oris,check



def B_spline_interpolate(X,num):
    tck, u = splprep([X[:, 0], X[:, 1], X[:, 2]], s=0., k=3)
    U = np.linspace(0, 1, num)
    new_points = splev(U, tck)
    return new_points

def get_strand_length(strand):
    delta = strand[:-1] - strand[1:]
    delta_length = np.sqrt(np.sum(delta**2, axis=1, keepdims=False))
    length = np.sum(delta_length, axis=0, keepdims=False)
    return length, delta_length

def spline_strand(strand, num_strand_points=100):
    num_ori_points = strand.shape[0]
    interp_spline = splines.CatmullRom(strand)
    interp_idx = np.arange(num_strand_points) / (num_strand_points / (num_ori_points - 1))
    interp_strand = interp_spline.evaluate(interp_idx)
    assert interp_strand.shape[0] == num_strand_points, "Spline error."

    return interp_strand


def pad_strand(strand, num_strand_points=100):
    num_ori_points = strand.shape[0]
    if num_ori_points > num_strand_points:
        return strand[:num_strand_points]

    num_pad = num_strand_points - num_ori_points
    last_delta = strand[-1] - strand[-2]
    offsets = np.arange(num_pad) + 1
    offsets = offsets[:, None]
    last_delta = last_delta[None, :]
    offsets = offsets * last_delta
    # padded_strand = np.zeros_like(offsets) + strand[-1]
    padded_strand = offsets + strand[-1]
    padded_strand = np.concatenate((strand, padded_strand), axis=0)

    ori_time = np.linspace(0, 1, num_ori_points)
    strd_len, delta_len = get_strand_length(strand)  # modify time by length
    ori_time[1:] = delta_len / strd_len
    ori_time = np.add.accumulate(ori_time)

    padded_time = 1. + (np.arange(num_pad) + 1) * (1. / num_ori_points)
    padded_time = np.concatenate((ori_time, padded_time), axis=0)
    return padded_strand, padded_time


def natural_cubic_spline_coeffs(t, x):
    """
    Calculates the coefficients of the natural cubic spline approximation to the batch of controls given.

    Arguments:
        t: Tensor of times. Must be monotonically increasing.
        x: Tensor of values, of shape (..., length, input_channels), where ... is some number of batch dimensions. This
            is interpreted as a (batch of) paths taking values in an input_channels-dimensional real vector space, with
            length-many observations. Missing values are supported, and should be represented as NaNs.

    In particular, the support for missing values allows for batching together elements that are observed at
    different times; just set them to have missing values at each other's observation times.

    Warning:
        If there are missing values then calling this function can be pretty slow. Make sure to cache the result, and
        don't reinstantiate it on every forward pass, if at all possible.

    Returns:
        A tuple of five tensors, which should in turn be passed to `torchcubicspline.NaturalCubicSpline`.

        Why do we do it like this? Because typically you want to use PyTorch tensors at various interfaces, for example
        when loading a batch from a DataLoader. If we wrapped all of this up into just the
        `torchcubicspline.NaturalCubicSpline` class then that sort of thing wouldn't be possible.

        As such the suggested use is to:
        (a) Load your data.
        (b) Preprocess it with this function.
        (c) Save the result.
        (d) Treat the result as your dataset as far as PyTorch's `torch.utils.data.Dataset` and
            `torch.utils.data.DataLoader` classes are concerned.
        (e) Call NaturalCubicSpline as the first part of your model.
    """

    a, b, two_c, three_d = cubic_spline_coeffs(t, x.transpose(-1, -2))

    # These all have shape (..., length - 1, channels)
    a = a.transpose(-1, -2)
    b = b.transpose(-1, -2)
    # The code so far has created twice the c value and three times the d value because it was written with a preference
    # for computing the derivative of the natural cubic spline, which need those values instead. I'm not going to try
    # and change that for this standalone torchcubicspline project, and instead this is a simple fix.
    c = two_c.transpose(-1, -2) / 2
    d = three_d.transpose(-1, -2) / 3
    return t, a, b, c, d

class NaturalCubicSpline:
    """Calculates the natural cubic spline approximation to the batch of controls given. Also calculates its derivative.

    Example:
        t = torch.linspace(0, 1, 7)
        # (2, 1) are batch dimensions. 7 is the time dimension (of the same length as t). 3 is the channel dimension.
        x = torch.rand(2, 1, 7, 3)
        coeffs = natural_cubic_spline_coeffs(t, x)

        # ...at this point you can save the coeffs, put them through PyTorch's Datasets and DataLoaders, etc...

        spline = NaturalCubicSpline(coeffs)

        point = torch.tensor(0.4)
        # will be a tensor of shape (2, 1, 3), corresponding to batch and channel dimensions
        out = spline.derivative(point)

        point = torch.tensor([0.4, 0.5])
        # will be a tensor of shape (2, 1, 2, 3), corresponding to batch, time and channel dimensions
        out = spline.derivative(point)
    """

    def __init__(self, coeffs, **kwargs):
        """
        Arguments:
            coeffs: As returned by `natural_cubic_spline_coeffs`.
        """
        super(NaturalCubicSpline, self).__init__(**kwargs)

        t, a, b, c, d = coeffs

        self._t = t
        self._a = a
        self._b = b
        self._c = c
        self._d = d

    def evaluate(self, t):
        maxlen = self._b.size(-2) - 1
        inners = torch.zeros((t.shape[0], t.shape[1], 3)).to(t.device)
        for i_b in range(self._t.shape[0]):
            index = torch.bucketize(t.detach()[i_b], self._t[i_b]) - 1
            index = index.clamp(0, maxlen)  # clamp because t may go outside of [t[0], t[-1]]; this is fine
            # will never access the last element of self._t; this is correct behaviour
            fractional_part = t[i_b] - self._t[i_b][index]
            fractional_part = fractional_part.unsqueeze(-1)
            inner = self._c[i_b, index, :] + self._d[i_b, index, :] * fractional_part
            inner = self._b[i_b, index, :] + inner * fractional_part
            inner = self._a[i_b, index, :] + inner * fractional_part
            inners[i_b] = inner
        return inners

    def derivative(self, t, order=1):
        fractional_part, index = self._interpret_t(t)
        fractional_part = fractional_part.unsqueeze(-1)
        if order == 1:
            inner = 2 * self._c[..., index, :] + 3 * self._d[..., index, :] * fractional_part
            deriv = self._b[..., index, :] + inner * fractional_part
        elif order == 2:
            deriv = 2 * self._c[..., index, :] + 6 * self._d[..., index, :] * fractional_part
        else:
            raise ValueError('Derivative is not implemented for orders greater than 2.')
        return deriv


def cubic_spline_coeffs(t, x):
    # x should be a tensor of shape (..., length)
    # Will return the b, two_c, three_d coefficients of the derivative of the cubic spline interpolating the path.

    length = x.size(-1)

    if length < 2:
        # In practice this should always already be caught in __init__.
        raise ValueError("Must have a time dimension of size at least 2.")
    elif length == 2:
        a = x[..., :1]
        b = (x[..., 1:] - x[..., :1]) / (t[..., 1:] - t[..., :1])
        two_c = torch.zeros(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)
        three_d = torch.zeros(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)
    else:
        # Set up some intermediate values
        time_diffs = t[..., 1:] - t[..., :-1]
        time_diffs_reciprocal = time_diffs.reciprocal()
        time_diffs_reciprocal_squared = time_diffs_reciprocal ** 2
        three_path_diffs = 3 * (x[..., 1:] - x[..., :-1])
        six_path_diffs = 2 * three_path_diffs
        path_diffs_scaled = three_path_diffs * time_diffs_reciprocal_squared[:, None, :]

        # Solve a tridiagonal linear system to find the derivatives at the knots
        system_diagonal = torch.empty((x.shape[0], length), dtype=x.dtype, device=x.device)
        system_diagonal[..., :-1] = time_diffs_reciprocal
        system_diagonal[..., -1] = 0
        system_diagonal[..., 1:] += time_diffs_reciprocal
        system_diagonal *= 2
        system_rhs = torch.empty_like(x)
        system_rhs[..., :-1] = path_diffs_scaled
        system_rhs[..., -1] = 0
        system_rhs[..., 1:] += path_diffs_scaled
        knot_derivatives = tridiagonal_solve(system_rhs, time_diffs_reciprocal,
                                             system_diagonal, time_diffs_reciprocal)

        # Do some algebra to find the coefficients of the spline
        time_diffs_reciprocal = time_diffs_reciprocal[:, None, :]
        time_diffs_reciprocal_squared = time_diffs_reciprocal_squared[:, None, :]
        a = x[..., :-1]
        b = knot_derivatives[..., :-1]
        two_c = (six_path_diffs * time_diffs_reciprocal
                 - 4 * knot_derivatives[..., :-1]
                 - 2 * knot_derivatives[..., 1:]) * time_diffs_reciprocal
        three_d = (-six_path_diffs * time_diffs_reciprocal
                   + 3 * (knot_derivatives[..., :-1]
                          + knot_derivatives[..., 1:])) * time_diffs_reciprocal_squared

    return a, b, two_c, three_d




def diff_spline(hair_data_dict, nr_verts_per_strand=256):
    points = hair_data_dict["points"].cuda()
    times = hair_data_dict["times"].cuda()
    # print(points.shape[:])
    # print('p:',points[0])
    # print('t:',times)
    coeffs = natural_cubic_spline_coeffs(times, points)
    spline = NaturalCubicSpline(coeffs)
    time_pts = torch.arange(nr_verts_per_strand).cuda() / (nr_verts_per_strand - 1)
    # print(time_pts)
    time_pts = time_pts.repeat(points.shape[0], 1)

    splined_points = spline.evaluate(time_pts)
    # print('sp:',self.splined_points)
    splined_points = splined_points.detach()
    return splined_points


#assumes the input is (Batch,Time,dim) or (Batch,Time)
#return (B,Freq,T,dim) containing real and imaginary values
def compute_stft(input, fft_size, hop_size, win_length, window_type="hann_window"):
    window = get_window(window_type, win_length).cuda()

    if len(input.shape)==2:
        #we have a (B,T) input so we can just run stft
         x_stft = torch.stft(
            input,
            fft_size,
            hop_size,
            win_length,
            window,
            return_complex=True,
        )
    elif len(input.shape)==3:
        #do stft for each dimension
        dim=input.shape[-1]
        stft_total=[]
        for i in range(dim):
            input_axis = input[:,:,i]
            x_stft = torch.stft(
                input_axis,
                fft_size,
                hop_size,
                win_length,
                window,
                return_complex=True,
            ).unsqueeze(-1)
            stft_total.append(x_stft)
        x_stft = torch.cat(stft_total,-1)
    else:
        return None
    

    return x_stft

#inverse of compute_stft
def compute_istft(input, fft_size, hop_size, win_length, window_type="hann_window", spatial_size=256):
    window = get_window(window_type, win_length).cuda()

    #if the input if 5 dimensional and the last dimension 2, then we view it as complex
    if len(input.shape)==5 and input.shape[-1]==2:
        input=torch.view_as_complex(input)

    if len(input.shape)==3:
        #we have a (B,T) input so we can just run stft
         x= torch.istft(
            input,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            window=window,
            onesided=True,
            return_complex=False,
        )
    elif len(input.shape)==4:
        #do stft for each dimension
        dim=input.shape[-1]
        x_total=[]
        for i in range(dim):
            input_axis = input[:,:,:,i]
            x_axis = torch.istft(
                input_axis,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_length,
                window=window,
                onesided=True,
                return_complex=False,
                length=spatial_size,
            ).unsqueeze(-1)
            x_total.append(x_axis)
        x = torch.cat(x_total,-1)
    else:
        return None
    

    return x


def compute_fft(input):
    if len(input.shape)==2:
        #we have a (B,T) input so we can just run stft
        x_fft = torch.fft.rfft(
            input,
        )
    elif len(input.shape)==3:
        #do stft for each dimension
        dim=input.shape[-1]
        fft_total=[]
        for i in range(dim):
            input_axis = input[:,:,i]
            x_fft = torch.fft.rfft(
                input_axis,
            ).unsqueeze(-1)
            fft_total.append(x_fft)
        x_fft = torch.cat(fft_total,-1)
    else:
        return None
    

    return x_fft

#inverse of compute_stft
def compute_ifft(input, spatial_size=256):

    #if the input if 4 dimensional and the last dimension 2, then we view it as complex
    if len(input.shape)==4 and input.shape[-1]==2:
        input=torch.view_as_complex(input)

    if len(input.shape)==2:
        #we have a (B,T) input so we can just run stft
         x= torch.fft.irfft(
            input,
        )
    elif len(input.shape)==3:
        #do stft for each dimension
        dim=input.shape[-1]
        x_total=[]
        for i in range(dim):
            input_axis = input[:,:,i]
            x_axis = torch.fft.irfft(
                input_axis,
                n=spatial_size,
            ).unsqueeze(-1)
            x_total.append(x_axis)
        x = torch.cat(x_total,-1)
    else:
        return None
    

    return x


#strand positions should be [nr_strands,nr_verts,3]
def compute_dirs(strand_positions, append_last_dir=True):
    nr_verts_per_strand=strand_positions.shape[1]
    cur_points = strand_positions[:, 0:nr_verts_per_strand - 1, :]
    next_points = strand_positions[:,  1:nr_verts_per_strand, :]
    dirs = next_points - cur_points

    #dirs have nr_verts-1 because the last point doesn-t have a direction. To get the same nr_verts we duplicate the last dir 
    if append_last_dir:
        last_dir=dirs[:,-1:,:]
        dirs = torch.cat([dirs, last_dir],1) # make the direction nr_strands, 100, 3

    return dirs

#strand positions should be [nr_strands,nr_verts,3]
def compute_curv(strand_dirs, append_last_curv=True):
    nr_verts_per_strand=strand_dirs.shape[1]
    cur_dirs = strand_dirs[:, 0:nr_verts_per_strand - 1, :]
    next_dirs = strand_dirs[:,  1:nr_verts_per_strand, :]
    curvs = next_dirs - cur_dirs

    #dirs have nr_verts-1 because the last point doesn-t have a direction. To get the same nr_verts we duplicate the last dir 
    if append_last_curv:
        last_curv=curvs[:,-1:,:]
        curvs = torch.cat([curvs, last_curv],1) # make the direction nr_strands, 100, 3

    return curvs




#density map is B,1,H,W
#we randomly samples some indices from it considering the density as probability
#return binary_map of B,1,H,W with some pixels on white where we should sample strands and some on black where there shouldn't be any strand
def sample_from_density_map(density_map, downsample_lvl=None):
    if downsample_lvl is not None:
        density_map = torch.nn.functional.interpolate(density_map, scale_factor=1.0/downsample_lvl)

    rand_map = torch.rand_like(density_map)

    density_clamped= torch.clamp(density_map, 0.0, 1.0)

    binary_map = (rand_map<=density_clamped).float()

    return binary_map


def sample_strands_from_scalp_with_density(scalp_texture, density_map, strand_codec, normalization_dict,
                                           scalp_mesh_data, tbn_space_to_world_func, nr_chunks=300,
                                           upsample_multiplier=3, return_uvs=False):
    if density_map.shape[2] != scalp_texture.shape[2]:
        # downsample the density to be the same size as the scalp texture
        density_map = torch.nn.functional.interpolate(density_map,
                                                      size=[scalp_texture.shape[2], scalp_texture.shape[3]])

    binary_map = sample_from_density_map(density_map)
    strand_mask_binary = binary_map
    strands_indices = torch.nonzero(strand_mask_binary)[:,
                      -2:]  # torch nonzero gives you a Nx4 matrices because the mask has 4 dimensions (NCHW) but we are only interested in the last 2 dimensions HW
    binary_map_size = strand_mask_binary.shape[2]

    root_uv01 = strands_indices.float() / binary_map_size + 0.5 / binary_map_size

    # print("strands_indices", strands_indices)

    # get more root uvs since usually this amount until now is about a third than what we need (we get 30k and we need 100k)
    # move maximum half pixel away
    # cur_nr_strands= root_uv01.shape[0]
    # upsample_multiplier=2
    # pixel_size=1.0/binary_map_size
    # rand_deviation = torch.rand([cur_nr_strands*upsample_multiplier,2], device=scalp_texture.device)
    # #normalize the deviation so all deviations move exactly one pixel away so they actually encode the direction of movement
    # rand_deviation=torch.nn.functional.normalize(rand_deviation, dim=-1)
    # rand_deviation=rand_deviation*pixel_size
    # print("rand_deviation min max", rand_deviation.min(), rand_deviation.max())
    # #concat
    # root_uv01=torch.cat([root_uv01, root_uv01.repeat(upsample_multiplier,1) + rand_deviation ], 0)
    # root_uv01=root_uv01.clamp(1e-5, 0.9999)

    # attempt2, the new strands are not just random jitters of the existing ones but rather random samples
    cur_nr_strands = root_uv01.shape[0]
    new_strands_rand_uv01 = torch.rand([cur_nr_strands * upsample_multiplier, 2], device=scalp_texture.device)
    # print("new_strands_rand_uv01", new_strands_rand_uv01.shape)
    # normalize the deviation so all deviations move exactly one pixel away so they actually encode the direction of movement
    new_strands_rand_uv01 = new_strands_rand_uv01.clamp(1e-5, 0.9999)
    # remove the ones in low density areas
    rand_map = torch.rand([new_strands_rand_uv01.shape[0], 1], device=scalp_texture.device)  # new_strands_new, 1
    density_clamped = torch.clamp(density_map, 0.0, 1.0)
    # sample density for these new uvs
    new_strands_rand_uv11 = new_strands_rand_uv01 * 2 - 1.0
    new_strands_rand_uv11_pt = torch.cat([new_strands_rand_uv11[:, 1:2], new_strands_rand_uv11[:, 0:1]],
                                         1)  # swap xy because torch expects to be HW so yx
    new_strands_density = torch.nn.functional.grid_sample(density_clamped,
                                                          new_strands_rand_uv11_pt.view(1, 1, -1, 2)).cuda()
    new_strands_density = new_strands_density.permute(0, 2, 3, 1).view(-1, 1)  # Make N,1
    is_new_strand_valid = rand_map < new_strands_density
    # print("is_new_strand_valid", is_new_strand_valid.shape)
    # new_strands_rand_uv01=new_strands_rand_uv01[is_new_strand_valid.repeat(1,2)]
    new_strands_rand_uv01 = torch.masked_select(new_strands_rand_uv01, is_new_strand_valid).view(-1, 2)
    # print("new_strands_rand_uv01 after filtering", new_strands_rand_uv01.shape)
    # combine with previous uvs
    root_uv01 = torch.cat([root_uv01, new_strands_rand_uv01], 0)

    # sample latents
    root_uv11 = root_uv01 * 2 - 1.0
    # swap xy because torch expects to be HW so yx
    # print("root_uv11", root_uv11.shape)
    root_uv11_pt = torch.cat([root_uv11[:, 1:2], root_uv11[:, 0:1]], 1)
    # sample
    selected_latents = torch.nn.functional.grid_sample(scalp_texture,
                                                       root_uv11_pt.view(1, 1, -1, 2),
                                                       mode='nearest')  # selected latent is bchw
    selected_latents = selected_latents.permute(0, 2, 3, 1).view(-1, scalp_texture.shape[1])  # Make N,64

    # selected_latents = scalp_texture[:,:, strands_indices[:,0], strands_indices[:,1]]
    # selected_latents = selected_latents.squeeze(0).transpose(0,1)

    # print("selected_latents", selected_latents.shape)

    # decode
    selected_latents_chunked = torch.chunk(selected_latents, nr_chunks)
    pred_points_list = []
    for selected_latents_cur in selected_latents_chunked:
        pred_dict = strand_codec.decoder(selected_latents_cur, None, normalization_dict)
        pred_points = pred_dict["strand_positions"]
        pred_points_list.append(pred_points)
    strand_points_tbn = torch.cat(pred_points_list, 0)

    # print("root_uv", root_uv01)
    # print("root_uv", root_uv01.shape)
    # print("root_uv min max", root_uv01.min(), root_uv01.max())
    strand_points_world = tbn_space_to_world_func(root_uv01.cpu(), strand_points_tbn, scalp_mesh_data)
    
    if return_uvs:
        return strand_points_world, strand_points_tbn, root_uv01
    return strand_points_world, strand_points_tbn, None

