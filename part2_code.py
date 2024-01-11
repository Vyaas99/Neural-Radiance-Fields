import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import time

def positional_encoding(x, num_frequencies=6, incl_input=True):
    
    """
    Apply positional encoding to the input.
    
    Args:
    x (torch.Tensor): Input tensor to be positionally encoded. 
      The dimension of x is [N, D], where N is the number of input coordinates,
      and D is the dimension of the input coordinate.
    num_frequencies (optional, int): The number of frequencies used in
     the positional encoding (default: 6).
    incl_input (optional, bool): If True, concatenate the input with the 
        computed positional encoding (default: True).

    Returns:
    (torch.Tensor): Positional encoding of the input tensor. 
    """
    
    results = []
    if incl_input:
        results.append(x)
    #############################  TODO 1(a) BEGIN  ############################
    # encode input tensor and append the encoded tensor to the list of results.
    # print("input to pos",x)

    # for i in range(num_frequencies):
    #     freq = 2 ** i
    #     results.append(torch.sin(freq * torch.pi * x))
    #     results.append(torch.cos(freq * torch.pi * x))
    # print(torch.cat(results, dim=-1).shape,"pos")

    N, D = x.shape
    exponents = torch.arange(num_frequencies).view(-1, 1, 1)  # shape: (num_frequencies, 1, 1)
    frequencies = 2 ** exponents  # shape: (num_frequencies, 1, 1)

    x_expanded = x.unsqueeze(0)  # reshape x, shape: (1, N, D)

    # Broadcasting the multiplication, result shape: (num_frequencies, N, D)
    sin_input = (frequencies * torch.pi * x_expanded)
    cos_input = (frequencies * torch.pi * x_expanded)

    # Calculate sin and cos, result shape: (num_frequencies, N, D)
    sin_results = torch.sin(sin_input)
    cos_results = torch.cos(cos_input)

    # Concatenate sin and cos results along the last dimension, result shape: (num_frequencies, N, 2 * D)
    results = torch.cat((sin_results, cos_results), dim=-1)

    # Reshape the tensor, result shape: (N, num_frequencies * 2 * D)
    results = results.permute(1, 0, 2).reshape(N, -1)

    if incl_input:
        results = torch.cat((x, results), dim=-1)

    return results
    



    #############################  TODO 1(a) END  ##############################
    return torch.cat(results, dim=-1)

def get_rays(height, width, intrinsics, Rcw, Tcw):
    
    """
    Compute the origin and direction of rays passing through all pixels of an image (one ray per pixel).

    Args:
    height: the height of an image.
    width: the width of an image.
    intrinsics: camera intrinsics matrix of shape (3, 3).
    Rcw: Rotation matrix of shape (3,3) from  world to camera coordinates.
    Tcw: Translation vector of shape (3,1) that transforms from world to camera coordinates

    Returns:
    ray_origins (torch.Tensor): A tensor of shape (height, width, 3) denoting the centers of
      each ray. Note that desipte that all ray share the same origin, here we ask you to return 
      the ray origin for each ray as (height, width, 3).
    ray_directions (torch.Tensor): A tensor of shape (height, width, 3) denoting the
      direction of each ray.
    """

    device = intrinsics.device
    ray_directions = torch.zeros((height, width, 3), device=device)  # placeholder
    ray_origins = torch.zeros((height, width, 3), device=device)  # placeholder
    
    #############################  TODO 2.1 BEGIN  ##########################  

    inv_intrinsics = np.linalg.inv(intrinsics)
    for u in range(width):
      for v in range(height):
        ray_origins[v][u] = Tcw
        calibrated_coords = inv_intrinsics @ np.array([u, v, 1])
        direction = Rcw @ calibrated_coords
        ray_directions[v][u] = direction




    #############################  TODO 2.1 END  ############################
    return ray_origins, ray_directions

def stratified_sampling(ray_origins, ray_directions, near, far, samples):

    """
    Sample 3D points on the given rays. The near and far variables indicate the bounds of sampling range.

    Args:
    ray_origins: Origin of each ray in the "bundle" as returned by the
      get_rays() function. Shape: (height, width, 3).
    ray_directions: Direction of each ray in the "bundle" as returned by the
      get_rays() function. Shape: (height, width, 3).
    near: The 'near' extent of the bounding volume.
    far:  The 'far' extent of the bounding volume.
    samples: Number of samples to be drawn along each ray.
  
    Returns:
    ray_points: Query 3D points along each ray. Shape: (height, width, samples, 3).
    depth_points: Sampled depth values along each ray. Shape: (height, width, samples).
    """

    #############################  TODO 2.2 BEGIN  ############################

    device = ray_origins.device

    # Normalize ray directions
    ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)

    # Compute the depth points (ti) for each ray
    depth_points = torch.linspace(near, far, samples, device=device)
    depth_points = depth_points.view(1, 1, samples).expand(ray_origins.shape[0], ray_origins.shape[1], samples)

    # Compute the 3D points along each ray
    ray_points = ray_origins[..., None, :] + depth_points[..., None] * ray_directions[..., None, :]
    #############################  TODO 2.2 END  ############################
    return ray_points, depth_points
    
class nerf_model(nn.Module):
    
    """
    Define a NeRF model comprising eight fully connected layers and following the
    architecture described in the NeRF paper. 
    """

    def __init__(self, filter_size=256, num_x_frequencies=6, num_d_frequencies=3):
        super().__init__()

        #############################  TODO 2.3 BEGIN  ############################

        self.num_x_frequencies = num_x_frequencies
        self.num_d_frequencies = num_d_frequencies
        self.filter_size = filter_size


        in_1=(2*self.num_x_frequencies*3)+3
        self.layer_1=nn.Linear(in_1,self.filter_size)
        self.layer_2=nn.Linear(self.filter_size,self.filter_size)
        self.layer_3=nn.Linear(self.filter_size,self.filter_size)
        self.layer_4=nn.Linear(self.filter_size,self.filter_size)
        self.layer_5=nn.Linear(self.filter_size,self.filter_size)
        in_2=self.filter_size+((2*self.num_x_frequencies*3)+3)
        self.layer_6=nn.Linear(in_2,self.filter_size)
        self.layer_7=nn.Linear(self.filter_size,self.filter_size)
        self.layer_8=nn.Linear(self.filter_size,self.filter_size)
        self.sigma_value=nn.Linear(self.filter_size,1)
        self.layer_9=nn.Linear(self.filter_size,self.filter_size)
        in_3=self.filter_size+((2*self.num_d_frequencies*3)+3)
        self.layer_10=nn.Linear(in_3,128)
        self.layer_11=nn.Linear(128,3)


        

        

        #############################  TODO 2.3 END  ############################


    def forward(self, x, d):
        #############################  TODO 2.3 BEGIN  ############################


        out=torch.relu(self.layer_1(x))
        out=torch.relu(self.layer_2(out))
        out=torch.relu(self.layer_3(out))
        out=torch.relu(self.layer_4(out))
        out=torch.relu(self.layer_5(out))
        # print("5 layers done")
        # in_2=self.filter_size+((2*self.num_x_frequencies*3)+3)
        # print("in_2",in_2,x.shape[1])
        # if in_2!=self.filter_size+x.shape[1]:
        #     self.layer_6=nn.Linear(self.filter_size+x.shape[1],self.filter_size)
        out=torch.cat((out,x),dim=1)
        out=torch.relu(self.layer_6(out))
        # print("6 layers done")
        out=torch.relu(self.layer_7(out))
        # print("7 layers done")
        out=torch.relu(self.layer_8(out))
        # print("8 layers done")
        sigma=self.sigma_value(out)
        out=self.layer_9(out)
        # print("9 layers done")
        # in_3=self.filter_size+((2*self.num_d_frequencies*3)+3)
        # print("in_3",in_3,d.shape[1])
        # if in_3!=self.filter_size+d.shape[1]:
        #     self.layer_10=nn.Linear(self.filter_size+d.shape[1],128)
        out=torch.cat((out,d),dim=1)
        out=torch.relu(self.layer_10(out))
        # print("10 layers done")
        rgb=torch.sigmoid(self.layer_11(out))
        # print("11 layers done")

        
        





        #############################  TODO 2.3 END  ############################
        return rgb, sigma
   


def get_batches(ray_points, ray_directions, num_x_frequencies, num_d_frequencies):
    
    def get_chunks(inputs, chunksize = 2**15):
        return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]
    
    """
    This function returns chunks of the ray points and directions to avoid memory errors with the
    neural network. It also applies positional encoding to the input points and directions before 
    dividing them into chunks, as well as normalizing and populating the directions.
    """
    #############################  TODO 2.3 BEGIN  ############################

    
    print("Entered get batches")

    ray_directions_norm = torch.linalg.norm(ray_directions, dim=-1, keepdim=True)
    print("get_batches:")
    print("ray_directions.shape",ray_directions.shape)
    ray_directions_normed = ray_directions / ray_directions_norm


    ray_directions_normed = ray_directions_normed.view(ray_points.shape[0], ray_points.shape[1], 1, 3)



    print("ray_directions_normed.shape",ray_directions_normed.shape)
    ray_directions_populated = ray_directions_normed.repeat(1, 1, ray_points.shape[2], 1)





    # print("ray_directions_populated.shape",ray_directions_populated.shape)
    flattened_directions = ray_directions_populated.reshape(-1,3)
    # print("flattened_directions.shape",flattened_directions.shape)
    encoded_directions = positional_encoding(flattened_directions, num_frequencies=num_d_frequencies)
    # print("num_d_frequencies",num_d_frequencies)
    # print("encoded_directions.shape",encoded_directions.shape)
    ray_directions_batches = get_chunks(encoded_directions)
    flattened_points = ray_points.reshape(-1,3)
    encoded_points = positional_encoding(flattened_points, num_frequencies=num_x_frequencies)
    # print("num_x_frequencies",num_x_frequencies)
    # print("encoded_points.shape",encoded_points.shape)
    ray_points_batches = get_chunks(encoded_points)
    # print("len p",len(ray_points_batches))
    # print("len d",len(ray_directions_batches))
    #############################  TODO 2.3 END  ############################

    return ray_points_batches, ray_directions_batches

def volumetric_rendering(rgb, s, depth_points):

    """
    Differentiably renders a radiance field, given the origin of each ray in the
    "bundle", and the sampled depth values along them.

    Args:
    rgb: RGB color at each query location (X, Y, Z). Shape: (height, width, samples, 3).
    sigma: Volume density at each query location (X, Y, Z). Shape: (height, width, samples).
    depth_points: Sampled depth values along each ray. Shape: (height, width, samples).
  
    Returns:
    rec_image: The reconstructed image after applying the volumetric rendering to every pixel.
    Shape: (height, width, 3)
    """
    
    #############################  TODO 2.4 BEGIN  ############################

    # Pass sigma through ReLU activation
    s = torch.relu(s)
    
    # Calculate the distance between adjacent sampled depth values
    deltas = depth_points[..., 1:] - depth_points[..., :-1]
    deltas = torch.cat([deltas, torch.ones_like(deltas[..., :1]) * 1e9], axis=-1)
    
    # Compute Ti values using cumprod()
    transmittance = torch.exp(-torch.cumsum(s * deltas, axis=-1))
    transmittance = torch.cat([torch.ones_like(transmittance[..., :1]), transmittance[..., :-1]], axis=-1)
    
    # Compute the compositing weights
    weights = transmittance * (1 - torch.exp(-s * deltas))
    
    # Compute the final pixel color
    rec_image = torch.sum(weights[..., None] * rgb, axis=-2)
    #############################  TODO 2.4 END  ############################

    return rec_image

def one_forward_pass(height, width, intrinsics, pose, near, far, samples, model, num_x_frequencies, num_d_frequencies):
    #############################  TODO 2.5 BEGIN  ############################
    
    # print("Entered one forward pass")

    # Compute all the rays from the image
    ray_origins, ray_directions = get_rays(height, width, intrinsics, pose[:3,:3], pose[:3,3])
    # print("get_rays done")

    # Sample the points from the rays
    ray_points, depth_points = stratified_sampling(ray_origins, ray_directions, near, far, samples)
    # print("stratified done")

    # Divide data into batches to avoid memory errors
    ray_points_batches,ray_directions_batches = get_batches(ray_points,ray_directions, num_x_frequencies, num_d_frequencies)
    # print("get_batches done")

    # Forward pass the batches and concatenate the outputs at the end
    model = model.to(ray_points_batches[0].device)
    # print("model saved to device")
    outputs = []
    # print("len ray points",len(ray_points_batches))
    # print("len ray directions",len(ray_directions_batches))
    for i in range(len(ray_points_batches)):
        # print("i",i)
        # print("ray point batches[i]",ray_points_batches[i].shape)
        x=ray_points_batches[i]
        d = ray_directions_batches[i]
        output = model(x,d)
        outputs.append(output)

    rgb = torch.cat([o[0] for o in outputs], dim=0).view(height,width,samples,3)
    sigma = torch.cat([o[1] for o in outputs], dim=0).view(height,width,samples)
    # print("rgb",rgb.shape)
    # print("sigma vol",sigma.shape)

    # Apply volumetric rendering to obtain the reconstructed image
    rec_image = volumetric_rendering(rgb, sigma, depth_points)

    #############################  TODO 2.5 END  ############################
    return rec_image
