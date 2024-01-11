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
    # print(x.shape,"input")
    # D=x.shape[1]
    # print("D",D)
    # print(num_frequencies,"freq")

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

class model_2d(nn.Module):
    
    """
    Define a 2D model comprising of three fully connected layers,
    two relu activations and one sigmoid activation.
    """
    
    def __init__(self, filter_size=128, num_frequencies=6):
        super().__init__()
        #############################  TODO 1(b) BEGIN  ############################
        self.num_frequencies = num_frequencies
        self.filter_size = filter_size
        input_size=(2*self.num_frequencies*2)+2
        print("input size:",input_size)
        print("filter size:",self.filter_size)
        self.layer1 = nn.Linear(input_size, self.filter_size)  # Initialized with dummy input size, will be updated in forward()
        self.layer2 = nn.Linear(self.filter_size, self.filter_size)
        self.layer3 = nn.Linear(self.filter_size, 3)  # Output size is 3 for RGB color



        #############################  TODO 1(b) END  ##############################        

    def forward(self, x):
        #############################  TODO 1(b) BEGIN  ############################

        print("Dimensions of x:",x.shape)

        # x = positional_encoding(x, num_frequencies=self.num_frequencies)  # Apply positional encoding

        if self.layer1.in_features != x.shape[1]:
            self.layer1 = nn.Linear(x.shape[1], self.filter_size)
        print("x:",x.shape[1])

        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))  


        #############################  TODO 1(b) END  ##############################  
        return x
    
def train_2d_model(test_img, num_frequencies, device, model=model_2d, positional_encoding=positional_encoding, show=True):

    # Optimizer parameters
    lr = 5e-4
    iterations = 10000
    height, width = test_img.shape[:2]

    # Number of iters after which stats are displayed
    display = 2000  
    
    # Define the model and initialize its weights.
    model2d = model(num_frequencies=num_frequencies)
    model2d.to(device)

    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

    model2d.apply(weights_init)

    #############################  TODO 1(c) BEGIN  ############################
    # Define the optimizer
    optimizer = torch.optim.Adam(model2d.parameters(), lr=lr)

    #############################  TODO 1(c) END  ############################

    # Seed RNG, for repeatability
    seed = 5670
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Lists to log metrics etc.
    psnrs = []
    iternums = []

    t = time.time()
    t0 = time.time()

    #############################  TODO 1(c) BEGIN  ############################
    # Create the 2D normalized coordinates, and apply positional encoding to them
    coords = torch.stack(torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width)), dim=-1)
    coords = coords.to(device)
    encoded_coords = positional_encoding(coords.reshape(-1, 2), num_frequencies=num_frequencies)
    






    #############################  TODO 1(c) END  ############################

    for i in range(iterations+1):
        optimizer.zero_grad()
        #############################  TODO 1(c) BEGIN  ############################
        # Run one iteration
        pred = model2d(encoded_coords)
        pred = pred.view(height, width, 3)


        # Compute mean-squared error between the predicted and target images. Backprop!
        loss = torch.mean((pred - test_img) ** 2)
        loss.backward()
        optimizer.step()



        #############################  TODO 1(c) END  ############################

        # Display images/plots/stats
        if i % display == 0 and show:
            #############################  TODO 1(c) BEGIN  ############################
            # Calculate psnr
            mse = loss.item()
            psnr = 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse)))

            #############################  TODO 1(c) END  ############################

            print("Iteration %d " % i, "Loss: %.4f " % loss.item(), "PSNR: %.2f" % psnr.item(), \
                "Time: %.2f secs per iter" % ((time.time() - t) / display), "%.2f secs in total" % (time.time() - t0))
            t = time.time()

            psnrs.append(psnr.item())
            iternums.append(i)

            plt.figure(figsize=(13, 4))
            plt.subplot(131)
            plt.imshow(pred.detach().cpu().numpy())
            plt.title(f"Iteration {i}")
            plt.subplot(132)
            plt.imshow(test_img.cpu().numpy())
            plt.title("Target image")
            plt.subplot(133)
            plt.plot(iternums, psnrs)
            plt.title("PSNR")
            plt.show()

            #if i==iterations:
            #  np.save('result_'+str(num_frequencies)+'.npz',pred.detach().cpu().numpy())

    print('Done!')
    return pred.detach().cpu()

# # Test data
# x = torch.randn(32, 2)  # A batch of 32 random 2D coordinates

# # Test positional_encoding function
# encoded_x = positional_encoding(x)
# print("Positional Encoding Shape: ", encoded_x.shape)  # Expected shape: [32, 26]

# # Test model_2d class
# model = model_2d()
# output = model(x)
# print("Model Output Shape: ", output.shape)  # Expected shape: [32, 3]

# # Test model_2d class with a different num_frequencies value
# model_custom_frequencies = model_2d(num_frequencies=8)
# output_custom_frequencies = model_custom_frequencies(x)
# print("Model Output Shape (custom frequencies): ", output_custom_frequencies.shape)  # Expected shape: [32, 3]

