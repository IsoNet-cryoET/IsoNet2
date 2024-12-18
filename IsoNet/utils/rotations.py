'''
rotation_list = {

0: [(((0,1),1),((1,2),1)), (((0,2),1),((1,2),1))],
1: [(((1,0),1),((1,2),1)), (((2,0),1),((1,2),1))],
2: [(((0,1),1),((1,2),3)), (((0,2),1),((1,2),3))],
3: [(((1,0),1),((1,2),3)), (((2,0),1),((1,2),3))],
4: [(((0,1),1),((1,2),0)), (((0,2),1),((1,2),0))],
5: [(((1,0),1),((1,2),0)), (((2,0),1),((1,2),0))],
6: [(((0,1),1),((1,2),2)), (((0,2),1),((1,2),2))],
7: [(((1,0),1),((1,2),2)), (((2,0),1),((1,2),2))],
}



rotation_list = {

0: [(((0,1),1),((1,2),0)), (((0,1),1),((1,2),1)), (((0,2),1),((1,2),0)), (((0,2),1),((1,2),1))],
1: [(((0,1),1),((1,2),2)), (((0,1),1),((1,2),3)), (((0,2),1),((1,2),2)), (((0,2),1),((1,2),3))],
2: [(((0,1),3),((1,2),0)), (((0,1),3),((1,2),1)), (((0,2),3),((1,2),0)), (((0,2),1),((1,2),1))],
3: [(((0,1),3),((1,2),2)), (((0,1),3),((1,2),3)), (((0,2),3),((1,2),2)), (((0,2),1),((1,2),3))],
}

'''
# this is used in mwr_cli 26 Dec
# rotation_list = [(((0,1),1),((1,2),0)), (((0,1),1),((1,2),1)), (((1,2),1),((0,2),0)), (((0,2),1),((1,2),1)), 
#                 (((0,1),1),((1,2),2)), (((0,1),1),((1,2),3)), (((1,2),1),((0,2),2)), (((0,2),1),((1,2),3)), 
#                 (((0,1),3),((1,2),0)), (((0,1),3),((1,2),1)), (((1,2),3),((0,2),0)), (((0,2),1),((1,2),1)), 
#                 (((0,1),3),((1,2),2)), (((0,1),3),((1,2),3)), (((1,2),3),((0,2),2)), (((0,2),1),((1,2),3))]
# this is from old dgx(workhorse branch) version
# rotation_list = [(((0,1),1),((1,2),0)), (((0,1),1),((1,2),1)), (((0,2),1),((1,2),0)), (((0,2),1),((1,2),1)), 
                # (((0,1),1),((1,2),2)), (((0,1),1),((1,2),3)), (((0,2),1),((1,2),2)), (((0,2),1),((1,2),3)), 
                # (((0,1),3),((1,2),0)), (((0,1),3),((1,2),1)), (((0,2),3),((1,2),0)), (((0,2),1),((1,2),1)), 
                # (((0,1),3),((1,2),2)), (((0,1),3),((1,2),3)), (((0,2),3),((1,2),2)), (((0,2),1),((1,2),3))]

#All 20 rotation
# rotation_list = [(((0,1),1),((1,2),0)), (((0,1),1),((1,2),1)), (((0,2),1),((1,2),0)), (((0,2),1),((1,2),1)), 
#                 (((0,1),1),((1,2),2)), (((0,1),1),((1,2),3)), (((0,2),1),((1,2),2)), (((0,2),1),((1,2),3)), 
#                 (((0,1),3),((1,2),0)), (((0,1),3),((1,2),1)), (((0,2),3),((1,2),0)), (((0,2),1),((1,2),1)), 
#                 (((0,1),3),((1,2),2)), (((0,1),3),((1,2),3)), (((0,2),3),((1,2),2)), (((0,2),1),((1,2),3)),
#                 (((1,2),1),((0,2),0)), (((1,2),1),((0,2),2)), (((1,2),3),((0,2),0)), (((1,2),3),((0,2),2))]

# rotation_list = [(((0,1),1),((1,2),0)), (((0,1),1),((1,2),1)), (((0,2),1),((1,2),0)), (((0,2),1),((1,2),1)), 
#                 (((0,1),1),((1,2),2)), (((0,1),1),((1,2),3)), (((0,2),1),((1,2),2)), (((0,2),1),((1,2),3)), 
#                 (((0,1),3),((1,2),0)), (((0,1),3),((1,2),1)), (((0,2),3),((1,2),0)), (((0,2),3),((1,2),1)), 
#                 (((0,1),3),((1,2),2)), (((0,1),3),((1,2),3)), (((0,2),3),((1,2),2)), (((0,2),3),((1,2),3)),
#                 (((1,2),1),((0,2),0)), (((1,2),1),((0,2),2)), (((1,2),3),((0,2),0)), (((1,2),3),((0,2),2))]

rotation_list_24 = [(((0,1),1),((0,2),0)), (((0,1),1),((0,2),1)), (((0,1),1),((0,2),2)), (((0,1),1),((0,2),3)), 
                    (((0,1),3),((0,2),0)), (((0,1),3),((0,2),1)), (((0,1),3),((0,2),2)), (((0,1),3),((0,2),3)), 
                    (((1,2),1),((0,2),0)), (((1,2),1),((0,2),1)), (((1,2),1),((0,2),2)), (((1,2),1),((0,2),3)), 
                    (((1,2),3),((0,2),0)), (((1,2),3),((0,2),1)), (((1,2),3),((0,2),2)), (((1,2),3),((0,2),3)), 
                    (((0,1),0),((0,2),0)), (((0,1),0),((0,2),1)), (((0,1),0),((0,2),2)), (((0,1),0),((0,2),3)), 
                    (((0,1),2),((0,2),0)), (((0,1),2),((0,2),1)), (((0,1),2),((0,2),2)), (((0,1),2),((0,2),3))]

rotation_list = [(((0,1),1),((0,2),0)), (((0,1),1),((0,2),1)), (((0,1),1),((0,2),2)), (((0,1),1),((0,2),3)), 
                    (((0,1),3),((0,2),0)), (((0,1),3),((0,2),1)), (((0,1),3),((0,2),2)), (((0,1),3),((0,2),3)), 
                    (((1,2),1),((0,2),0)), (((1,2),1),((0,2),1)), (((1,2),1),((0,2),2)), (((1,2),1),((0,2),3)), 
                    (((1,2),3),((0,2),0)), (((1,2),3),((0,2),1)), (((1,2),3),((0,2),2)), (((1,2),3),((0,2),3)), 
                    (((0,1),0),((0,2),1)), (((0,1),0),((0,2),3)), 
                    (((0,1),2),((0,2),1)), (((0,1),2),((0,2),3))]


#rotation_list = [(((0,1),0),((0,2),0)), (((0,1),1),((0,1),0)), (((0,1),1),((0,1),1)),
#                 (((0,2),0),((0,2),0)), (((0,2),1),((0,1),0)), (((0,2),1),((0,1),1))]

#((0,1),0),((1,2),0)), (((0,1),0),((1,2),1)),
#((0,1),0),((1,2),2)), (((0,1),0),((1,2),3)),
#((0,1),2),((1,2),0)), (((0,1),2),((1,2),1)),
#((0,1),2),((1,2),2)), (((0,1),2),((1,2),3)),

import torch
import torch.nn.functional as F

def sample_rot_axis_and_angle():
    rotvec = torch.randn(3, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  # Random vector
    
    rot_axis = rotvec / rotvec.norm()
    
    rot_angle = torch.rand(1,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) * 2 * torch.pi #rotvec.norm()  # This gives the angle in radians
    
    #rot_angle_deg = torch.rad2deg(rot_angle)
    
    return [rot_axis, rot_angle]


def rotation_matrix(axis, angle):
    axis = axis / axis.norm()  # Normalize the axis
    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)
    ux, uy, uz = axis

    # Rotation matrix using Rodrigues' rotation formula
    R = torch.tensor([
        [cos_theta + ux**2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta, ux * uz * (1 - cos_theta) + uy * sin_theta],
        [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy**2 * (1 - cos_theta), uy * uz * (1 - cos_theta) - ux * sin_theta],
        [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta, cos_theta + uz**2 * (1 - cos_theta)]
    ], dtype=torch.float32)

    return R

# Function to rotate the volume using affine transformation
def rotate_vol_around_axis_torch(volume, rot):
    axis = rot[0]
    angle = rot[1]
    # Ensure volume is on the correct device (either 'cpu' or 'cuda')
    device = volume.device
    
    batch_size, _, Z, Y, X = volume.shape
    # Compute the center of the volume
    center = torch.tensor([Z / 2, Y / 2, X / 2], dtype=torch.float32, device=device)

    # Get the 3x3 rotation matrix
    R = rotation_matrix(axis, angle).to(device)

    # Create the 4x4 affine matrix (for 3D transformation)
    affine_matrix = torch.eye(4, dtype=torch.float32, device=device)
    affine_matrix[:3, :3] = R

    # Apply translation to center the volume around (0, 0, 0)
    affine_matrix[:3, 3] = -center

    # Apply translation back after rotation to move the volume back to its original space
    affine_matrix[:3, 3] = affine_matrix[:3, 3] + center

    # Convert the affine matrix to the correct format for affine_grid
    affine_matrix = affine_matrix.unsqueeze(0).repeat(batch_size, 1, 1)

    # Use affine_grid to generate the grid for sampling
    grid = F.affine_grid(affine_matrix[:, :3, :], volume.size(), align_corners=True)

    # Ensure grid is also on the same device as the volume
    grid = grid.to(device)

    # Use grid_sample to apply the rotation
    rotated_volume = F.grid_sample(volume, grid, mode='bilinear', padding_mode='reflection', align_corners=True)

    return rotated_volume