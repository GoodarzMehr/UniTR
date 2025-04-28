import torch

import numpy as np

class LoadSimBEVPointsFromFile:
    '''
    Load lidar point cloud from NumPy file.

    Args:
        - trim_step: channel step size for trimming the point cloud.
    '''

    def __init__(self, trim_step):
        self.trim_step = trim_step
    
    def __call__(self, results):
        '''
        Load point cloud data from file.

        Args:
            results: dictionary containing path to point cloud file.
        
        Returns:
            results: dictionary containing point cloud data.
        '''
        lidar_path = results['lidar_path']
        
        points = load_points(lidar_path)

        if self.trim_step > 1:
            points = trim_points(points, self.trim_step)
        
        results['points'] = torch.from_numpy(points).to(torch.float32)

        return results


class LoadSimBEVPointsFromMultiSweeps:
    '''
    Load and aggregate multiple point clouds.

    Args:
        - num_sweeps: desired number of point clouds.
        - trim_step: channel step size for trimming the point cloud.
        - time_step: time step between successive point clouds, i.e.
            simulation time step.
        - is_train: whether the data is for training.
    '''

    def __init__(self, num_sweeps, trim_step, time_step, is_train=True):
        self.num_sweeps = num_sweeps
        self.trim_step = trim_step
        self.time_step = time_step
        self.is_train = is_train

    def __call__(self, results):
        '''
        Load and aggregate multiple point clouds.

        Args:
            results: dictionary containing path to point cloud files.
        
        Returns:
            results: dictionary containing point cloud data.
        '''
        points = results['points']

        # Add the time dimension to the principal point cloud data and set it
        # to zero.
        points = torch.cat((points, torch.zeros((points.shape[0], 1))), dim=1)

        ego2global = results['ego2global']
        lidar2ego = results['lidar2ego']

        total_num_sweeps = len(results['sweeps_lidar_paths'])
        
        # If, during training, the desired number of point clouds is less than
        # the total number available, randomly sample the desired number of
        # point clouds from those available. Otherwise, and during testing,
        # choose the previous point clouds in order.
        if self.num_sweeps >= total_num_sweeps:
            choices = np.arange(total_num_sweeps)
        elif not self.is_train:
            choices = np.arange(self.num_sweeps)
        else:
            choices = np.random.choice(total_num_sweeps, self.num_sweeps, replace=False)

        sweep_points_list = [points]

        for i in choices:
            lidar_path = results['sweeps_lidar_paths'][i]
            
            sweep_points = load_points(lidar_path)

            if self.trim_step > 1:
                sweep_points = trim_points(sweep_points, self.trim_step)

            # Transform point cloud to the coordinate system of the principal
            # point cloud.
            sweep_ego2global = results['sweeps_ego2global'][i]

            lidar2lidar = np.linalg.inv(ego2global @ lidar2ego) @ sweep_ego2global @ lidar2ego

            sweep_points = torch.from_numpy(
                (lidar2lidar @ np.append(sweep_points, np.ones((sweep_points.shape[0], 1)), 1).T)[:3].T
            ).to(torch.float32)

            sweep_points = torch.cat(
                (sweep_points, torch.full((sweep_points.shape[0], 1), self.time_step * (i + 1))),
                dim=1
            )
            
            sweep_points_list.append(sweep_points)
        
        points = torch.cat(sweep_points_list, dim=0).numpy()

        results['points'] = points

        return results


def load_points(lidar_path):
    '''
    Load point cloud data from file.

    Args:
        lidar_path: path to the point cloud file.

    Returns:
        points: array of point cloud data.
    '''
    if lidar_path.endswith('.npz'):
        points = np.load(lidar_path)['data']
    else:
        points = np.load(lidar_path)

    return points

def trim_points(points, trim_step):
    '''
    Trim point cloud data based on the provided trim step.

    Args:
        points: array of point cloud data.
        trim_step: channel step size for trimming the point cloud.

    Returns:
        points: trimmed array of point cloud data.
    '''
    # Calculate beam angles.
    angles = np.arctan(points[:, 2] / np.linalg.norm(points[:, :2], axis=1))
    angles = np.trunc(angles * 1000.0) / 1000.0

    unique_angles = np.sort(np.unique(angles))

    # Some beams may have duplicate corresponding angles due to truncation,
    # e.g. 0.186 and 0.187. For each set, take one angle as the representative
    # and replace all other duplicates with that one.
    channels = []
    extras = []

    for angle in unique_angles:
        if len(channels) == 0:
            channels.append(angle)
        elif abs(np.array(channels) - angle).min() < 0.0015:
            extras.append(angle)
        else:
            channels.append(angle)
    
    for extra in extras:
        angles[angles == extra] = channels[np.abs(np.array(channels) - extra).argmin()]
    
    # Trim the point cloud based on the provided trim step.
    lidar_angles = np.sort(np.array(channels))[::trim_step]

    mask = np.isin(angles, lidar_angles)

    trimmed_points = points[mask]

    return trimmed_points