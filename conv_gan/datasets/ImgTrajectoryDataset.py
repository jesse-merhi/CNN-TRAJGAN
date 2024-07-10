import io
import logging
import math
from statistics import mean

import PIL.Image
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from hausdorff import hausdorff_distance as fast_hausdorff_distance
from scipy.stats import wasserstein_distance
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from conv_gan.metrics import time_reversal_ratio, calculate_total_distance

IMG_SIZE = 12
UP_SAMPLE_SIZE = 24
CHANNELS = 4

log = logging.getLogger()

class ImgTrajectoryDataset(Dataset):
    def __init__(self, data, device, min_lat,max_lat,min_lon,max_lon):

        self.device = device
        self.traj = 0
        self.masks = []
        self.generated_display = 0
        self.actual_display = 0
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.len = len(data)
        self.trajectories = []
        log.info("Loading Trajectories...")
        for traj in data:
            image = self.get_singular_traj(traj)
            self.trajectories.append(image)
            mask = self.create_mask(len(traj), IMG_SIZE)
            self.masks.append(mask)
        log.info("Into Tensors...")
        self.trajectories = torch.Tensor(np.array(self.trajectories)).to(self.device)
        self.masks = torch.Tensor(np.array(self.masks)).to(self.device)
        log.info("Trajectories Loaded")


    def create_mask(self,length,max_length):
        mask1 = np.ones((length))
        mask2 = np.zeros((max_length*max_length-length))
        mask = np.concatenate((mask1,mask2)).reshape((IMG_SIZE,IMG_SIZE))
        m = nn.Upsample(size=(UP_SAMPLE_SIZE,UP_SAMPLE_SIZE), mode="nearest")
        array = m(torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float())
        array = array.squeeze(0).squeeze(0)
        array = array.repeat(CHANNELS,1,1)
        return array.cpu().numpy()


    def gen_plot(self, visualised, eval=False,label="Actual",lr=0.005, traces=True):
        """Create a pyplot plot and save to buffer."""
        if eval:
            fig =go.Figure(
                go.Densitymapbox(
                    name=f"{label}-{lr} Trajectory",
                    lon=visualised[1][:500000],
                    lat=visualised[0][:500000],
                    radius=5,
                ))
            
        else:
            fig = go.Figure(
                go.Scattermapbox(
                    name="Real Trajectory",
                    mode=("markers+lines" if traces  else "markers"),
                    lon=visualised[1],
                    lat=visualised[0],
                    marker={"size": 10},
                )
            )
        fig.update_layout(
            mapbox_style="white-bg",
            mapbox_zoom=11,
            mapbox_center_lat=mean(np.array(visualised[0])),
            mapbox_center_lon=mean(np.array(visualised[1])),
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        if eval:
            fig.show()
            return
        img = io.BytesIO()
        try:
            fig_jpg = fig.to_image(format="jpeg")  # kaleido library
            img.write(fig_jpg)
            img.seek(0)
            return img
        except Exception as e:
            log.error(f"Failed to make figure:{e}")
            return 0

    def remove_padding(self,array):
        sum_array = np.sum(array.detach().cpu().numpy(), axis=0)
        max_lat_pad = np.sum(sum_array[:,:] == 0, axis=1)
        padding = np.sum(max_lat_pad[:] == len(array[0]))
        if padding == 0:
            return array
        return array[:,:-padding,:-padding]

    def log_eval(self, imgs, label, masks):
        all_points_list = []
        distances = np.empty(len(masks), dtype=float)
        time_reversal = []
        trajectories = []
        for i in range(len(masks)):
            split = imgs[i]
            split = torch.transpose(split, 0, 2)
            visualised = np.transpose(self.revert_img(split, 0))
            if len(visualised[0]) != 144:
                log.warning(label, len(visualised[0]))
            total_distance = calculate_total_distance(visualised, masks[i])
            distances[i] = total_distance
            all_points_list.append(visualised)
            trajectories.append(visualised)
            time_reversal.append(time_reversal_ratio(visualised[2], visualised[3], masks[i]))
        all_points = np.concatenate(all_points_list, axis=1)
        all_points=np.transpose(all_points[0:2])
        return all_points, distances, time_reversal, trajectories
    
    def log(self, imgs,batch, masks,writer,labels=["actual","generated"]):
        collected_arrays = []
        collected_distances = []
        generated_points = []
        actual_points = []
        generated_time_reversal = []
        actual_time_reversal = []
        masks  = F.interpolate(
            masks, size=(12,12), mode="nearest-exact"
        )
        masks = [torch.count_nonzero(
            mask[0]).item() for mask in masks]
        for i in range(len(labels)):
            img = imgs[i]
            label = labels[i]
            all_points = np.empty((CHANNELS,0))
            distances = []
            visualised = []
            for j in range(len(img)):
                split = img[j]
                split = torch.transpose(split, 0, 2)
                new_vis = np.transpose(self.revert_img(split, 0))
                total_distance = calculate_total_distance(new_vis, masks[j])
                visualised.append(new_vis)
                distances.append(total_distance)
                all_points = np.concatenate((all_points,new_vis),axis=1) 
            collected_arrays.append(all_points)
            collected_distances.append(distances)
            # This doesnt really help us much.
            # plot_buf = self.gen_plot(all_points)
            # if plot_buf != 0:
            #     image = PIL.Image.open(plot_buf)
            #     image = ToTensor()(image).unsqueeze(0).squeeze(0)
            #     writer.add_image(f"{label}/All Points", image, batch)
            writer.add_histogram(f"{label}/Histogram",np.array(distances),batch)
            for k in range(10):
                split = img[k]
                split = torch.transpose(split, 0, 2)
                vis = visualised[k]
                time_reversal = time_reversal_ratio(vis[2], vis[3], masks[k])
                if time_reversal > 0 and label == "actual":
                    log.error(split)
                    log.error(time_reversal)
                    log.error(vis[0],vis[1],vis[2])
                    log.error("FAILED TO REVERSE_TIME")
                    exit(1)
                plot_buf = self.gen_plot(vis)
                if plot_buf != 0:
                    image = PIL.Image.open(plot_buf)
                    image = ToTensor()(image)
                    writer.add_image(f"{label}/Trajectories", image, self.actual_display)
                #writer.add_image(f"{label}/Tensors", split.detach().permute(2,1,0), self.actual_display)
                writer.add_scalar(f"{label}/Time Reversal Ratio",time_reversal,self.actual_display)
                generated_time_reversal.append(time_reversal)
                actual_time_reversal.append(0)
                self.actual_display += 1
        t1 = np.transpose(collected_arrays[0][0:2])
        t2 = np.transpose(collected_arrays[1][0:2])
        hausdorff = fast_hausdorff_distance(t1, t2, distance='haversine')
        writer.add_scalar(f"Generated/All Points Haussdorff Distance",hausdorff,global_step=batch)
        emd = wasserstein_distance(collected_distances[0], collected_distances[1])
        writer.add_scalar(f"Generated/Total Distance Traveled EMD",emd,global_step=batch)
        time_reversal = wasserstein_distance(generated_time_reversal, actual_time_reversal)
        writer.add_scalar(f"Generated/Time Reversal Distance",time_reversal,global_step=batch)
        
    def __len__(self):
        return self.len

    def get_cloud_points(self, actual,generated, label):
        return (actual[:, 0],actual[:, 1]),(generated[:, 0],generated[:, 1])

    def plot_trajectory(self, visualised, length):
        fig = go.Figure(
                go.Scattermapbox(
                    name="Real Trajectory",
                    mode=("markers+lines"),
                    lon=visualised[1][:length],

                    lat=visualised[0][:length],
                    marker={"size": 10},
                )
            )
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_center_lat=mean(np.array(visualised[0])),
            mapbox_center_lon=mean(np.array(visualised[1])),
            mapbox_zoom=11,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        return fig

    def __getitem__(self, idx):
        return (self.trajectories[idx], self.masks[idx])

    def get_singular_traj(self, traj):
        img = []
        # Keeping Track of Variables, setting all ranges to max negative and positive.
        # Getting the first sequence number.
        img = self.get_new_img(traj)
        array = np.zeros((CHANNELS, IMG_SIZE, IMG_SIZE))
        counter = 0
        for i in range(int(math.sqrt(len(img)))):
            for j in range(int(math.sqrt(len(img)))):
                array[0][i][j] = img[counter][0]
                array[1][i][j] = img[counter][1]
                array[2][i][j] = img[counter][2]
                array[3][i][j] = img[counter][3]
                counter += 1
        m = nn.Upsample(scale_factor=2, mode="nearest")
        array = m(torch.from_numpy(array).unsqueeze(0))
        array = array.squeeze(0).cpu().numpy()
        return array

    def revert_img(self, array, idx):
        array = torch.transpose(array, 0, 2)
        array = self.remove_padding(array)
        array = F.interpolate(
            array.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode="nearest-exact"
        )
        # Removing the extra dimension
        array = array.squeeze(0)
        array = torch.flatten(array, start_dim=1, end_dim=2)
        array = torch.transpose(array, 0, 1)
        img = np.array(array.detach().cpu(),dtype=float)
        for i in range(len(img)):
            # img[i, :] = img[i, :] + [
            #     1,
            #     1,
            #     1,
            #     1,
            # ]
            # img[i, :] = img[i, :] / [
            #     2,
            #     2,
            #     2,
            #     2,
            # ]
            img[i, :] = img[i, :] * [
                self.max_lat - self.min_lat,
                self.max_lon - self.min_lon,
                6,
                23
            ]
            img[i, :] = img[i, :] + [
                self.min_lat,
                self.min_lon,
                0,
                0
            ]
            img[i, -2:] = np.round(img[i, -2:])
        return img

    def get_new_img(self, traj): 
        # Start getting through trajectories.
        img = []
        for values in traj:
            # Day number + hour normalised.
            values = values[2:]
            img.append(values)
        img = np.array(img)
        for i in range(len(img)):
            img[i, :] = img[i, :] - [
                self.min_lat,
                self.min_lon,
                0,
                0,
            ]
            img[i, :] = img[i, :] / [
                self.max_lat - self.min_lat,
                self.max_lon - self.min_lon,
                6,
                23,
            ]
            # img[i, :] = img[i, :] * [
            #     2,
            #     2,
            #     2,
            #     2,
            # ]
            # img[i, :] = img[i, :] - [
            #     1,
            #     1,
            #     1,
            #     1,
            # ]
        
        return img



