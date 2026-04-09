# for code release

import os
import pickle
import numpy as np
import trimesh
import torch
import torch.nn.functional as F
import argparse
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
from tqdm import tqdm
from typing import Dict, List, Optional
from collections import deque
from scipy.spatial.transform import Rotation as R
import rerun as rr
import rerun.blueprint as rrb
from smplx import SMPLX
from utils.rotation2 import convert_rotation
from utils.metrics import compute_mpjpe
import matplotlib.pyplot as plt
import custom_path as cp

@dataclass(frozen=True)
class ViewerConfig:
    output_rrd: Optional[Path] = None
    sample_fps: float = 1
    rotate_rgb: bool = True
    downsample_rgb: bool = True
    jpeg_quality: int = 90
    traj_tail_length: int = 100

    point_radii: float = 0.008
    line_radii: float = 0.008
    skel_radii: float = 0.01

@dataclass(frozen=True)
class MotionVizSequenceParams:
    """Paths to prediction (required) and optional baseline .npy rollouts."""

    predict_path: str
    baseline_path: Optional[str] = None
    show_pose_axis_plot: bool = False


def smooth_time_series(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """
    Smooth a [ntime, nvar] tensor along the time dimension using
    a moving-average filter.

    Args:
        x: [ntime, nvar]
        kernel_size: odd int, size of temporal smoothing window

    Returns:
        smoothed: [ntime, nvar]
    """
    assert x.dim() == 2, "x must be [ntime, nvar]"
    ntime, nvar = x.shape

    # [ntime, nvar] -> [1, nvar, ntime] for conv1d (batch, channels, length)
    x_ = x.T.unsqueeze(0)

    # Moving-average kernel
    kernel = torch.ones(1, 1, kernel_size, device=x.device, dtype=x.dtype)
    kernel = kernel / kernel_size
    # One kernel per variable (groups conv)
    kernel = kernel.expand(nvar, 1, kernel_size)

    padding = kernel_size // 2  # keep same length

    smoothed = F.conv1d(x_, kernel, padding=padding, groups=nvar)  # [1, nvar, ntime]
    smoothed = smoothed.squeeze(0).T  # -> [ntime, nvar]

    return smoothed

class DataViewer(ViewerConfig):

    palette: Dict[str, list] = {
        "scene": [200, 200, 200],
        "smplx": [255, 180, 0],
        "bev": [150, 150, 255],
        "traj": [
            [81, 71, 252], 
            [80, 244, 204],
            [255, 34, 17],
        ],
    }

    def __init__(
        self,
        smplx_model_path: str,
        object_mesh_root: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        rr.init(
            "data viewer 3",
            spawn=(self.output_rrd is None),
            recording_id=uuid4()
        )
        if self.output_rrd is not None:
            rr.save(self.output_rrd)

        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

        # Load SMPL-X model
        self.smplx_model = SMPLX(model_path=smplx_model_path, use_pca=False)
        
        self._epaths_3d: set[str] = set()
        self._traj_deques: dict[str, deque] = {}
        self._log_coordinate_axes()  # world coordinate axes

        self.seq_len_limit = 1000     # avoid logging too long sequences
        self.log_description = False  # log sample descriptions
        self.object_mesh_root = object_mesh_root

    @torch.no_grad()
    def __call__(self, seq: MotionVizSequenceParams) -> None:
        predict_path = str(Path(seq.predict_path).resolve())

        # Get the canonical pelvis position (without any transformations)
        default_smplx_output = self.smplx_model()
        rest_pelvis = default_smplx_output.joints[0, 0].detach().cpu().numpy()

        """
        dict_keys(['imu_traj', 'motion_smpl', 'text', 'objects'])

        imu_traj: [n_frame, n_imu, 6]
        motion_smpl: [n_frame, 75]  # global_orient(3), body_pose(63), global_transl(3)
        text: list of str
        objects: dict of object_name -> object_data
            {object_name: data: [7] (quaternion(4), transl(3))}
        """
        predict_data = np.load(predict_path, allow_pickle=True).item()
        # predict_data['pred'] = predict_data['gt'] # FIXME: remove this
        original_mpjpe = compute_mpjpe(predict_data)[0]
        mpjpe = original_mpjpe

        baseline_data = None
        if seq.baseline_path is not None:
            baseline_path = str(Path(seq.baseline_path).resolve())
            baseline_data = np.load(baseline_path, allow_pickle=True).item()

        print(f'original MPJPE: {original_mpjpe:.2f} mm, merged MPJPE: {mpjpe:.2f} mm, difference: {mpjpe - original_mpjpe:.2f} mm')

        for obj_name, obj_data in predict_data['pred']['objects'].items():
            if isinstance(obj_data['rot'], torch.Tensor):
                predict_data['pred']['objects'][obj_name]['rot'] = predict_data['pred']['objects'][obj_name]['rot'].float().detach().cpu().numpy()
            if isinstance(obj_data['transl'], torch.Tensor):
                predict_data['pred']['objects'][obj_name]['transl'] = predict_data['pred']['objects'][obj_name]['transl'].float().detach().cpu().numpy()

        gt_sample = predict_data['gt']
        gt_smpl_transl = gt_sample['transl'].reshape(-1, 3)
        gt_smpl_orient = gt_sample['orient'].reshape(-1, 3)
        gt_smpl_pose = gt_sample['pose'].reshape(-1, 21, 3)
        n_time = gt_smpl_orient.shape[0]

        pred_sample = predict_data['pred']
        pred_smpl_transl = pred_sample['transl'].reshape(-1, 3)
        pred_smpl_orient = pred_sample['orient'].reshape(-1, 3)
        pred_smpl_pose = pred_sample['pose'].reshape(-1,  3)

        pred_smpl_pose = convert_rotation(torch.from_numpy(pred_smpl_pose).float(), 'aa', '6d').reshape(-1, 21*6)
        # pred_smpl_pose = reshape_and_smooth_timeseries(pred_smpl_pose.float().numpy(), window_size=2)
        pred_smpl_pose = smooth_time_series(pred_smpl_pose.float(), kernel_size=5).reshape(-1, 21*6).numpy()
        pred_smpl_pose = convert_rotation(torch.from_numpy(pred_smpl_pose).float().reshape(-1, 6), '6d', 'aa').reshape(-1, 21, 3).cpu().numpy()    
        pred_sample['pose'] = pred_smpl_pose

        ground_height = pred_sample['objects']['ground']['transl'][1] # y of the ground plane
        pred_ground_vec = np.array([0, ground_height,  0]) # Just for visualization!
        viz_ground_vec = pred_ground_vec

        if 'ground' in gt_sample['objects']:
            ground_height = gt_sample['objects']['ground']['transl'][1] # y of the ground plane
            if isinstance(ground_height, torch.Tensor):
                ground_height = ground_height.detach().cpu().numpy()
            gt_ground_vec = np.array([0, ground_height,  0]) # Just for visualization!
            viz_ground_vec = gt_ground_vec
        
        sample_gt = {
            'transl': gt_smpl_transl - gt_ground_vec - rest_pelvis,
            'orient': gt_smpl_orient,
            'pose': gt_smpl_pose
        }
        sample_pred = {
            'transl': pred_smpl_transl - pred_ground_vec - rest_pelvis,
            'orient': pred_smpl_orient,
            'pose': pred_smpl_pose
        }
        
        # Process baseline data if available
        sample_baseline = None
        if baseline_data is not None:
            baseline_sample = baseline_data.get('pred', baseline_data)  # Try 'pred' key, fallback to root
            
            # Convert tensors to numpy if needed
            if isinstance(baseline_sample['transl'], torch.Tensor):
                baseline_sample['transl'] = baseline_sample['transl'].float().detach().cpu().numpy()
            if isinstance(baseline_sample['orient'], torch.Tensor):
                baseline_sample['orient'] = baseline_sample['orient'].float().detach().cpu().numpy()
            if isinstance(baseline_sample['pose'], torch.Tensor):
                baseline_sample['pose'] = baseline_sample['pose'].float().detach().cpu().numpy()
            
            baseline_smpl_transl = baseline_sample['transl'].reshape(-1, 3)
            baseline_smpl_orient = baseline_sample['orient'].reshape(-1, 3)
            baseline_smpl_pose = baseline_sample['pose'].reshape(-1, 21, 3)
            
            # Use GT ground height for baseline alignment (baseline has no objects)
            sample_baseline = {
                'transl': baseline_smpl_transl - gt_ground_vec - rest_pelvis,
                'orient': baseline_smpl_orient,
                'pose': baseline_smpl_pose
            }
        
        seq_len = n_time

        gt_text = gt_sample['description'] # list of strings
        pred_text = pred_sample['description'] # single string

        if seq.show_pose_axis_plot:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            key = 'pose'
            time = np.arange(sample_gt[key].shape[0]) / self.sample_fps
            ax.plot(time, sample_gt[key][:, 0], label='X', color='r')
            ax.plot(time, sample_gt[key][:, 1], label='Y', color='g')
            ax.plot(time, sample_gt[key][:, 2], label='Z', color='b')
            ax.plot(time, sample_pred[key][:, 0], label='X_pred', color='r', linestyle='dashed')
            ax.plot(time, sample_pred[key][:, 1], label='Y_pred', color='g', linestyle='dashed')
            ax.plot(time, sample_pred[key][:, 2], label='Z_pred', color='b', linestyle='dashed')
            ax.set_xlabel('Time (s)')
            if key == 'transl':
                ax.set_ylabel('Position (m)')
                ax.set_ylim([-0.15, 0.85])
                ax.set_title('Trajectory')
            elif key == 'orient':
                ax.set_ylabel('Orientation (axis-angle)')
                ax.set_ylim([-1.5, 1.5])
                ax.set_title('Pelvis Orientation (axis-angle)')
            elif key == 'pose':
                ax.set_ylabel('Pose (axis-angle)')
                ax.set_ylim([-1.5, 1.5])
                ax.set_title('Pelvis Pose (axis-angle)')
            ax.legend()
            plt.show()

        # mixamo_vertices = fitted_smplx['mixamo_vertices'] + transl[:, None] # [n_frame, n_point, 3]
        # mixamo_faces = fitted_smplx['mixamo_faces'] # [n_face, 3]

        # load object
        gt_objects = gt_sample['objects']
        objects_data = []
        for obj_name, obj_data in gt_objects.items():
            obj_name = obj_name.split('.')[0]
            # obj_name: std
            # obj_data: [7] # quaternion(4) + transl(3)
            if obj_name != 'ground': 
                object_mesh_path = os.path.join(self.object_mesh_root, obj_name, f'{obj_name}.obj')
                if not os.path.exists(object_mesh_path):
                    raise FileNotFoundError(f'Object mesh not found: {object_mesh_path}')
                mesh = trimesh.load(object_mesh_path, process=False)
                if isinstance(mesh, trimesh.Scene):
                    # Convert all geometry in the scene into a single mesh
                    mesh = trimesh.util.concatenate(mesh.dump())
                else:
                    mesh = mesh  
                vertices = mesh.vertices  # [n_point, 3]
                faces = mesh.faces  # [n_face, 3]

            if isinstance(obj_data['rot'], torch.Tensor):
                obj_data['rot'] = obj_data['rot'].detach().cpu().numpy()
            rot_mat = convert_rotation(torch.from_numpy(obj_data['rot']).float(), '6d', 'mat').numpy()  # [3, 3]
            transl_vec = obj_data['transl']  # [3]
            
            # apply rotation and translation
            if obj_name == 'ground':
                size = 10
                y = 0
                vertices = np.array([
                    [-size, y, -size],
                    [ size, y, -size],
                    [ size, y,  size],
                    [-size, y,  size],
                ])  # [4, 3]
                faces = np.array([
                    [0, 1, 2],
                    [0, 2, 3],
                ])  # [2, 3]
                obj_verts = ((vertices + transl_vec) - gt_ground_vec)[None, :, :]  # [1, n_point, 3]
            else:
                # obj_verts = (((vertices @ rot_mat.T + transl_vec))  * scale_factors - gt_ground_vec)[None, :, :]  # [1, n_point, 3]
                obj_verts = ((vertices @ rot_mat.T + transl_vec) - gt_ground_vec)[None, :, :]  # [1, n_point, 3]
            obj_verts = np.repeat(obj_verts, n_time, axis=0)  # [n_frame, n_point, 3] static object for now
            if obj_name != 'ground':
                obj_verts = obj_verts  # for visualization, align with human origin
            objects_data.append({
                'name': obj_name,
                'verts': obj_verts, # for visualization, align with human origin
                'faces': faces,  # [1, n_face, 3]
                'color': [100, 255, 100],
            })
            _=1

        pred_objects = pred_sample['objects']
        pred_objects_data = []
        for obj_name, obj_data in pred_objects.items():
            obj_name = obj_name.split('.')[0]
            # obj_name: std
            # obj_data: [7] # quaternion(4) + transl(3)
            if obj_name != 'ground': 
                object_mesh_path = os.path.join(self.object_mesh_root, obj_name, f'{obj_name}.obj')
                if not os.path.exists(object_mesh_path):
                    raise FileNotFoundError(f'Object mesh not found: {object_mesh_path}')
                mesh = trimesh.load(object_mesh_path, process=False)
                if isinstance(mesh, trimesh.Scene):
                    # Convert all geometry in the scene into a single mesh
                    mesh = trimesh.util.concatenate(mesh.dump())
                else:
                    mesh = mesh  
                vertices = mesh.vertices  # [n_point, 3]
                faces = mesh.faces  # [n_face, 3]

            if isinstance(obj_data['rot'], torch.Tensor):
                obj_data['rot'] = obj_data['rot'].detach().cpu().numpy()
            if isinstance(obj_data['transl'], torch.Tensor):
                obj_data['transl'] = obj_data['transl'].detach().cpu().numpy()
            rot_mat = convert_rotation(torch.from_numpy(obj_data['rot']).float(), '6d', 'mat').numpy()  # [3, 3]
            transl_vec = obj_data['transl']  # [3]
            
            # apply rotation and translation
            if obj_name == 'ground':
                size = 10
                y = 0
                vertices = np.array([
                    [-size, y, -size],
                    [ size, y, -size],
                    [ size, y,  size],
                    [-size, y,  size],
                ])  # [4, 3]
                faces = np.array([
                    [0, 1, 2],
                    [0, 2, 3],
                ])  # [2, 3]
                obj_verts = ((vertices + transl_vec) - pred_ground_vec)[None, :, :]  # [1, n_point, 3]
            else:
                obj_verts = (((vertices @ rot_mat.T + transl_vec) - pred_ground_vec)  )[None, :, :]  # [1, n_point, 3]
            obj_verts = np.repeat(obj_verts, n_time, axis=0)  # [n_frame, n_point, 3] static object for now
            if obj_name != 'ground':
                obj_verts = obj_verts  # for visualization, align with human origin
            pred_objects_data.append({
                'name': obj_name,
                'verts': obj_verts, # for visualization, align with human origin
                'faces': faces,  # [1, n_face, 3]
                'color': [255, 150, 150],  # Light red/pink to distinguish from GT objects
            })
        


        # object_path = f'/media/tianhang/Getea/humoto_0805_render_canonical/{seq_name}/gt_objects.pkl'
        # with open(object_path, 'rb') as f:
        #     objects = pickle.load(f)
        # # structure objects into a list of dicts for easy per-frame logging
        # objects_data = []
        # for obj_name, obj_data in objects.items():
        #     # obj_data: [verts[n_frame, n_point, 3], faces[n_face, 3], color[3]]
        #     obj_verts = obj_data[0].detach().cpu().numpy()
        #     # apply inverse scale and translation to the vertices
        #     # obj_verts = (obj_verts - mean_coord + translation / scale_factors) * scale_factors
        #     obj_verts = (obj_verts ) * scale_factors
        #     # obj_verts = (obj_verts_canonical @ pred_rot + pred_transl - mean_coord + translation / scale_factors) * scale_factors
        #     obj_faces = obj_data[1].detach().cpu().numpy()
        #     obj_color = [100, 255, 100]
        #     objects_data.append({
        #         'name': obj_name,
        #         'verts': obj_verts,
        #         'faces': obj_faces,
        #         'color': obj_color,
        #     })


        # sample_gt = sample_pred.copy()
        # seqlen = sample_pred['transl'].shape[0]  # Use the length of the transl sequence as seqlen
        # sample_pred['transl'] = sample_pred['transl'][:seqlen] 
        # # sample_pred['transl'] = sample_gt['transl']
        # sample_pred['orient'] = sample_pred['orient'][:seqlen]
        # sample_pred['pose'] = sample_pred['pose'][:seqlen]

        # Load SMPL-X meshes for all samples
        smplx_meshes_list = []
        smplx_meshes_list_quant = []
        smplx_meshes_list_baseline = []
        smplx_trajs_list = []
        smplx_trajs_list_quant = []
        smplx_trajs_list_baseline = []
        # smplx_orient_list = []
        
        meshes = self._load_smplx_meshes(sample_gt)
        smplx_meshes_list.append(meshes)
        
        # Trajectories from simulating the SMPL-X model
        # trajs_sim = self._compute_pelvis_trajectories(sample)

        # Trajectories from applying transformations to the pelvis original position
        meshes_quant = self._load_smplx_meshes(sample_pred)
        smplx_meshes_list_quant.append(meshes_quant)
        
        # Load baseline meshes if available
        if sample_baseline is not None:
            meshes_baseline = self._load_smplx_meshes(sample_baseline)
            smplx_meshes_list_baseline.append(meshes_baseline)
            smplx_trajs_list_baseline.append(sample_baseline['transl'])
        
        smplx_trajs_list.append(sample_gt['transl'])
        smplx_trajs_list_quant.append(sample_pred['transl'])


        # TODO: debug to log pelvis orientations
        # self._log_orientations(smplx_orient_list)
        
        # Visualize trajectories
        # texts = [sample["description"] for sample in samples] if self.log_description else None
        self._log_trajectories(smplx_trajs_list, labels=None, is_pred=False)
        self._log_trajectories(smplx_trajs_list_quant, labels=None, is_pred=True)
        if len(smplx_trajs_list_baseline) > 0:
            self._log_trajectories(smplx_trajs_list_baseline, labels=None, is_pred=False, is_baseline=True)
        # print(f'gt description: {sample_gt["description"]}\n')
        # print(f'pred description: {sample_pred["description"]}\n')
            
        # Determine sequence length (max among all samples)
        # seqlen = max([len(meshes) for meshes in smplx_meshes_list])
        
        # Log sequences frame by frame
        dt = 1.0 / self.sample_fps
        for frame_idx in tqdm(range(seq_len)):
            rr.set_time_sequence("frames", frame_idx)
            rr.set_time_seconds("sensor_time", frame_idx * dt)
            
            # Log SMPL-X meshes for this frame
            meshes_at_frame = []
            for meshes in smplx_meshes_list:
                if frame_idx < len(meshes):
                    meshes_at_frame.append(meshes[frame_idx])
                else:
                    meshes_at_frame.append(meshes[-1])  # Use last frame if out of range
            self._log_smplx_meshes(meshes_at_frame)

            meshes_at_frame_quant = []
            for meshes in smplx_meshes_list_quant:
                if frame_idx < len(meshes):
                    meshes_at_frame_quant.append(meshes[frame_idx])
                else:
                    meshes_at_frame_quant.append(meshes[-1])
            self._log_smplx_meshes(meshes_at_frame_quant, is_pred=True)

            # Log baseline meshes for this frame
            if len(smplx_meshes_list_baseline) > 0:
                meshes_at_frame_baseline = []
                for meshes in smplx_meshes_list_baseline:
                    if frame_idx < len(meshes):
                        meshes_at_frame_baseline.append(meshes[frame_idx])
                    else:
                        meshes_at_frame_baseline.append(meshes[-1])
                self._log_smplx_meshes(meshes_at_frame_baseline, is_pred=False, is_baseline=True)

            # Log Mixamo mesh for this frame
            # if frame_idx < len(mixamo_vertices):
            #     # Create trimesh object to compute vertex normals
            #     # mixamo_mesh = trimesh.Trimesh(
            #     #     vertices=mixamo_vertices[frame_idx]  # [n_point, 3],
            #     #     faces=mixamo_faces,
            #     #     process=False
            #     # )
            #     mixamo_mesh = trimesh.Trimesh(
            #         vertices=mixamo_vertices[frame_idx],
            #         faces=mixamo_faces,
            #         process=False
            #     )
            #     ep = "world/mixamo_mesh"
            #     # Use green color to distinguish from SMPL meshes
            #     mixamo_color = [100, 255, 100]  # Light green
            #     rr.log(
            #         ep,
            #         rr.Mesh3D(
            #             vertex_positions=mixamo_mesh.vertices,
            #             vertex_normals=mixamo_mesh.vertex_normals,
            #             triangle_indices=mixamo_mesh.faces,
            #             vertex_colors=mixamo_color,
            #         ),
            #         static=False,
            #     )
            #     self._epaths_3d.add(ep)

            # Log variable number of objects for this frame (aligned with human origin)
            if len(objects_data) > 0:
                for obj in objects_data:
                    obj_len = obj['verts'].shape[0]
                    frame_id = frame_idx if frame_idx < obj_len else obj_len - 1
                    # verts_f = obj['verts'][frame_id] - init_global_transl[None, :]
                    verts_f = obj['verts'][frame_id]
                    faces = obj['faces']
                    color =np.array(obj['color'])
                    
                    # Create trimesh object to compute vertex normals
                    obj_mesh = trimesh.Trimesh(
                        vertices=verts_f,
                        faces=faces,
                        process=False
                    )
                    
                    ep = f"world/obj/{obj['name']}"
                    rr.log(
                        ep,
                        rr.Mesh3D(
                            vertex_positions=obj_mesh.vertices,
                            vertex_normals=obj_mesh.vertex_normals,
                            triangle_indices=obj_mesh.faces,
                            vertex_colors=color,
                        ),
                        static=False,
                    )
                    self._epaths_3d.add(ep)

            # Log predicted objects for this frame (aligned with human origin)
            if len(pred_objects_data) > 0: 
                for obj in pred_objects_data:
                    obj_len = obj['verts'].shape[0]
                    frame_id = frame_idx if frame_idx < obj_len else obj_len - 1
                    # verts_f = obj['verts'][frame_id] - init_global_transl[None, :]
                    verts_f = obj['verts'][frame_id]
                    faces = obj['faces']
                    color = np.array(obj['color'])
                    
                    # Create trimesh object to compute vertex normals
                    obj_mesh = trimesh.Trimesh(
                        vertices=verts_f,
                        faces=faces,
                        process=False
                    )
                    
                    ep = f"world/obj_pred/{obj['name']}"
                    rr.log(
                        ep,
                        rr.Mesh3D(
                            vertex_positions=obj_mesh.vertices,
                            vertex_normals=obj_mesh.vertex_normals,
                            triangle_indices=obj_mesh.faces,
                            vertex_colors=color,
                        ),
                        static=False,
                    )
                    self._epaths_3d.add(ep)

            # Log text labels near the human
            self._log_text_labels(
                meshes_at_frame, 
                meshes_at_frame_quant,
                gt_text, 
                pred_text, 
                frame_idx
            )

            # Log IMU data for this frame
            # if log_imus:
            #     self._log_imu_transforms(sample_input, frame_idx, sample_idx=0)


        # save all info into 1 file
        """
        sample_pred: {'transl': [n_frame, 3], 'orient': [n_frame, 3], 'pose': [n_frame, 21, 3]}
        sample_gt: {'transl': [n_frame, 3], 'orient': [n_frame, 3], 'pose': [n_frame, 21, 3]}
        objects_data: list of dicts, each dict contains 'name', 'verts', 'faces', 'color'
            e.g. {'name': 'bowl', 'verts': [n_frame, n_point, 3], 'faces': [n_face, 3], 'color': [3]}
        smplx_meshes_vertices: [n_frame, n_point, 3]
        smplx_meshes_faces: [n_face, 3]
        """
        stem = Path(predict_path).stem
        seq_name = stem
        save_path = str(Path(predict_path).with_name(f'{stem}_viz_info.pkl'))
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        meshes_vertices_gt = np.stack([mesh.vertices for mesh in smplx_meshes_list[0]], axis=0) # [n_frame, n_point, 3]
        meshes_vertices_pred = np.stack([mesh.vertices for mesh in smplx_meshes_list_quant[0]], axis=0) # [n_frame, n_point, 3]
        if len(smplx_meshes_list_baseline) > 0:
            meshes_vertices_baseline = np.stack([mesh.vertices for mesh in smplx_meshes_list_baseline[0]], axis=0) # [n_frame, n_point, 3]
        else:
            meshes_vertices_baseline = meshes_vertices_pred


        viz_info = {
            'seq_name': seq_name, # string
            'sample_pred': sample_pred, # dict, 'transl' [n_frame, 3], 'orient' [n_frame, 3], 'pose' [n_frame, 21, 3]
            'sample_gt': sample_gt, # dict, 'transl' [n_frame, 3], 'orient' [n_frame, 3], 'pose' [n_frame, 21, 3]
            'objects_data_gt': objects_data, # list of dicts
            'objects_data_pred': pred_objects_data, # list of dicts
            'human_faces': self.smplx_model.faces, # [n_face, 3]
            'human_vertices_gt': meshes_vertices_gt, # [n_frame, n_point, 3]
            'human_vertices_pred': meshes_vertices_pred, # [n_frame, n_point, 3]
            'human_vertices_baseline': meshes_vertices_baseline, # [n_frame, n_point, 3]
        }
        pickle.dump(viz_info, open(save_path, 'wb'))
        print(f'Visualization info saved to {save_path}')
        exit(0)

    def _log_imu_transforms(self, sample, frame_idx, sample_idx) -> None:
        """
        Log IMU transformations for the current frame index
        """
        devices = ["airpod", "watch", "iphone"]
        device_colors = {
            "airpod": [0, 191, 255],  # Deep Sky Blue
            "watch": [255, 69, 0],    # Orangered
            "iphone": [50, 205, 50]   # Lime Green
        }

        axis_scale = 0.1  # Scale for the coordinate axes

        for device in devices:
            trans_key = f'imu_trans_{device}'
            if trans_key not in sample:
                continue

            device_color = device_colors[device]
            imu_trans = sample[trans_key]  # Shape: [seq_len, 6]

            # log full trajectory path only once (when first frame is processed)
            if frame_idx == 0:
                positions = imu_trans[:, 3:6]
                rr.log(
                    f"world/imu_traj_{sample_idx}_{device}",
                    rr.LineStrips3D(
                        positions,
                        colors=device_color,
                        radii=self.line_radii * 0.8,
                        labels=[device]
                    ),
                    static=True,
                )
            
            if frame_idx < len(imu_trans):
                frame_data = imu_trans[frame_idx]
            else:
                frame_data = imu_trans[-1]

            rot_vec = frame_data[0:3]
            trans_vec = frame_data[3:6]
            rot_matrix = R.from_rotvec(rot_vec).as_matrix()

            # calculate the local coordinate axes
            origin = trans_vec
            x_axis = origin + rot_matrix[:, 0] * axis_scale
            y_axis = origin + rot_matrix[:, 1] * axis_scale
            z_axis = origin + rot_matrix[:, 2] * axis_scale

            ep_base = f"world/imu_local_rot_{sample_idx}_{device}"

            # Log X-axis (red)
            rr.log(
                f"{ep_base}/x_axis",
                rr.LineStrips3D(
                    [origin, x_axis],
                    colors=[255, 0, 0],
                    radii=self.line_radii
                ),
                static=False,
            )
            
            # Log Y-axis (green)
            rr.log(
                f"{ep_base}/y_axis",
                rr.LineStrips3D(
                    [origin, y_axis],
                    colors=[0, 255, 0],
                    radii=self.line_radii
                ),
                static=False,
            )
            
            # Log Z-axis (blue)
            rr.log(
                f"{ep_base}/z_axis",
                rr.LineStrips3D(
                    [origin, z_axis],
                    colors=[0, 0, 255],
                    radii=self.line_radii
                ),
                static=False,
            )


    def _log_coordinate_axes(self, scale=0.5) -> None:
        """Add coordinate axes to visualize the world coordinate system"""
        # Define the axes
        origin = np.array([0, 0, 0])
        x_axis = np.array([scale, 0, 0]) 
        y_axis = np.array([0, scale, 0])
        z_axis = np.array([0, 0, scale])
        
        # Log X-axis (red)
        rr.log("world/axes/x", 
            rr.LineStrips3D(
                [origin, x_axis],
                colors=[255, 0, 0],
                radii=self.line_radii*1.5,
                labels=["X"]
            ),
            static=True)
        
        # Log Y-axis (green)
        rr.log("world/axes/y", 
            rr.LineStrips3D(
                [origin, y_axis],
                colors=[0, 255, 0],
                radii=self.line_radii*1.5,
                labels=["Y"]
            ),
            static=True)
        
        # Log Z-axis (blue) - the first frame's forward direction
        rr.log("world/axes/z", 
            rr.LineStrips3D(
                [origin, z_axis],
                colors=[0, 0, 255], 
                radii=self.line_radii*1.5,
                labels=["Z (forward)"]
            ),
            static=True)


    def _remove_wall(self, mesh) -> trimesh.Trimesh:
        """Remove the wall from the mesh"""
        vs = mesh.vertices
        v_min = np.min(vs, axis=0)
        v_max = np.max(vs, axis=0)
        # print(v_min, v_max)
        valid = (vs[:, 1] < v_max[1] - 0.1)
        fs = mesh.faces
        valid_fs = valid[fs.reshape(-1)].reshape(fs.shape[0], 3).min(axis=1)
        mesh.faces = fs[valid_fs]
        return mesh


    def _load_smplx_meshes(self, sample):
        """Convert SMPL-X parameters to meshes"""
        orients = sample['orient']
        poses = sample['pose']
        transls = sample['transl']
        
        meshes = []
        for i, (orient, pose, transl) in enumerate(zip(orients, poses, transls)):
            
            if i >= self.seq_len_limit:
                break

            smplx_output = self.smplx_model(
                global_orient=torch.tensor(orient[None]).float(),
                body_pose=torch.tensor(pose[None]).float(),
                transl=torch.tensor(transl[None]).float(),
            )
            mesh = trimesh.Trimesh(
                vertices=smplx_output.vertices.detach().cpu().numpy()[0], 
                faces=self.smplx_model.faces, 
                process=False
            )
            # set to orange
            meshes.append(mesh)

        return meshes
    

    def _compute_pelvis_trajectories(self, sample):
        """Extract pelvis trajectories from SMPL-X parameters"""
        orients = sample['orient']
        poses = sample['pose']
        transls = sample['transl']
        
        trajs = []
        for i, (orient, pose, transl) in enumerate(zip(orients, poses, transls)):

            if i >= self.seq_len_limit:
                break

            smplx_output = self.smplx_model(
                global_orient=torch.tensor(orient[None]).float(),
                body_pose=torch.tensor(pose[None]).float(),
                transl=torch.tensor(transl[None]).float(),
            )

            # The 0th joint is the pelvis
            trajs.append(smplx_output.joints[0, 0].detach().cpu().numpy())

        return np.array(trajs)
    

    def _log_trajectories(self, trajs_list, labels=None, is_pred=False, is_baseline=False) -> None:
        """Visualize trajectories of the pelvis"""
        for i, traj in enumerate(trajs_list):
            ep = f"world/traj_pelvis_{i}"
            if is_pred:
                ep = ep + "_pred"
            elif is_baseline:
                ep = ep + "_baseline"
            label = labels[i] if labels is not None else 'traj'

            if is_pred:
                if label is None:
                    label = "pred"
                else:
                    label += "_pred"
            elif is_baseline:
                if label is None:
                    label = "baseline"
                else:
                    label += "_baseline"
            
            # Use color from palette cycling through available colors
            if is_baseline:
                # Use a distinct color for baseline (purple/magenta)
                color = [255, 0, 255]  # Magenta
            else:
                color_idx = i % len(self.palette.get("traj"))
                color = self.palette.get("traj")[color_idx]
            
            rr.log(
                ep,
                rr.LineStrips3D(
                    traj,
                    colors=color,
                    radii=self.line_radii if not is_pred else self.line_radii * 0.3,
                    labels=[label] if label is not None else None,
                ),
                static=True,
            )
            self._epaths_3d.add(ep)
    

    def _log_smplx_meshes(self, meshes, is_pred=False, is_baseline=False) -> None:
        """Visualize SMPL-X meshes for the current frame"""
        for i, mesh in enumerate(meshes):
            ep = f"world/smplx_{i}"
            if is_pred:
                ep = ep + "_pred_human"
            elif is_baseline:
                ep = ep + "_baseline_human"
            cc = self.palette.get("smplx")
            if is_pred:
                # use light blue
                cc = [150, 150, 255]
            elif is_baseline:
                # use magenta/purple for baseline
                cc = [255, 0, 255]
            rr.log(
                ep,
                rr.Mesh3D(
                    vertex_positions=mesh.vertices,
                    vertex_normals=mesh.vertex_normals,
                    triangle_indices=mesh.faces,
                    vertex_colors=cc,
                ),
                static=False,
            )
            self._epaths_3d.add(ep)

    def _log_text_labels(self, gt_meshes, pred_meshes, gt_text, pred_text, frame_idx) -> None:
        """Log text labels near the human meshes using rerun's text logging"""
        # Get the position of the GT human (center of mesh, top of head)
        if len(gt_meshes) > 0 and gt_meshes[0] is not None:
            gt_mesh = gt_meshes[0]
            # Get the top of the mesh (highest Y coordinate)
            gt_vertices = gt_mesh.vertices
            gt_top_y = np.max(gt_vertices[:, 1])
            gt_center = np.mean(gt_vertices, axis=0)
            gt_text_pos = np.array([gt_center[0], gt_top_y + 0.1, gt_center[2]])  # Position above the head
            
            # Log GT text if not empty
            if gt_text and len(gt_text) > 0:
                # Handle both list and string cases
                if isinstance(gt_text, list):
                    text_to_show = gt_text[0] if len(gt_text) > 0 and gt_text[0] else None
                else:
                    text_to_show = gt_text if gt_text else None
                
                if text_to_show and text_to_show.strip():
                    # Use Points3D with labels to display text in 3D space
                    rr.log(
                        "world/text_labels/gt_text",
                        rr.Points3D(
                            positions=[gt_text_pos],
                            labels=[f"GT: {text_to_show}"],
                            radii=0.02,
                            colors=[255, 200, 0],  # Orange color for GT
                        ),
                        static=False,
                    )
        
        # Get the position of the predicted human
        if len(pred_meshes) > 0 and pred_meshes[0] is not None:
            pred_mesh = pred_meshes[0]
            # Get the top of the mesh (highest Y coordinate)
            pred_vertices = pred_mesh.vertices
            pred_top_y = np.max(pred_vertices[:, 1])
            pred_center = np.mean(pred_vertices, axis=0)
            pred_text_pos = np.array([pred_center[0], pred_top_y + 0.3, pred_center[2]])  # Position above the head
            
            # Log predicted text if not empty
            if pred_text and len(pred_text) > 0:
                # Handle both list and string cases
                if isinstance(pred_text, list):
                    # sort to get the longest text
                    pred_text.sort(key=len, reverse=True)
                    text_to_show = pred_text[0] if len(pred_text) > 0 and pred_text[0] else None
                else:
                    text_to_show = pred_text if pred_text else None
                
                if text_to_show and text_to_show.strip():
                    # Use Points3D with labels to display text in 3D space
                    rr.log(
                        "world/text_labels/pred_text",
                        rr.Points3D(
                            positions=[pred_text_pos],
                            labels=[f"Pred: {text_to_show}"],
                            radii=0.02,
                            colors=[150, 150, 255],  # Light blue color for prediction
                        ),
                        static=False,
                    )

    def _convert_occ_grid_to_pcd(self, occ_grid, scene_grid_params):
        """
        Convert an occupancy grid to 3D point cloud.
        
        Args:
            occupancy_grid: Boolean tensor/array of shape [nx, ny, nz]
            scene_grid_params: Parameters defining the grid dimensions [x_min, y_min, z_min, x_max, y_max, z_max, nx, ny, nz]
        
        Returns:
            points: Numpy array of shape [N, 3] containing the center coordinates of occupied voxels
        """
        # Extract grid parameters
        x_min, y_min, z_min = scene_grid_params[0:3]
        x_max, y_max, z_max = scene_grid_params[3:6]
        nx, ny, nz = scene_grid_params[6:9]

        # Calculate voxel size
        voxel_size_x = (x_max - x_min) / nx
        voxel_size_y = (y_max - y_min) / ny
        voxel_size_z = (z_max - z_min) / nz

        if isinstance(occ_grid, torch.Tensor):
            occ_grid = occ_grid.cpu().numpy()
        
        occupied_indices = np.nonzero(occ_grid)

        indices = np.stack(occupied_indices, axis=1)
        scales = np.array([voxel_size_x, voxel_size_y, voxel_size_z])
        mins = np.array([x_min, y_min, z_min])
        offsets = np.array([0.5, 0.5, 0.5])
        
        points = (indices + offsets) * scales + mins
    
        return points


def _parse_motion_viz_args() -> MotionVizSequenceParams:
    p = argparse.ArgumentParser(description='Visualize motion predictions in Rerun (SMPL-X + scene objects).')
    p.add_argument('-p', '--predict_path', type=str, help='Path to prediction rollout .npy (gt/pred structure as before).')
    p.add_argument(
        '-b', '--baseline_path',
        type=str,
        default=None,
        help='Optional path to baseline .npy for comparison.',
    )
    p.add_argument(
        '-s', '--show_pose_axis_plot',
        action='store_true',
        help='Show matplotlib axis-angle plot for pelvis pose before Rerun.',
    )
    args = p.parse_args()
    return MotionVizSequenceParams(
        predict_path=args.predict_path,
        baseline_path=args.baseline_path,
        show_pose_axis_plot=args.show_pose_axis_plot,
    )


if __name__ == '__main__':
    viewer = DataViewer(
        smplx_model_path=cp.smplx_model_dir,
        object_mesh_root=cp.humoto_objects_folder,
    )
    viewer(_parse_motion_viz_args())