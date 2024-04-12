import copy
import pickle
import torch
import numpy as np
from skimage import io
import sys,time
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from easydict import EasyDict
from pcdet.datasets.kitti import kitti_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from pcdet.datasets.dataset import DatasetTemplate
import torch.nn.functional as F


class KittiSemanticV2(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.kitti_infos = []
        self.include_kitti_data(self.mode)

        self.pt_feat_tag = self.dataset_cfg.get('pt_feat_tag', None) 
        if self.pt_feat_tag:
            pt_feat_dir = self.root_split_path / self.pt_feat_tag
            pt_feat_dir.mkdir(parents=True,exist_ok=True)
            pt_feat_dir1 = self.root_split_path / (self.pt_feat_tag + '_reduced')
            pt_feat_dir1.mkdir(parents=True, exist_ok=True)

    def include_kitti_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading KITTI dataset')
        kitti_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.save_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)

        self.kitti_infos.extend(kitti_infos)

        if self.logger is not None:
            self.logger.info('Total samples for KITTI dataset: %d' % (len(kitti_infos)))

    def reload_infos(self,split):
        if split == 'trainval':
            self.kitti_infos = []
            self.include_kitti_data(mode='train')
        elif split == 'test':
            self.kitti_infos = []
            self.include_kitti_data(mode='test')

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.pt_feat_tag = self.dataset_cfg.get('pt_feat_tag', False) 
        if self.pt_feat_tag:
            pt_feat_dir = self.root_split_path / self.pt_feat_tag
            pt_feat_dir.mkdir(parents=True, exist_ok=True)
            pt_feat_dir1 = self.root_split_path / (self.pt_feat_tag + '_reduced')
            pt_feat_dir1.mkdir(parents=True, exist_ok=True)

    def set_feats_dir(self, feats_dir):
        self.feats_dir = Path(feats_dir).resolve()
    def set_img_feat_dir(self,img_feat_dir):
        self.feats_dir = Path(self.root_split_path / img_feat_dir).resolve()

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_point_feats(self, idx, num_feats, reduced=False):
        dir_tag = self.pt_feat_tag+'{}'.format('_reduced' if reduced else '')
        lidar_file = self.root_split_path / dir_tag / ('%s.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, num_feats)

    def get_image(self, idx):
        """
        Loads image for a sample
        Args:
            idx: int, Sample index
        Returns:
            image: (H, W, 3), RGB Image
        """
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        image = io.imread(img_file)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def get_image_shape(self, idx):
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()
        return object3d_kitti.get_objects_from_label(label_file)

    def get_depth_map(self, idx):
        """
        Loads depth map for a sample
        Args:
            idx: str, Sample index
        Returns:
            depth: (H, W), Depth map
        """
        depth_file = self.root_split_path / 'depth_2' / ('%s.png' % idx)
        assert depth_file.exists()
        depth = io.imread(depth_file)
        depth = depth.astype(np.float32)
        depth /= 256.0
        return depth

    def get_calib(self, idx):
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    def get_road_plane(self, idx):
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info
            calib = self.get_calib(sample_idx)

            P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

            info['calib'] = calib_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                loc_lidar = calib.rect_to_lidar(loc)
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                loc_lidar[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar

                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_lidar(sample_idx)
                    calib = self.get_calib(sample_idx)
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])

                    fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    pts_fov = points[fov_flag]
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)


    def create_gt_dbinfos_withImg(self, info_path=None, used_classes=None, split='train', save_path=None):
        database_save_path = Path(save_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(save_path) / ('kitti_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']

            points = self.get_point_feats(sample_idx, num_feats=6) 

            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.kitti_infos[0].keys():
            return None, {}

        from .kitti_object_eval_python import eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs

        return len(self.kitti_infos)

    def __getitem__(self, index):

        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']
        calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            if "gt_boxes2d" in get_item_list:
                input_dict['gt_boxes2d'] = annos["bbox"]

            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        if "points" in get_item_list:
            if self.split == 'test': 
                self.dataset_cfg.Reduced = False
            if self.dataset_cfg.Reduced:
                points = self.get_point_feats(sample_idx, num_feats=6, reduced=True)
            else:
                points = self.get_point_feats(sample_idx, num_feats=6)
                if self.dataset_cfg.FOV_POINTS_ONLY:
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])
                    fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                    points = points[fov_flag]

            input_dict['points'] = points

        if "images" in get_item_list:
            input_dict['images'] = self.get_image(sample_idx)

        if "depth_maps" in get_item_list:
            input_dict['depth_maps'] = self.get_depth_map(sample_idx)

        if "calib_matricies" in get_item_list:
            input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calib)

        data_dict = self.prepare_data(data_dict=input_dict)

        data_dict['image_shape'] = img_shape
        return data_dict

    def sample_feats(self, index):
        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']
        calib = self.get_calib(sample_idx)

        print('dataset sample: %d/%d, %s' % (index + 1, len(self.kitti_infos), sample_idx))

        feats = self.get_img_feats(sample_idx)[:,0:img_shape[0],0:img_shape[1]]

        points = self.get_lidar(sample_idx)



        pts_img, pts_depth = calib.lidar_to_img(points[:, 0:3])

        key_corners = pts_img.copy()
        key_corners[:, 0] = (key_corners[:, 0] / (img_shape[1] - 1.0)) * 2.0 - 1.0
        key_corners[:, 1] = (key_corners[:, 1] / (img_shape[0] - 1.0)) * 2.0 - 1.0
        key_corners = torch.from_numpy(key_corners.astype(np.float32)).view(1, -1, 1, 2).contiguous().cuda()  # (N*k,2)

        imgfeat = torch.from_numpy(feats.astype(np.float32)).unsqueeze(0).cuda()


        key_feats = F.grid_sample(imgfeat, key_corners, padding_mode='border',
                                  align_corners=True).squeeze(0).squeeze(-1)
        key_feats = key_feats.transpose(0,1).cpu().numpy()


        point_feats = np.concatenate([points,key_feats],axis=1)

        if self.pt_feat_tag:
            point_feats_file = self.root_split_path / self.pt_feat_tag / ('%s.bin' % sample_idx)
            with open(str(point_feats_file), 'w') as f:
                point_feats.tofile(f)

        return point_feats


    def vis_feats(self,feats,pts_img,key_feats):
        """
        Args:
            feats: 
            pts_img: 
            key_feats:
        """
        vis_feats = np.zeros_like(feats)
        pts_img_int = np.floor(pts_img).astype(int)
        nums = pts_img_int.shape[0]
        for nn in range(nums):
            jj, ii = pts_img_int[nn]
            vis_feats[0, ii, jj] = key_feats[nn, 0]
            vis_feats[1, ii, jj] = key_feats[nn, 1]
        fig = plt.figure()
        fig.add_subplot(2, 2, 1)
        plt.imshow(feats[0])
        fig.add_subplot(2, 2, 2)
        plt.imshow(feats[1])
        fig.add_subplot(2, 2, 3)
        plt.imshow(vis_feats[0])
        fig.add_subplot(2, 2, 4)
        plt.imshow(vis_feats[1])
        plt.show()

    def reduced_pts(self, index):
        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']
        calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        points = self.get_point_feats(sample_idx, num_feats=6)

        if self.dataset_cfg.FOV_POINTS_ONLY:
            pts_rect = calib.lidar_to_rect(points[:, 0:3])
            fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
            points = points[fov_flag]

        if self.pt_feat_tag:
            point_feats_file = self.root_split_path / (self.pt_feat_tag+'_reduced') / ('%s.bin' % sample_idx)
            with open(str(point_feats_file), 'w') as f:
                points.tofile(f)

        print('dataset sample: %d/%d, %s' % (index + 1, len(self.kitti_infos), sample_idx))


def visualize_kitti_feats(dataset_cfg, class_names, data_path):

    dataset = KittiSemanticV2(dataset_cfg=dataset_cfg,class_names=class_names,root_path=data_path,training=True)
    feats_path = '/media/zlin/T4/kitti3D/segmentation/pred_kitti_training/output'
    dataset.set_feats_dir(feats_path)
    for k in range(len(dataset)):
        info = copy.deepcopy(dataset.kitti_infos[k])
        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']
        feats = dataset.get_img_feats(sample_idx)
        fig = plt.figure(figsize=(8, 8))
        fig.add_subplot(2, 1, 1)
        plt.imshow(feats[0])
        fig.add_subplot(2, 1, 2)
        plt.imshow(feats[1])
        plt.show()

def sample_img_feats(dataset_cfg, class_names, data_path, save_path):
    """
    @Functions:

    """
    dataset = KittiSemanticV2(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)

    trainval_split = 'trainval'
    test_split = 'test'

    # trainval
    dataset.set_split(trainval_split)
    dataset.reload_infos(trainval_split)
    dataset.set_img_feat_dir(img_feat_dir=dataset_cfg.img_feat_dir)

    for k in range(len(dataset)):
        point_feats = dataset.sample_feats(k)

    # test
    dataset.set_split(test_split)
    dataset.reload_infos(test_split)
    dataset.set_img_feat_dir(img_feat_dir=dataset_cfg.img_feat_dir)

    for k in range(len(dataset)):
        point_feats = dataset.sample_feats(k)

def create_kitti_dbinfos(dataset_cfg, class_names, data_path, save_path):

    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    dataset = KittiSemanticV2(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)

    print('---------------Start create groundtruth database for data augmentation---------------')
    trainval_split = 'trainval'
    dataset.set_split(trainval_split)
    dataset.reload_infos(trainval_split)

    trainval_filename = save_path / ('kitti_infos_%s.pkl' % trainval_split)

    dataset.set_img_feat_dir(img_feat_dir=dataset_cfg.img_feat_dir)
    dataset.create_gt_dbinfos_withImg(trainval_filename, split=trainval_split, save_path=save_path)

    print('---------------Data preparation Done---------------')

def create_kitti_infos(dataset_cfg, class_names, data_path, save_path, workers=4, test=False):

    train_split, val_split = 'train', 'val'
    test_split = 'test'
    train_filename = save_path / ('kitti_infos_%s.pkl' % train_split)
    val_filename = save_path / ('kitti_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'kitti_infos_trainval.pkl'
    test_filename = save_path / 'kitti_infos_test.pkl'
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    dataset = KittiSemanticV2(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)

    print('---------------Start to generate data infos---------------')

    if not train_filename.exists():
        dataset.set_split(train_split)
        kitti_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
        with open(train_filename, 'wb') as f:
            pickle.dump(kitti_infos_train, f)
        print('Kitti info train file is saved to %s' % train_filename)
    else:
        print('Kitti info train file is saved to %s' % train_filename)

    if not val_filename.exists():
        dataset.set_split(val_split)
        kitti_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
        with open(val_filename, 'wb') as f:
            pickle.dump(kitti_infos_val, f)
        print('Kitti info val file is saved to %s' % val_filename)

        with open(trainval_filename, 'wb') as f:
            pickle.dump(kitti_infos_train + kitti_infos_val, f)
        print('Kitti info trainval file is saved to %s' % trainval_filename)
    else:
        print('Kitti info val file is saved to %s' % val_filename)
        print('Kitti info trainval file is saved to %s' % trainval_filename)

    if not trainval_filename.exists():
        with open(train_filename, 'rb') as f:
            kitti_infos_train = pickle.load(f)
        with open(val_filename, 'rb') as f:
            kitti_infos_val = pickle.load(f)
        with open(trainval_filename, 'wb') as f:
            pickle.dump(kitti_infos_train + kitti_infos_val, f)
        print('Kitti info trainval file is saved to %s' % trainval_filename)
    # else:
    #     with open(trainval_filename, 'rb') as f:
    #         kitti_infos_trainval = pickle.load(f)
    #         print(len(kitti_infos_trainval))

    if not test_filename.exists():
        dataset.set_split('test')
        kitti_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
        with open(test_filename, 'wb') as f:
            pickle.dump(kitti_infos_test, f)
        print('Kitti info test file is saved to %s' % test_filename)
    else:
        print('not create Kitti info test file')

def TEST_training_data(dataset_cfg, class_names, data_path):
    """
    Args:
        dataset_cfg: 
    """
    dataset = KittiSemanticV2(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=True)
    for k in range(len(dataset)):
        st_time = time.time()
        dataset[k]
        ed_time = time.time()
        print('cost time: {:.2f} ms'.format(1000 * (ed_time - st_time)))

def generate_reduced_pts(dataset_cfg, class_names, data_path):
    # training=True, train set; training=False, val set.
    # dataset_cfg.DATA_SPLIT['test'] = 'train'
    # dataset_cfg.INFO_PATH['test'] = ['kitti_infos_train.pkl',]

    dataset = KittiSemanticV2(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)

    # trainval
    trainval_split = 'trainval'
    dataset.set_split(trainval_split)
    dataset.reload_infos(trainval_split)
    for k in range(len(dataset)):
        dataset.reduced_pts(k)

    # test
    # test_split = 'test'
    # dataset.set_split(test_split)
    # dataset.reload_infos(test_split)
    # for k in range(len(dataset)):
    #     dataset.reduced_pts(k)


if __name__ == '__main__':

    proj_dir = Path(__file__).resolve().parent.parent.parent.parent

    dataset_type = {

        '0': 'tools/cfgs/_data/kitti_testing_sem.yaml',
    }

    kitti_info_yaml = proj_dir / dataset_type['0']
    dataset_cfg = EasyDict(yaml.safe_load(open(str(kitti_info_yaml))))

    data_path = Path(dataset_cfg.DATA_PATH)

    save_tag = dataset_cfg.tag
    save_path = data_path / 'infos' / save_tag
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    print('\n********************************** create kitti infos **************************************\n')
    print('tag: {} ;\nsave_path: {} ;'.format(save_tag,save_path))


    # create_kitti_infos(
    #     dataset_cfg=dataset_cfg,
    #     class_names=['Car', 'Pedestrian', 'Cyclist'],
    #     data_path=data_path,
    #     save_path=save_path,
    #     workers=6,
    #     test=True
    # )


    # visualize_kitti_feats(dataset_cfg,class_names=['Car', 'Pedestrian', 'Cyclist'],data_path=data_path)


    # sample_img_feats(dataset_cfg,class_names=['Car', 'Pedestrian', 'Cyclist'],data_path=data_path, save_path=save_path)


    # create_kitti_dbinfos(dataset_cfg,class_names=['Car', 'Pedestrian', 'Cyclist'],data_path=data_path, save_path=save_path)


    # TEST_training_data(dataset_cfg=dataset_cfg,class_names=['Car'],data_path=data_path)


    # generate_reduced_pts(dataset_cfg=dataset_cfg,class_names=['Car'],data_path=data_path)


