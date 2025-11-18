import os
import torch
import numpy as np
import MinkowskiEngine as ME  # ← 添加这个导入！
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from utils.transforms import make_transforms_clouds
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud

# parametrizing set, to try out different parameters
CUSTOM_SPLIT = [
    "scene-0008", "scene-0009", "scene-0019", "scene-0029", "scene-0032", "scene-0042",
    "scene-0045", "scene-0049", "scene-0052", "scene-0054", "scene-0056", "scene-0066",
    "scene-0067", "scene-0073", "scene-0131", "scene-0152", "scene-0166", "scene-0168",
    "scene-0183", "scene-0190", "scene-0194", "scene-0208", "scene-0210", "scene-0211",
    "scene-0241", "scene-0243", "scene-0248", "scene-0259", "scene-0260", "scene-0261",
    "scene-0287", "scene-0292", "scene-0297", "scene-0305", "scene-0306", "scene-0350",
    "scene-0352", "scene-0358", "scene-0361", "scene-0365", "scene-0368", "scene-0377",
    "scene-0388", "scene-0391", "scene-0395", "scene-0413", "scene-0427", "scene-0428",
    "scene-0438", "scene-0444", "scene-0452", "scene-0453", "scene-0459", "scene-0463",
    "scene-0464", "scene-0475", "scene-0513", "scene-0533", "scene-0544", "scene-0575",
    "scene-0587", "scene-0589", "scene-0642", "scene-0652", "scene-0658", "scene-0669",
    "scene-0678", "scene-0687", "scene-0701", "scene-0703", "scene-0706", "scene-0710",
    "scene-0715", "scene-0726", "scene-0735", "scene-0740", "scene-0758", "scene-0786",
    "scene-0790", "scene-0804", "scene-0806", "scene-0847", "scene-0856", "scene-0868",
    "scene-0882", "scene-0897", "scene-0899", "scene-0976", "scene-0996", "scene-1012",
    "scene-1015", "scene-1016", "scene-1018", "scene-1020", "scene-1024", "scene-1044",
    "scene-1058", "scene-1094", "scene-1098", "scene-1107",
]


def custom_collate_fn(list_data):
    """
    Custom collate function adapted for creating batches with MinkowskiEngine.
    """
    input = list(zip(*list_data))
    labelized = len(input) == 7
    
    if labelized:
        xyz, coords, feats, labels, evaluation_labels, inverse_indexes, lidar_name = input
    else:
        xyz, coords, feats, inverse_indexes = input

    coords_batch, len_batch = [], []

    for batch_id, coo in enumerate(coords):
        N = coords[batch_id].shape[0]
        coords_batch.append(
            torch.cat((coo, torch.ones(N, 1, dtype=torch.int32) * batch_id), 1)
        )
        len_batch.append(N)

    coords_batch = torch.cat(coords_batch, 0).int()
    feats_batch = torch.cat(feats, 0).float()
    
    if labelized:
        labels_batch = torch.cat(labels, 0).long()
        return {
            "pc": xyz,
            "sinput_C": coords_batch,
            "sinput_F": feats_batch,
            "len_batch": len_batch,
            "labels": labels_batch,
            "evaluation_labels": evaluation_labels,
            "inverse_indexes": inverse_indexes,
            "lidar_name": lidar_name
        }
    else:
        return {
            "pc": xyz,
            "sinput_C": coords_batch,
            "sinput_F": feats_batch,
            "len_batch": len_batch,
            "inverse_indexes": inverse_indexes,
        }


class NuScenesDataset(Dataset):
    """
    Dataset returning a lidar scene and associated labels.
    """

    def __init__(self, phase, config, transforms=None, cached_nuscenes=None):
        self.phase = phase
        self.labels = self.phase != "test"
        self.transforms = transforms
        self.voxel_size = config["voxel_size"]
        self.cylinder = config["cylindrical_coordinates"]
        
        # 从配置中读取数据根目录
        self.dataroot = config.get("dataRoot_nuscenes", "/data/zy/OpenPCDet/data/nuscenes/v1.0-trainval")

        # 加载 NuScenes 数据集
        if phase != "test":
            if cached_nuscenes is not None:
                self.nusc = cached_nuscenes
            else:
                self.nusc = NuScenes(
                    version="v1.0-trainval", 
                    dataroot=self.dataroot, 
                    verbose=False
                )
        else:
            self.nusc = NuScenes(
                version="v1.0-test", 
                dataroot=self.dataroot, 
                verbose=False
            )

        # 确定场景划分（和预训练完全一样）
        if phase in ("train", "val", "test"):
            phase_scenes = create_splits_scenes()[phase]
        elif phase == "parametrizing":
            phase_scenes = list(
                set(create_splits_scenes()["train"]) - set(CUSTOM_SPLIT)
            )
        elif phase == "verifying":
            phase_scenes = CUSTOM_SPLIT
        else:
            raise ValueError(f"Unknown phase: {phase}")

        # Skip ratio
        if phase in ("val", "verifying"):
            skip_ratio = 1
        else:
            skip_ratio = config.get("dataset_skip_step", 1)

        # 构建数据列表（和预训练完全一样的逻辑）
        self.list_keyframes = []
        skip_counter = 0
        
        for scene_idx in range(len(self.nusc.scene)):
            scene = self.nusc.scene[scene_idx]
            if scene["name"] in phase_scenes:
                skip_counter += 1
                if skip_counter % skip_ratio == 0:
                    self.create_list_of_scans(scene)

        print(f"{phase}: {len(self.list_keyframes)} keyframes")

        # labels' names lookup table
        self.eval_labels = {
            0: 0, 1: 0, 2: 7, 3: 7, 4: 7, 5: 0, 6: 7, 7: 0, 8: 0, 9: 1, 10: 0, 11: 0,
            12: 8, 13: 0, 14: 2, 15: 3, 16: 3, 17: 4, 18: 5, 19: 0, 20: 0, 21: 6, 22: 9,
            23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 0, 30: 16, 31: 0,
        }

    def create_list_of_scans(self, scene):
        #从预训练代码迁移过来的方法
        current_sample_token = scene["first_sample_token"]
        
        while current_sample_token != "":
            current_sample = self.nusc.get("sample", current_sample_token)
            self.list_keyframes.append(current_sample["data"])
            current_sample_token = current_sample["next"]

    def __len__(self):
        return len(self.list_keyframes)

    def __getitem__(self, idx):
        # 获取数据字典（包含所有传感器的 token）
        data_dict = self.list_keyframes[idx]
        lidar_token = data_dict["LIDAR_TOP"]
        
        # 获取点云数据
        pointsensor = self.nusc.get("sample_data", lidar_token)
        pcl_path = os.path.join(self.nusc.dataroot, pointsensor["filename"])
        points = LidarPointCloud.from_file(pcl_path).points.T
        pc = points[:, :3]
        
        # 获取标签
        if self.labels:
            lidarseg_data = self.nusc.get("lidarseg", lidar_token)
            lidarseg_labels_filename = os.path.join(
                self.nusc.dataroot, lidarseg_data["filename"]
            )
            points_labels = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)

        pc = torch.tensor(pc)

        # 数据增强
        if self.transforms:
            pc = self.transforms(pc)

        # 体素化
        if self.cylinder:
            x, y, z = pc.T
            rho = torch.sqrt(x ** 2 + y ** 2) / self.voxel_size
            phi = torch.atan2(y, x) * 180 / np.pi
            z = z / self.voxel_size
            coords_aug = torch.cat((rho[:, None], phi[:, None], z[:, None]), 1)
        else:
            coords_aug = pc / self.voxel_size

        # 稀疏量化（修复了 bug：使用 coords_aug 而不是 coords）
        discrete_coords, indexes, inverse_indexes = ME.utils.sparse_quantize(
            coords_aug.contiguous(), return_index=True, return_inverse=True
        )

        # 提取唯一体素的特征（强度值）
        # 关键修复：确保特征维度正确
        unique_feats = points[indexes][:, 3:]  # 提取强度特征（第4列开始）
        
        # 确保特征是 2D 张量 [N, C]，即使只有一个通道也要保持形状
        if unique_feats.ndim == 1:
            unique_feats = unique_feats[:, None]  # [N] -> [N, 1]
        unique_feats = torch.from_numpy(unique_feats).float()

        if self.labels:
            points_labels = torch.tensor(
                np.vectorize(self.eval_labels.__getitem__)(points_labels),
                dtype=torch.int32,
            )
            unique_labels = points_labels[indexes]
            
            # 使用实际的文件名
            lidar_name = pointsensor["filename"]
            
            return (
                pc,
                discrete_coords,
                unique_feats,
                unique_labels,
                points_labels,
                inverse_indexes,
                lidar_name,
            )
        else:
            return pc, discrete_coords, unique_feats, inverse_indexes


def make_data_loader(config, phase, num_threads=0):
    """
    Create the data loader for a given phase and a number of threads.
    """
    if phase == "train":
        transforms = make_transforms_clouds(config)
    else:
        transforms = None

    dset = NuScenesDataset(phase=phase, transforms=transforms, config=config)
    collate_fn = custom_collate_fn
    batch_size = config["batch_size"] // config["num_gpus"]

    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=phase == "train",
        num_workers=num_threads,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=phase == "train",
        worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
    )
    return loader
