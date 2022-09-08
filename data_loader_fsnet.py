# @Time    : 25/09/2020 18:02
# @Author  : Wei Chen
# @Project : Pycharm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import _pickle as pickle
from uti_tool import *
import random
import trimesh
import open3d as o3d

def getFiles(file_dir,suf):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        #print('root: ',dirs)
        for file in files:
            if os.path.splitext(file)[1] == suf:
                L.append(os.path.join(root, file))
        L.sort(key=lambda x:int(x[-11:-4]))
    return L

def getDirs(file_dir):
    L=[]

    dirs = os.listdir(file_dir)

    return dirs


def load_depth(depth_path):
    """ Load depth image from img_path. """

    depth = cv2.imread(depth_path, -1)
    if len(depth.shape) == 3:
        # This is encoded depth image, let's convert
        # NOTE: RGB is actually BGR in opencv
        depth16 = depth[:, :, 1]*256 + depth[:, :, 2]
        depth16 = np.where(depth16==32001, 0, depth16)
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'
    return depth16


def chooselimt(pts0, lab, zmin, zmax):


    pts = pts0.copy()
    labs = lab.copy()

    pts1=pts[np.where(pts[:,2]<zmax)[0],:]
    lab1 = labs[np.where(pts[:,2]<zmax)[0], :]

    ptsn = pts1[np.where(pts1[:, 2] > zmin)[0], :]
    labs = lab1[np.where(pts1[:, 2] > zmin)[0],:]

    return ptsn,labs

def circle_iou(pts,lab, dia):
    # fx = K[0, 0]
    # ux = K[0, 2]
    # fy = K[1, 1]
    # uy = K[1, 2]
    a = pts[lab[:, 0] == 1, :]
    ptss = pts[lab[:, 0] == 1, :]
    idx = np.random.randint(0, a.shape[0])

    zmin = max(0,ptss[idx,2]-dia)
    zmax = ptss[idx,2]+dia

    return zmin, zmax


class CateDataset(Dataset):
    def __init__(self, root_dir, K, cate,lim=1,transform=None,corners=0, temp=None):

        cats = ["banana", "bowl", "box", "can", "cup", "marker", "pear", "sugar"]

        objs = cats
        self.objs_name = objs
        self.objs = np.zeros((len(objs), 1),dtype=np.uint) # 判断是否有每个类别

        for i in range(len(objs)):
            for cat in cate:
                if cat in objs[i]:
                    self.objs[i]=1

        # self.cate_id = np.where(np.array(cats) == cate)[0][0]+1
        self.ids = np.where(self.objs==1)

        self.root_dir = root_dir
        self.lim = lim
        self.transform = transform
        self.cate = cate
        self.K = K
        self.corners = corners
        self.rad = temp

        # self.root = root_dir
        self.c = 0
        self.count = 0
        # self.model_path = os.path.join(root_dir, model_path)

    def __len__(self):


        return  1500##1500


    def __getitem__(self, index):

        # c随机选择的类别名称id
        c = np.random.randint(0,len(self.ids[0]))

        # 获取类别名称 cate
        obj_id = self.ids[0][c]
        cate = self.objs_name[obj_id]

        #pc = load_ply(self.root_dir+'/%s/%s.ply'%(cate,cate))['pts']*1000.0
        # 改用我们自己的pc
        mesh = trimesh.load_mesh(self.root_dir+'/%s/%s.ply'%(cate,cate))
        vpc = mesh.vertices
        pc = np.asarray(vpc).copy()
        pc = pc * 1000.0

        root_dir = self.root_dir + '/%s/' % (cate)
        # print(root_dir)
        pts_ps = getFiles_ab(root_dir+'points/','.txt', -12, -4)  #should be '/home/lcl/object-deformnet-master/data/laptop/points/XXX.txt'

        # pts_ps 是该类别下, 所有点云的路径
        # idx是随机抽取一个
        idx = random.randint(0, len(pts_ps) - 1)
        pts_name = pts_ps[idx]
        lab_name = getFiles_ab(root_dir+'points_labs/','.txt', -12, -4)[idx]

        scene_id = int(pts_name[-12:-4])//1000+1 ## you can change according to your own name rules

        img_id = int(pts_name[-12:-4])-(scene_id-1)*1000

        depth_p  = root_dir + '%d'%(scene_id) + '/%d_depth.png'%(img_id)
        label_p = root_dir + '%d'%(scene_id) + '/%d_label.pkl'%(img_id)       #04d to 4d
        seg_p = root_dir + '%d' % (scene_id) + '/%d_seg.png' % (img_id)
        seg_gt = cv2.imread(seg_p, -1)  # gt mask
        #y, x = np.where(seg_gt == 255)

        # print(depth_p,label_p,cate)

        gts = pickle.load(open(label_p, 'rb'))
        # idin = np.where(np.array(gts['model_list']) == 'laptop_air_xin_norm')
        # for id_index,obj_name in enumerate(gts['model_list']):
        #     if(obj_name.find(self.cate)!=-1):
        #         idin=(id_index,)
        #
        #
        # if len(idin)==0: ## fix some wrong cases
        #     bbx = np.array([1,2,3,4]).reshape((1, 4))
        #     R = np.eye(3)
        #     T = np.array([0,0,0]).reshape(1,3)
        # else:
        bbx = gts['bboxes'].reshape((1, 4)) ## y1 x1 y2 x2
        R = gts['rotations'].reshape(3,3)
        T = gts['translations'].reshape(1,3)*1000.0
        scale = gts['scales']
        self.pc = pc
        self.R = R
        self.T = T
        depth = cv2.imread(depth_p,-1)

        # dep_seg = np.zeros(depth.shape, dtype=np.uint16)          # seg depth from 2D mask
        # dep_seg[y, x] = depth[y, x]
        # y1, x1 = np.where(dep_seg < 650)
        # depth1 = np.zeros(depth.shape, dtype=np.uint16)
        # depth1[y1, x1] = dep_seg[y1, x1]
        # depth = depth1

        label = np.loadtxt(lab_name)

        label_ = label.reshape((-1, 1))
        points_ = np.loadtxt(pts_name)

        #points_, label_,sx,sy,sz = self.aug_pts_labs(depth,points_,label_,bbx,cate)

        #Scale = np.array([sx,sy,sz])
        Scale = scale

        if  points_.shape[0]!=label_.shape[0]:
            print(self.root_dir[idx])

        choice = np.random.choice(len(points_), 2000, replace=True)
        points = points_[choice, :]
        label = label_[choice, :]

        choice = np.random.choice(len(points_), 2500, replace=True)
        points_1 = points_[choice, :]
        label_1 = label_[choice, :]


        cats = ["banana", "bowl", "box", "can", "cup", "marker", "pear", "sugar"]
        self.cate_id = np.where(np.array(cats) == cate)[0][0]+1

        # 模板降采样为500
        pc = pc[np.random.choice(len(pc), 500, replace=True),:]

        sample = {'points': points, 'label': label, 'R':R, 'T':T,'cate_id':self.cate_id,'scale':Scale,'dep':depth_p,'model':pc,
                  'ptsori': points_1}

        return sample

    def aug_pts_labs(self, depth, pts, labs, bbx, cate):

        ## 2D bounding box augmentation and fast relabeling
        bbx_gt = [bbx[0, 1], bbx[0, 3], bbx[0, 0], bbx[0, 2]]  # x1,x2, y1 , y2
        bbx = shake_bbx(bbx_gt)  ## x1,x2,y1,y2
        depth, bbx_iou = depth_out_iou(depth, bbx, bbx_gt)

        mesh = depth_2_mesh_bbx(depth, [bbx[2], bbx[3], bbx[0], bbx[1]], self.K)
        mesh = mesh[np.where(mesh[:, 2] > 0.0)]
        mesh = mesh[np.where(mesh[:, 2] < 5000.0)]

        if len(mesh) > 1000:
            choice = np.random.choice(len(mesh), len(mesh) // 2, replace=True)
            mesh = mesh[choice, :]

        pts_a, labs_a = pts_iou(pts.copy(), labs.copy(), self.K, bbx_iou)

        assert pts_a.shape[0] == labs_a.shape[0]

        if len(pts_a[labs_a[:, 0] == 1, :]) < 50:  ## too few points in intersection region
            pts_ = pts_a.copy()
            labs_ = labs_a.copy()
        else:
            pts_ = pts.copy()
            labs_ = labs.copy()

        N = pts_.shape[0]
        M = mesh.shape[0]
        mesh = np.concatenate([mesh, pts_], axis=0)
        label = np.zeros((M + N, 1), dtype=np.uint)
        label[M:M + N, 0] = labs_[:, 0]
        points = mesh

        if self.lim == 1:
            zmin, zmax = circle_iou(points.copy(), label.copy(), self.rad)
            points, label = chooselimt(points, label, zmin, zmax)

        ### 3D deformation
        Rt = get_rotation(180, 0, 0)
        # self.pc = np.dot(Rt, self.pc.T).T ## the object 3D model is up-side-down along the X axis in our case, you may not need this code to reverse

        s = 0.8
        e = 1.2
        pointsn, ex, ey, ez, s = defor_3D(points, label, self.R, self.T, self.pc, scalex=(s, e), scalez=(s, e),
                                          scaley=(s, e), scale=(s, e), cate=cate)
        sx, sy, sz = var_2_norm(self.pc, ex, ey, ez, c=cate)
        return pointsn, label.astype(np.uint8), sx, sy, sz


def load_pts_train_cate(data_path ,bat,K,cate,lim=1,rad=400,shuf=True,drop=False,corners=0,nw=0):

    data=CateDataset(data_path, K, cate,lim=lim,transform=None,corners=corners, temp=rad)

    dataloader = DataLoader(data, batch_size=bat, shuffle=shuf, drop_last=drop,num_workers=nw)

    return dataloader










