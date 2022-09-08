# @Time    : 12/05/2021
# @Author  : Wei Chen
# @Project : Pycharm

import cv2
import numpy as np
import os
import pickle
from uti_tool import getFiles_cate, depth_2_mesh_all, depth_2_mesh_bbx
from prepare_data.renderer import create_renderer


def render_pre(model_path):
    renderer = create_renderer(640, 480, renderer_type='python')
    models = getFiles_ab_cate(model_path, 'ply')  # model name example: laptop_air_1_norm.ply please adjust the
    # corresponding functions according to the model name.
    objs = []
    for model in models:
        obj = model.split('.')[0].split('/')[-1]
        print(obj)
        objs.append(obj)
        renderer.add_object(obj, model)
    print(objs)
    return renderer


def getFiles_ab_cate(file_dir, suf):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if file.split('.')[1] == suf:
                L.append(os.path.join(root, file))
    return L


def get_dis_all(pc, dep, dd=15):
    N = pc.shape[0]
    M = dep.shape[0]
    depp = np.tile(dep, (1, N))

    depmm = depp.reshape((M, N, 3))
    delta = depmm - pc
    diss = np.linalg.norm(delta, 2, 2)

    aa = np.min(diss, 1)
    bb = aa.reshape((M, 1))

    ids, cc = np.where(bb[:] < dd)

    return ids


def get_one(depth, bbx, vispt, seg, K, idx, objid, bp):
    ##save_path = bp + '%s/points' % (objid)
    ##save_pathlab = bp + '%s/points_labs' % (objid)
    save_path = bp + 'points'
    save_pathlab = bp + 'points_labs'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(save_pathlab):
        os.makedirs(save_pathlab)

    ##VIS = depth_2_mesh_all(vispt, K)
    VIS = depth_2_mesh_bbx(vispt, [0, 479, 0, 639], K)
    #VIS = depth_2_mesh_bbx(vispt, [0, 639, 0, 479], K)######
    VIS = VIS[np.where(VIS[:, 2] > 0.0)] * 1000.0

    numbs = 6000

    numbs2 = 1000
    if VIS.shape[0] > numbs2:
        choice2 = np.random.choice(VIS.shape[0], numbs2, replace=False)
        VIS = VIS[choice2, :]

    filename = save_path + ("/pose%08d.txt" % (idx))
    w_namei = save_pathlab + ("/lab%08d.txt" % (idx))

    dep3d_ = depth_2_mesh_bbx(depth, bbx, K, enl=0)
    ids = seg[bbx[0]:bbx[1]+1, bbx[2]:bbx[3]+1].copy().reshape((bbx[1]-bbx[0]+1)*(bbx[3]-bbx[2]+1))/255

    if dep3d_.shape[0] > numbs:
        choice = np.random.choice(dep3d_.shape[0], numbs, replace=False)
        dep3d = dep3d_[choice, :]
        ids = ids[choice]
    else:
        choice = np.random.choice(dep3d_.shape[0], numbs, replace=True)
        dep3d = dep3d_[choice, :]
        ids = ids[choice]

    dep3d = dep3d[np.where(dep3d[:, 2] != 0.0)]

    threshold = 1200

    #ids = get_dis_all(VIS, dep3d[:, 0:3], dd=threshold)  ## find the object points

    ##if len(ids) <= 10:
    if 0:
        if os.path.exists(filename):
            os.remove(filename)
        if os.path.exists(w_namei):
            os.remove(w_namei)

    ##if len(ids) > 10:
    if 1:
        np.savetxt(filename, dep3d, fmt='%f', delimiter=' ')
        ##lab = np.zeros((dep3d.shape[0], 1), dtype=np.uint)
        ##lab[ids, :] = 1
        np.savetxt(w_namei, ids, fmt='%d')
        print(idx, len(ids))


def get_point_wise_lab(basepath, fold, renderer, sp):
    base_path = basepath + '%d/' % (fold)

    depths = getFiles_cate(base_path, '_depth.png', 4, -4)

    labels = getFiles_cate(base_path, '_label2', 4, -4)
    depths = []
    labels = []
    segs = []
    for i in os.listdir(base_path):
        if 'depth' in i:
            depths.append(base_path + i)
        if 'label' in i:
            labels.append(base_path + i)
        if 'seg' in i:
            segs.append(base_path + i)
    depths.sort()
    labels.sort()
    segs.sort()
    
    L_dep = depths

    Ki = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])

    Lidx = 1000
    if fold == 1:
        s = 0
    else:
        s = 0
    for i in range(s, len(L_dep)):

        lab = pickle.load(open(labels[i], 'rb'))

        depth = cv2.imread(L_dep[i], -1)
        seg_image = cv2.imread(segs[i], -1)
        img_id = int(L_dep[i][-14:-10])
        ##for ii in range(len(lab['class_ids'])):

        # if ii==0:
        #     ii=4
        ##obj = lab['model_list']
        obj = 1
        print(110, lab)

        ##seg = lab['bboxes'].reshape((1, 4))  ## y1 x1 y2 x2  (ori x1,y1,w h)
        seg = lab['bboxes']  ## y1 x1 y2 x2  (ori x1,y1,w h)

        idx = (fold - 1) * Lidx + img_id

        R = lab['rotations']  # .reshape((3, 3))
        # s = np.linalg.det(R)

        T = lab['translations'].reshape((3, 1))  # -np.array([0,0,100]).reshape((3, 1))

        if T[2] < 0:
            T[2] = -T[2]
        ##vis = renderer.render_object(obj, R, T, Ki[0, 0], Ki[1, 1], Ki[0, 2], Ki[1, 2])
        ##vis_rgb = vis['rgb']
        ##cv2.imwrite(sp + '%s_rgb.png' % idx, vis_rgb)
        ##vis_part = vis['depth']

        ##bbx = [seg[0, 0], seg[0, 2], seg[0, 1], seg[0, 3]]
        bbx = [seg[0], seg[2], seg[1], seg[3]]
        #bbx = [seg[1], seg[3], seg[0], seg[2]]######

        ##if vis_part.max() > 0:
        ##    get_one(depth, bbx, vis_part, Ki, idx, obj, sp)
        get_one(depth, bbx, depth, seg_image, Ki, idx, obj, sp)


if __name__ == '__main__':
    path = '/home/lcl/位姿估计/dataset/code/gt_labels/cracker_box/'
    render = render_pre(path)
    base = "/home/lcl/位姿估计/dataset/code/gt_labels/cracker_box/"
    fold = 1
    save_path = base
    get_point_wise_lab(base, fold, render, save_path)
