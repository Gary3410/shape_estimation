import numpy as np
import matplotlib.pyplot as plt
def show_3D_single(data, id, cls_id):
    # data[:, :3] = pc_normalize(data[:, :3])
    data = data[data[:, 3] == cls_id]
    print(data.shape)
    colormap = []
    lab = np.asarray([[184, 179, 168],
                      [255, 0, 0],
                      [255, 127, 0],
                      [255, 255, 0],
                      [0, 255, 0],
                      [0, 0, 255],
                      [38, 0, 51],
                      [148, 0, 211]]) / 255.0
    colormap = [[] for _ in range(data.shape[0])]
    for i in range(data.shape[0]):
        colormap[i] = lab[int(data[i, 3])]
    # plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection='3d')
    # 设置视角
    # ax.view_init(elev=30, azim=-60)
    # 关闭坐标轴
    # plt.axis('off')
    # 设置坐标轴范围
    #ax.set_zlim3d(0, 0.1)
    #ax.set_ylim3d(-0.25, 0.35)
    #ax.set_xlim3d(-0.75, 0.35)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colormap, s=20, marker='.')  # , cmap='plasma')
    #plt.savefig('/home/zhenyu/code_test_2/point_save/' + '%d.png' % id, dpi=500, bbox_inches='tight', transparent=True)
    plt.close()
    #plt.show()