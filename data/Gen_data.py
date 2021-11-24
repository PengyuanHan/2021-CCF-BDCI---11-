import numpy as np
from numpy.lib.format import open_memmap

parts = {
    'joint', 'bone'
}

paris = {
    'ntu/xview': (
        (1, 8), (0, 1), (15, 0), (17, 15), (16, 0),
        (18, 16), (5, 1), (6, 5), (7, 6), (2, 1), (3, 2),
        (4, 3), (9, 8), (10, 9), (11, 10), (24, 11),
        (22, 11), (23, 22), (12, 8), (13, 12), (14, 13),
        (21, 14), (19, 14), (20, 19)
    )
}


from tqdm import tqdm

def gen_bone_data(mode):
    data = np.load('{}_data_joint.npy'.format(mode))
    N, C, T, V, M = data.shape
    fp_sp = open_memmap(
        '{}_data_bone.npy'.format(mode),
        dtype='float32',
        mode='w+',
        shape=(N, 3, T, V, M))

    fp_sp[:, :C, :, :, :] = data
    for v1, v2 in tqdm(paris['ntu/xview']):
        v1 -= 1
        v2 -= 1
        fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]
        
def gen_motion_data(set):
    for part in parts:
        data = np.load('{}_data_{}.npy'.format(set, part))
        N, C, T, V, M = data.shape
        fp_sp = open_memmap(
            '{}_data_{}_motion.npy'.format(set, part),
            dtype='float32',
            mode='w+',
            shape=(N, 3, T, V, M))
        for t in tqdm(range(T - 1)):
            fp_sp[:, :, t, :, :] = data[:, :, t + 1, :, :] - data[:, :, t, :, :]
        fp_sp[:, :, T - 1, :, :] = 0

if __name__ == '__main__':
    gen_bone_data('train')
    gen_motion_data('train')
    gen_bone_data('testA')
    gen_motion_data('testA')
    gen_bone_data('testB')
    gen_motion_data('testB')