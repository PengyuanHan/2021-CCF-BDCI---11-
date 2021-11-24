import csv
import numpy as np

headers = ['sample_index', 'predict_category']

motion_joint = open('submissionB_stgcn_joint_motion.csv')
motion_bone = open('submissionB_stgcn_bone_motion.csv')
joint = open('submissionB_stgcn_joint.csv')
bone = open('submissionB_stgcn_bone.csv')

motion_joint_1 = open('submissionB_joint_motion.csv')
motion_bone_1 = open('submissionB_bone_motion.csv')
joint_1 = open('submissionB_joint.csv')
bone_1 = open('submissionB_bone.csv')

"""REFINEDSTGCN"""
motion_joint_reader = csv.reader(motion_joint)
motion_joint_rows = [row for row in motion_joint_reader]
motion_bone_reader = csv.reader(motion_bone)
motion_bone_rows = [row for row in motion_bone_reader]
joint_reader = csv.reader(joint)
joint_rows = [row for row in joint_reader]
bone_reader = csv.reader(bone)
bone_rows = [row for row in bone_reader]

"""REFINEDAGCN"""
motion_joint_reader_1 = csv.reader(motion_joint_1)
motion_joint_rows_1 = [row for row in motion_joint_reader_1]
motion_bone_reader_1 = csv.reader(motion_bone_1)
motion_bone_rows_1 = [row for row in motion_bone_reader_1]
joint_reader_1 = csv.reader(joint_1)
joint_rows_1 = [row for row in joint_reader_1]
bone_reader_1 = csv.reader(bone_1)
bone_rows_1 = [row for row in bone_reader_1]

"""CTRGCN"""
agcn_concat = open("submissionB_ctrgcn.csv")
agcn_reader = csv.reader(agcn_concat)
agcn_rows = [row for row in agcn_reader]

"""CTRGCN"""
stgcn_concat = open("submissionB_ctrgcn.csv")
stgcn_reader = csv.reader(stgcn_concat)
stgcn_rows = [row for row in stgcn_reader]

values = []
i = 0

for i in range(len(bone_rows)):
    if i >= 1:
        "STGCN"
        score_add_joint = []
        score_add_bone = []
        score_add_bone_motion = []
        score_add_joint_motion = []

        id, clas, score_joint = joint_rows[i]
        id_, clas_, score_bone = bone_rows[i]
        idm, clasm, score_jointm = motion_joint_rows[i]
        id_m, clas_m, score_bonem = motion_bone_rows[i]

        scores_joint = score_joint.strip().split(' ')
        scores_bone = score_bone.strip().split(' ')
        scores_jointm = score_jointm.strip().split(' ')
        scores_bonem = score_bonem.strip().split(' ')

        "AGCN"
        score_add_joint_1 = []
        score_add_bone_1 = []
        score_add_bone_motion_1 = []
        score_add_joint_motion_1 = []

        id_1, clas_1, score_joint_1 = joint_rows_1[i]
        id__1, clas__1, score_bone_1 = bone_rows_1[i]
        idm_1, clasm_1, score_jointm_1 = motion_joint_rows_1[i]
        id_m_1, clas_m_1, score_bonem_1 = motion_bone_rows_1[i]

        scores_joint_1 = score_joint_1.strip().split(' ')
        scores_bone_1 = score_bone_1.strip().split(' ')
        scores_jointm_1 = score_jointm_1.strip().split(' ')
        scores_bonem_1 = score_bonem_1.strip().split(' ')

        "STGCN_concat"
        score_stgcn = []
        id_s, clas_s, score_joint_s = stgcn_rows[i]
        scores_joint_s = score_joint_s.strip().split(' ')

        "AGCN_concat"
        score_agcn = []
        id_a, clas_a, score_joint_a = stgcn_rows[i]
        scores_joint_a = score_joint_a.strip().split(' ')

        for score in scores_joint:
            flag = 0
            try:
                score_add_joint.append(float(score.strip().split('[')[1]))
                flag += 1
            except:
                pass
            if flag == 0:
                try:
                    score_add_joint.append(float(score.strip().split('\n')[0]))
                    flag += 1
                except:
                    pass
            if flag == 0:
                try:
                    score_add_joint.append(float(score.strip().split(']')[0]))
                    flag += 1
                except:
                    pass
            if flag == 0:
                try:
                    score_add_joint.append(float(score))
                except :
                    pass

        for score in scores_bone:
            flag = 0
            try:
                score_add_bone.append(float(score.strip().split('[')[1]))
                flag += 1
            except:
                pass
            if flag == 0:
                try:
                    score_add_bone.append(float(score.strip().split('\n')[0]))
                    flag += 1
                except:
                    pass
            if flag == 0:
                try:
                    score_add_bone.append(float(score.strip().split(']')[0]))
                    flag += 1
                except:
                    pass
            if flag == 0:
                try:
                    score_add_bone.append(float(score))
                except :
                    pass

        for score in scores_jointm:
            flag = 0
            try:
                score_add_joint_motion.append(float(score.strip().split('[')[1]))
                flag += 1
            except:
                pass
            if flag == 0:
                try:
                    score_add_joint_motion.append(float(score.strip().split('\n')[0]))
                    flag += 1
                except:
                    pass
            if flag == 0:
                try:
                    score_add_joint_motion.append(float(score.strip().split(']')[0]))
                    flag += 1
                except:
                    pass
            if flag == 0:
                try:
                    score_add_joint_motion.append(float(score))
                except :
                    pass

        for score in scores_bonem:
            flag = 0
            try:
                score_add_bone_motion.append(float(score.strip().split('[')[1]))
                flag += 1
            except:
                pass
            if flag == 0:
                try:
                    score_add_bone_motion.append(float(score.strip().split('\n')[0]))
                    flag += 1
                except:
                    pass
            if flag == 0:
                try:
                    score_add_bone_motion.append(float(score.strip().split(']')[0]))
                    flag += 1
                except:
                    pass
            if flag == 0:
                try:
                    score_add_bone_motion.append(float(score))
                except :
                    pass
        """分割线 上STGCN 下AGCN"""

        for score in scores_joint_1:
            flag = 0
            try:
                score_add_joint_1.append(float(score.strip().split('[')[1]))
                flag += 1
            except:
                pass
            if flag == 0:
                try:
                    score_add_joint_1.append(float(score.strip().split('\n')[0]))
                    flag += 1
                except:
                    pass
            if flag == 0:
                try:
                    score_add_joint_1.append(float(score.strip().split(']')[0]))
                    flag += 1
                except:
                    pass
            if flag == 0:
                try:
                    score_add_joint_1.append(float(score))
                except :
                    pass

        for score in scores_bone_1:
            flag = 0
            try:
                score_add_bone_1.append(float(score.strip().split('[')[1]))
                flag += 1
            except:
                pass
            if flag == 0:
                try:
                    score_add_bone_1.append(float(score.strip().split('\n')[0]))
                    flag += 1
                except:
                    pass
            if flag == 0:
                try:
                    score_add_bone_1.append(float(score.strip().split(']')[0]))
                    flag += 1
                except:
                    pass
            if flag == 0:
                try:
                    score_add_bone_1.append(float(score))
                except :
                    pass

        for score in scores_jointm_1:
            flag = 0
            try:
                score_add_joint_motion_1.append(float(score.strip().split('[')[1]))
                flag += 1
            except:
                pass
            if flag == 0:
                try:
                    score_add_joint_motion_1.append(float(score.strip().split('\n')[0]))
                    flag += 1
                except:
                    pass
            if flag == 0:
                try:
                    score_add_joint_motion_1.append(float(score.strip().split(']')[0]))
                    flag += 1
                except:
                    pass
            if flag == 0:
                try:
                    score_add_joint_motion_1.append(float(score))
                except :
                    pass

        for score in scores_bonem_1:
            flag = 0
            try:
                score_add_bone_motion_1.append(float(score.strip().split('[')[1]))
                flag += 1
            except:
                pass
            if flag == 0:
                try:
                    score_add_bone_motion_1.append(float(score.strip().split('\n')[0]))
                    flag += 1
                except:
                    pass
            if flag == 0:
                try:
                    score_add_bone_motion_1.append(float(score.strip().split(']')[0]))
                    flag += 1
                except:
                    pass
            if flag == 0:
                try:
                    score_add_bone_motion_1.append(float(score))
                except :
                    pass


        for score in scores_joint_a:
            flag = 0
            try:
                score_agcn.append(float(score.strip().split('[')[1]))
                flag += 1
            except:
                pass
            if flag == 0:
                try:
                    score_agcn.append(float(score.strip().split('\n')[0]))
                    flag += 1
                except:
                    pass
            if flag == 0:
                try:
                    score_agcn.append(float(score.strip().split(']')[0]))
                    flag += 1
                except:
                    pass
            if flag == 0:
                try:
                    score_agcn.append(float(score))
                except :
                    pass

        for score in scores_joint_s:
            flag = 0
            try:
                score_stgcn.append(float(score.strip().split('[')[1]))
                flag += 1
            except:
                pass
            if flag == 0:
                try:
                    score_stgcn.append(float(score.strip().split('\n')[0]))
                    flag += 1
                except:
                    pass
            if flag == 0:
                try:
                    score_stgcn.append(float(score.strip().split(']')[0]))
                    flag += 1
                except:
                    pass
            if flag == 0:
                try:
                    score_stgcn.append(float(score))
                except :
                    pass

        #print(np.array(score_add_joint_motion_1).shape)
        #print(np.array(score_add_bone).shape)

        score_add = 0.24 * np.array(score_add_bone) \
                    + 0.38 * np.array(score_add_joint) \
                    + 0.1 * np.array(score_add_bone_motion) \
                    + 0.18 * np.array(score_add_joint_motion) \
                    + 0.24 * np.array(score_add_bone_1) \
                    + 0.38 * np.array(score_add_joint_1) \
                    + 0.1 * np.array(score_add_bone_motion) \
                    + 0.18 * np.array(score_add_joint_motion_1) \
                    + 0.34 * np.array(score_agcn)

        label = np.where(score_add == np.max(score_add))[0]
        label = int(label)
        print(label)
        values.append((i-1, label))

    i = i + 1

with open(
        'submissionB_Fianl.csv',
        'w',
        newline='',
) as fp:
    writer = csv.writer(fp)
    writer.writerow(headers)
    for value in values:
        writer.writerow(value)


