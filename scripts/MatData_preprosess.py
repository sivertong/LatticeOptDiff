import numpy as np
import scipy.io as sio

# MaData = sio.loadmat('./HomoDataSet/MaHomoMutiStrutData.mat')
MaData = sio.loadmat('C:\\Users\long\Documents\GitHub\homogenization\CellData\HomoCellData_multi_TPMS_Strut_TestSet200_OnlyPara.mat')
Test = MaData['MaData']
Test = Test[0:200,:]
np.save('Test',Test)

# MaData = sio.loadmat('./HomoDataSet/MaHomoMutiStrutData.mat')
MaData = sio.loadmat('C:\\Users\long\Documents\GitHub\IH-GAN_CMAME_2022\HomoDataSet/multi_TPMS_strut_homo_data_whole1000_onlypara.mat')
Input = MaData['MaData']
Input = Input[0:1000,:]
np.save('Input',Input)

#Output = sio.loadmat('./HomoDataSet/CellData.mat')
Output = sio.loadmat('C:\\Users\long\Documents\GitHub\IH-GAN_CMAME_2022\HomoDataSet/CellData_ParaInput_Cellular_multi_strut_TPMS.mat')
Output = Output['XDataSave']
Output = Output[0:1000,:]
np.save('Output',Output)

# a1 = np.linspace(0.01,1,100).reshape(-1,1)
# a2 = 1-a1;
# a3 = np.zeros(100)
# t1 = np.zeros(100)-0.5
# t2 = np.zeros(100)-0.5
# t3 = np.zeros(100)-0.5
# a2 = a2.reshape(-1,1)
# t3 = t3.reshape(-1,1)
# a3 = a3.reshape(-1,1)
# t1 = t1.reshape(-1,1)
# t2 = t2.reshape(-1,1)
# StrData = np.concatenate((a1,a2,a3,t1,t2,t3),axis=1)
# np.save('Output',StrData)



