import numpy as np
import scipy.io as sio

'''# MaData = sio.loadmat('./HomoDataSet/MaHomoMutiStrutData.mat')
DataFile = sio.loadmat("C:\\Users\long\Documents\GitHub\homogenization\\50000_Homo_Save\InputOutput_49575.mat")
Input = DataFile['MaDataAll']
Input = Input[0:,:]
np.save('Input',Input)

#Output = sio.loadmat('./HomoDataSet/CellData.mat')
# Output = sio.loadmat('C:\\Users\long\Documents\GitHub\IH-GAN_CMAME_2022\HomoDataSet/CellData_ParaInput_Cellular_multi_strut_TPMS.mat')
Output = DataFile['StrucDataAll']
Output = Output[0:,:]
np.save('Output',Output)'''



# DataFile = sio.loadmat("C:\\Users\long\Documents\GitHub\homogenization\\50000_Homo_Save\Reg_IO_Data.mat")
# Input = DataFile['RegHomo']
# Input = Input[0:,:]
# np.save('Input_49917',Input)
#
# Output = DataFile['RegStruc']
# Output = Output[0:,:]
# np.save('Output_49917',Output)

# DataFile = sio.loadmat("D:\CodeSave\GitCode\\topodiff\scripts\data\TestData300.mat")
# Input = DataFile['TestInput']
# Input = Input[0:,:]
# np.save('Input_300',Input)
#
# Output = DataFile['TestOutput']
# Output = Output[0:,:]
# np.save('Output_300',Output)




DataFile = sio.loadmat("C:\\Users\long\Documents\GitHub\homogenization\\50000_Homo_Save\\MainDiff_IO_Data.mat")
Input = DataFile['MainDiffHomo']
Input = Input[0:,:]
np.save('Input_37437',Input)

Output = DataFile['MainDiffStruc']
Output = Output[0:,:]
np.save('Output_37437',Output)