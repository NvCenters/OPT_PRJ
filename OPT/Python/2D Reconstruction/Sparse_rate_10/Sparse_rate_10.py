import tigre
import numpy as np
import os
from tigre.utilities import sample_loader
from tigre.utilities import CTnoise
import tigre.algorithms as algs
from PIL import Image
from PIL import ImageFile
import tigre.utilities.gpu as gpu

ImageFile.LOAD_TRUNCATED_IMAGES = True
listGpuNames = gpu.getGpuNames()
if len(listGpuNames) == 0:
    print("Error: No gpu found")
else:
    for id in range(len(listGpuNames)):
        print("{}: {}".format(id, listGpuNames[id]))

gpuids = gpu.getGpuIds(listGpuNames[0])
print(gpuids)


geo = tigre.geometry()
geo.DSD = 1000  
geo.DSO = 1000  
# Image parameters
geo.nVoxel = np.array([1024, 1024, 1024])  # number of voxels              (vx)
geo.sVoxel = np.array([1024, 1024, 1024])  # total size of the image       (mm)
geo.dVoxel = geo.sVoxel / geo.nVoxel  # size of each voxel            (mm)
# Detector parameters
geo.nDetector = np.array([1024, 1024])  # number of pixels              (px)
geo.dDetector = np.array([geo.dVoxel[0], 1])  # size of each pixel            (mm)
geo.sDetector = geo.nDetector * geo.dDetector  # total size of the detector    (mm)
# Offsets
geo.offOrigin = np.array([0, 0, 0])  # Offset of image from origin   (mm)
geo.offDetector = np.array([0, 0])  # Offset of Detector            (mm)

geo.mode = "parallel"

folder_path = 'TIF'

file_names = os.listdir(folder_path)
file_names = sorted(file_names)
image_data = np.empty((81, 1024, 1024), dtype=np.uint16)

for i, file_name in enumerate(file_names):
    file_path = os.path.join(folder_path, file_name)
    if file_name.lower().endswith('.tif') or file_name.lower().endswith('.tiff'):
        img = Image.open(file_path)
        image_data[:, i, :] = np.array(img).T
        
angles = np.linspace(0, 2 * np.pi, 81)
niter = 50
# use SART
imgOSSART = algs.ossart(image_data, geo, angles, niter, gpuids=gpuids)
# use MLEM
imgMLEM = algs.mlem(image_data, geo, angles, niter, gpuids=gpuids)

result_folders = ['OSSART', 'MLEM']
base_path = 'DATA_01'

for folder in result_folders:
    folder_path = os.path.join(base_path, folder)
    os.makedirs(folder_path, exist_ok=True)

results = [imgOSSART, imgMLEM]

for i, folder in enumerate(result_folders):
    folder_path = os.path.join(base_path, folder)
    for j in range(results[i].shape[0]):
        image_data = np.rot90(results[i][j, :, :],1)
        normalized_array = image_data / np.max(image_data)
        image_data = np.round(normalized_array * 255).astype(np.uint8)
        image_path = os.path.join(folder_path, f'{folder}{j}.tif')  
        image = Image.fromarray(image_data)  
        image.save(image_path) 
