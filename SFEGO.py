import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import math
import cv2
import sys
import os

# System PATH for VC++ Compiler (cl.exe)
if (os.system("cl.exe")):
    os.environ['PATH'] += ';'+r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.32.31326\bin\Hostx64\x64"
if (os.system("cl.exe")):
    raise RuntimeError("cl.exe still not found, path probably incorrect")

M_PI=3.14159265358979323846

def build_list(radius):
    ar_len=0
    x_list=[]
    y_list=[]
    deg_list=[]
    radius_list=[]
    for i in range(-radius,radius+1):
        for j in range(-radius,radius+1):
            if ((np.sqrt(i*i+j*j) < radius+1.0) and not (i==0 and j==0)):
                x_list.append(j)
                y_list.append(i)
                deg=math.atan2(j,i)
                if deg<0.0:
                    deg+=M_PI*2
                deg_list.append(deg)
                radius_list.append(np.sqrt(i*i+j*j))
    zipped=zip(x_list, y_list, deg_list, radius_list)
    zipped=sorted(zipped, key = lambda x: (x[2], x[3]))
    return zipped

#Initial PyCUDA
mod = SourceModule(open('kernel.cu').read())

def SFEGO(input_data, radius):
    #Setup Radius
    ar_list=build_list(radius)
    x_list, y_list, deg_list, radius_list=zip(*ar_list)
    list_len=len(ar_list)
    
    target_height = input_data.shape[0]
    target_width = input_data.shape[1]

    #Convert to Numpy
    np_data = np.asarray(input_data).flatten().astype(np.float32)
    np_x_list = np.asarray(x_list).astype(np.int32)
    np_y_list = np.asarray(y_list).astype(np.int32)
    np_deg_list = np.asarray(deg_list).astype(np.float32)

    #CUDA Buffer 
    list_x = cuda.mem_alloc(np_x_list.size * np_x_list.dtype.itemsize)
    cuda.memcpy_htod(list_x, np_x_list)
    list_y = cuda.mem_alloc(np_y_list.size * np_y_list.dtype.itemsize)
    cuda.memcpy_htod(list_y, np_y_list)
    list_deg = cuda.mem_alloc(np_deg_list.size * np_deg_list.dtype.itemsize)
    cuda.memcpy_htod(list_deg, np_deg_list)
    data = cuda.mem_alloc(np_data.nbytes)
    cuda.memcpy_htod(data, np_data)
    diff = cuda.mem_alloc(np_data.nbytes)
    direct = cuda.mem_alloc(np_data.nbytes)
    result = cuda.mem_alloc(np_data.nbytes)

    #Define CUDA Function
    knl_gradient_fnc = mod.get_function("GMEMD_gradient")
    knl_integral_fnc = mod.get_function("GMEMD_integral")
    
    #Calculate CUDA Execution Dimension
    bdim = (16, 16, 1)
    dx, mx = divmod(target_width, bdim[0])
    dy, my = divmod(target_height, bdim[1])
    gdim = ( (dx + (mx>0)), (dy + (my>0)) )
    
    #CUDA Execution
    knl_gradient_fnc(data, diff, direct, list_x, list_y, list_deg, np.int32(list_len), np.int32(target_width), np.int32(target_height), block=bdim, grid=gdim)
    knl_integral_fnc(result, diff, direct, list_x, list_y, list_deg, np.int32(list_len), np.int32(target_width), np.int32(target_height), block=bdim, grid=gdim)

    #Get CUDA Result
    np_result = np.empty_like(np_data)
    cuda.memcpy_dtoh(np_result, result)
    np_result = np_result / list_len
    
    #Free Memory
    list_x.free()
    list_y.free()
    list_deg.free()
    data.free()
    diff.free()
    direct.free()
    result.free()

    # Reshape to correct width, height
    result_gray=np_result.reshape((target_height, target_width))
    
    return result_gray


#Read Image
filename = sys.argv[1]
img=cv2.imread(filename)
height = img.shape[0]
width = img.shape[1]
channels = img.shape[2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

file = open('default_radius')
for line in file:
    fields = line.strip().split()
    resize_ratio=float(fields[0])
    execute_radius=int(fields[1])
    effective_radius=resize_ratio*execute_radius
    target_height=int(height/resize_ratio)
    target_width=int(width/resize_ratio)
    print(resize_ratio, execute_radius, target_height, target_width)
    resized_gray=cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    # Run SFEGO CUDA Kernel Code
    result_gray=SFEGO(resized_gray, execute_radius)

    # Use this SpatialFrame_result for down stream task
    SpatialFrame_result=cv2.resize(result_gray, (width, height), interpolation=cv2.INTER_LINEAR)

    #Calculate min, max
    result_gray=SpatialFrame_result
    result_min=np.min(result_gray)
    result_max=np.max(result_gray)

    #Real Amplitude
    output_gray=(result_gray-result_min).astype(np.uint8)
    cv2.imshow('Result ', output_gray)
    cv2.waitKey(1)

    #Normalize to 0~255
    output_gray=255*(result_gray-result_min)/(result_max-result_min)
    output_filename=filename+"_GMEMD_SpatialFrame}"+str(round(effective_radius, 2))+"("+str(resize_ratio)+"x"+str(execute_radius)+").png"
    cv2.imwrite(output_filename, output_gray)




