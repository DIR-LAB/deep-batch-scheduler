import os
import numpy as np
def average(filename):
    with open(filename) as f:
        # result = np.array([0]*18, dtype="float32")
        result = 0
        job_num = 0
        for line in f.readlines():
            if not line.startswith(";"):
                job_num += 1
                line_arr = line.strip().split()
                if job_num > 1:
                    result += float(line_arr[1]) - temp
                temp = float(line_arr[1])
        return result/job_num


filenames = [
    "CTC-SP2-1996-3.1-cln.swf",
    "SDSC-SP2-1998-4.2-cln.swf",
    "SDSC-BLUE-2000-4.2-cln.swf",
    "HPC2N-2002-2.2-cln.swf",
    "ANL-Intrepid-2009-1.swf",
    "lublin_256.swf"
]
cpus = [338, 128, 1152, 240, 163840, 256]
path = "./data"
for index, i in enumerate(filenames):
    aver = average(os.path.join(path,i))
    print(i,aver)