import os
import orjson
from subprocess import check_output


def wc(filename):
    return int(check_output(["wc", "-l", filename]).split()[0])


file = "./criteo_subsample.txt"
len_file = wc(file)
num_process = 16
chunk = len_file // num_process + 1
dir_name = f"kaggle_terabyte_{num_process}"
file_counter = 0
line_counter = 0

os.mkdir(dir_name)

dataset_file = open(file, "r+b")
fd = open(f"{dir_name}/part_{file_counter}.txt", "wb")
for ln in dataset_file:
    ln = orjson.loads(ln)
    fd.write(orjson.dumps(ln, option=orjson.OPT_APPEND_NEWLINE))
    line_counter += 1
    if line_counter == chunk:
        fd.close()
        file_counter += 1
        line_counter = 0
        fd = open(f"{dir_name}/part_{file_counter}.txt", "wb")
fd.close()
