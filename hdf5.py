import h5py

file_path = "berkeley_sawyer_traj973.hdf5"

with h5py.File(file_path, "r") as f:
    print("文件里的顶层key：")
    print(list(f.keys()))
def print_hdf5(name, obj):
    print(name, type(obj))
with h5py.File("berkeley_sawyer_traj973.hdf5", "r") as f:
    f.visititems(print_hdf5)


with h5py.File(file_path, "r") as f:
    qpos = f["env/qpos"][:5]   # 前 5 帧
    actions = f["policy/actions"][:5]

print(qpos)
print(actions)
