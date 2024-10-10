model_name='phasenet_das'
resume=''
backbone='unet'
phases=['P', 'S']
device='cuda'
workers=1
batch_size=1
use_deterministic_algorithms=False
amp=False
world_size=1
dist_url='env://'
data_path='./MiDAS_good_72/test'
data_list='files.txt'
label_path='./MiDAS_good_72/test'
label_list=None
hdf5_file=None
prefix=''
format='h5'
dataset='das'
result_path='./results'
plot_figure=False
min_prob=0.3
add_polarity=False
add_event=False
highpass_filter=0.0
response_xml=None
folder_depth=0
cut_patch=False
#nt=20480
#nx=5120
nt = 12288
nx = 2048
resample_time=False
resample_space=False
system=None
location=None
skip_existing=False
rank=0
gpu=0
distributed=True
dist_backend='nccl'