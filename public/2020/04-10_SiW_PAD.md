Tera Operations Per Second (TOPS)

```sh
ffmpeg -i 006-2-3-4-1.mov -f image2 "foo/video-frame%05d.png"
```
```sh
# Find .face files which has lines with only 3 elements.
find ./Train -name *.face -exec grep -l '  ' {} \;

find ./Train -name *.mov | wc -l
# 2417
find ./Test -name *.mov | wc -l
# 2061

CUDA_VISIBLE_DEVICES='-1' ./extract_faces.py -R 'Train/*/*/*.mov'
CUDA_VISIBLE_DEVICES='0' ./extract_faces.py -R 'Train/*/*/*.mov'
CUDA_VISIBLE_DEVICES='1' ./extract_faces.py -R 'Train/*/*/*.mov'

CUDA_VISIBLE_DEVICES='-1' ./extract_faces.py -R 'Test/*/*/*.mov'
CUDA_VISIBLE_DEVICES='0' ./extract_faces.py -R 'Test/*/*/*.mov'
CUDA_VISIBLE_DEVICES='1' ./extract_faces.py -R 'Test/*/*/*.mov'
```
```py
# ls detect_frame/Train/live/003/003-1-1-1-1
import glob2
ddr = "detect_frame/Train/*/*/*"
image_counts = lambda rr: {os.path.sep.join(os.path.relpath(ii).split(os.path.sep)[1:]): len(os.listdir(ii)) for ii in glob2.glob(rr)}
dss = pd.Series(image_counts(ddr), name='detect')

oor = "orign_frame/Train/*/*/*"
oss = pd.Series(image_counts(oor), name='original')

tt = pd.concat([dss, oss], axis=1, sort=False).fillna(0)
tt.sort_values('detect').head(10)
#                              detect  original  sub
# Train/spoof/121/121-2-3-2-1       0         0    0
# Train/spoof/081/081-2-3-2-1       0         0    0
# Train/spoof/101/101-2-3-2-1       0         0    0
# Train/spoof/101/101-2-3-3-1       0         0    0
# Train/spoof/041/041-2-3-2-1       0         0    0
# Train/spoof/077/077-1-3-3-2       0         0    0
# Train/spoof/041/041-2-3-3-1       0         0    0
# Train/spoof/121/121-2-3-3-1       0         0    0
# Train/spoof/006/006-2-3-4-1     202       202    0
# Train/spoof/060/060-2-3-1-2     202       209    7

for ii in tt[tt.detect == 0].index:
    print(os.path.join('./detect_frame/', ii))
    os.rmdir(os.path.join('./detect_frame/', ii))
    os.rmdir(os.path.join('./orign_frame/', ii))

tt = tt[tt.detect != 0].copy()

tt['sub'] = tt['original'] - tt['detect']
tt['sub'].describe()
# count    2409.000000
# mean        4.427978
# std        13.497904
# min         0.000000
# 25%         0.000000
# 50%         0.000000
# 75%         2.000000
# max       210.000000
# Name: sub, dtype: float64
tt.sort_values('sub')[-5:]
# detect  original  sub
# Train/spoof/156/156-2-3-2-2     274       394  120
# Train/spoof/055/055-2-3-4-2     293       430  137
# Train/spoof/104/104-2-3-4-2     279       426  147
# Train/spoof/032/032-2-3-4-1     266       437  171
# Train/spoof/159/159-2-3-2-1     204       414  210

files_size = lambda dd: [os.stat(os.path.join(dd, ii)).st_size for ii in  os.listdir(dd)]
samples = tt.index[np.random.choice(tt.shape[0], 120, replace=False)]
aa = [files_size(os.path.join("detect_frame", ii)) for ii in samples]
mm = np.mean([np.mean(ii) for ii in aa])
print("~%.2fGB" % (tt['detect'].sum() * mm / 1024 / 1024 / 1024))
# ~28.99GB
!du -hd1 detect_frame/
# 33G     detect_frame/Train

aa = [files_size(os.path.join("orign_frame", ii)) for ii in samples]
mm = np.mean([np.mean(ii) for ii in aa])
print("~%.2fGB" % (tt['original'].sum() * mm / 1024 / 1024 / 1024))
# ~180.40GB
!du -hd1 orign_frame/
# 188G    orign_frame/Train
```
```sh
[sdb]
comment = sdb
browseable = yes
path = /home/tdface/sdb
create mask = 0770
directory mask = 0770
valid users = tdface
force user = tdface
force group = tdface
public = yes
available = yes
writable = yes
```
