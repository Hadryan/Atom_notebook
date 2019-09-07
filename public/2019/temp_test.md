```py
from scipy import stats

aa = pd.ExcelFile('./555(1).xlsx')
pp = aa.parse('预测值')
ll = aa.parse('真实值')
pp = pp.values.reshape(-1)
ll = ll.values.reshape(-1)
ss = pp - ll
plt.hist(ss, bins=50)

mean_heights = np.mean(ss)
SE = np.std(ss) / np.sqrt(ss.shape[0])
l, u = stats.t.interval(0.95, df=9, loc=mean_heights, scale=SE)
print(l, u)
plt.plot([l, u], [400, 400], '-', color='r', linewidth=4, label="Confidence Interval")
```
```py
keyword
number
number = 5
_q = dict(key="only my railgun", pagingVO=dict(page=1, pageSize=number))
_q = json.dumps(_q)
url = "https://www.xiami.com/search?key={}".format(keyword)
resp = requests.get(url)
cookie = resp.cookies.get("xm_sg_tk", "").split("_")[0]
origin_str = "%s_xmMain_/api/search/searchSongs_%s" % (cookie, _q)

import hashlib
_s = hashlib.md5(origin_str.encode()).hexdigest()
params = dict(_q=_q, _s=_s)
requests.get("https://www.xiami.com/api/search/searchSongs", params=params, headers=headers_xiami).json()
_s

```

```py
keyword = "fripSide only my railgun"
search_url = 'http://api.xiami.com/web'
params = {
    "key": keyword,
    "v": "2.0",
    "app_key": "1",
    "r": "search/songs",
    "page": 1,
    "limit": 20,
}
resp = requests.get(search_url, params=params, headers=headers_xiami)
```
```py
cd test_data/
import glob2
aa = glob2.glob('./test_images/*.jpg')
bb = glob2.glob('./test_results/*.png')

from skimage.io import imread
fig, axes = plt.subplots(2, 4)
for id, ii in enumerate(aa):
    imm = imread(ii)
    axes[0, id].imshow(imm)
    axes[0, id].set_axis_off()
for id, ii in enumerate(bb):
    imm = imread(ii)
    axes[1, id].imshow(imm)
    axes[1, id].set_axis_off()
fig.tight_layout()
```
