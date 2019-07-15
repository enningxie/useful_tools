# useful_tools
code snippets from daily life.

1. 展示数据的分布图

```  python
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(model_data_01.diff_2, kde=True) # kde控制是否显示核密度估计图
plt.show()
```

2. 数据持久化

```python
# 保存数据
with open('les_05.pickle', 'wb') as f:
    pickle.dump(les, f, -1)

# 恢复数据
with open(file_path, 'rb') as file:
    file_data = pickle.load(file)
```

3. 时间字符串转成`datetime`类型

```python
from datetime import datetime
date_dt = datetime.strptime('2018-11-01 00:00:00', '%Y-%m-%d %H:%M:%S')
```

4. 使用`tqdm`显示循环进度条

```python
from tqdm import tqdm
from time import sleep

for i in tqdm(range(10)):
    sleep(1)
```

5. manage files is using the "with" statement

```python
with open("hello.txt") as hello_file:
    for line in hello_file:
        print(line)
```

6. dataframe sample op / contruct dataset from DataFrame

```python
y_name="price"
# Shuffle the data
train_fraction=0.7
np.random.seed(seed)
# Split the data into train/test subsets.
x_train = data.sample(frac=train_fraction, random_state=seed)
x_test = data.drop(x_train.index)
# Extract the label from the features DataFrame.
y_train = x_train.pop(y_name)
y_test = x_test.pop(y_name)
```

7. tf_op / Create a slice Dataset from a pandas DataFrame and labels

```python
def make_dataset(batch_sz, x, y=None, shuffle=False, shuffle_buffer_size=1000):
    """Create a slice Dataset from a pandas DataFrame and labels"""

    def input_fn():
        if y is not None:
            dataset = tf.data.Dataset.from_tensor_slices((dict(x), y))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(dict(x))
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_sz).repeat()
        else:
            dataset = dataset.batch(batch_sz)
        return dataset.make_one_shot_iterator().get_next()

    return input_fn
```

8. DataFrame to train_x, train_y

```python
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
train_x, train_y = train, train.pop(y_name)
```

9. Download data from URL use `tf.keras`

```python
TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
def maybe_download():
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    return train_path, test_path
```

10. 将本地代码包加入系统路径，避免本地代码找不到情况发生；

```python
import os
import sys

models_path = os.path.join(os.getcwd(), 'models')
sys.path.append(models_path)
```
11. 向`PYTHONPATH`中添加路径；

```python
import os
#export PYTHONPATH=${PYTHONPATH}:"$(pwd)/models"
#running from python you need to set the `os.environ` or the subprocess will not see the directory.

if "PYTHONPATH" in os.environ:
  os.environ['PYTHONPATH'] += os.pathsep +  models_path
else:
  os.environ['PYTHONPATH'] = models_path
```

12. Linux 中开启网易云音乐命令；

``` shell
nohup netease-cloud-music --no-sandbox %U &
```

13. python 中实现类似`switch`操作（通过字典dict方式解决）；

```python
priorities = {
        # t_3会因车辆优先级发生改变? 未解决
        '1': lambda w_: 0,
        '2': lambda w_: EPSILON_1,
        '3': lambda w_: EPSILON_1 + EPSILON_2,  # w_ - DELTA_1
        '4': lambda w_: max(w_ - helpers.convert_datetime_to_day(helpers.DATE_MATCH-car.date_collect), EPSILON_1),
        '5': lambda w_: max(w_ - helpers.convert_datetime_to_day(helpers.DATE_MATCH-car.date_collect), EPSILON_1) + EPSILON_2,
        '6': lambda w_: R,
        '7': lambda w_: R + EPSILON_1,
        '8': lambda w_: EPSILON_1 + EPSILON_2,  # w_ - DELTA_1
        '9': lambda w_: max(w_ - helpers.convert_datetime_to_day(helpers.DATE_MATCH-car.date_collect), EPSILON_1) + R,
        '10': lambda w_: max(w_ - helpers.convert_datetime_to_day(helpers.DATE_MATCH-car.date_collect), EPSILON_1) + R + EPSILON_2,
        '11': lambda w_: max(w_ - EPSILON_2, 0),
        '12': lambda w_: w_,
        '13': lambda w_: w_ + DELTA_2
    }
```

14. numpy 中的花式索引，选取某些索引之外的数据；

```python
orders_lists.append(list(orders_list[over_time]))
orders_lists.append(list(orders_list[~over_time]))
cost_matrix_list.append(cost_matrix[over_time, :])
cost_matrix_list.append(cost_matrix[~over_time, :])
```

15. 判断Series为空；

```python
a = pd.Series([])
a.empty
```

16. 忽略代码中的告警信息；

```python
import warnings
warnings.filterwarnings('ignore')
```

17. 指定显卡运行；

```python
CUDA_VISIBLE_DEVICES=1 python run.py
```

18. python 并发编程;

```python
# 顺序执行
start_time = time.time()
for item in number_list:
    print(evaluate_item(item))
print("Sequential execution in " + str(time.time() - start_time), "seconds")
# 线程池执行
start_time_1 = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(evaluate_item, item) for item in number_list]
    for future in concurrent.futures.as_completed(futures):
        print(future.result())
print ("Thread pool execution in " + str(time.time() - start_time_1), "seconds")
# 进程池
start_time_2 = time.time()
with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(evaluate_item, item) for item in number_list]
    for future in concurrent.futures.as_completed(futures):
        print(future.result())
print ("Process pool execution in " + str(time.time() - start_time_2), "seconds")
```

19. 同步最新的`tensorflow/mdoels`;

``` shell
# 添加远端地址
git remote add upstream git@github.com:tensorflow/models.git
# 获取最新代码
git fetch upstream
# 合并最新代码
git merge upstream/master
# 提交
git push
```

20. Python文件前缀

```python
# coding=utf-8
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
```

21. TensorFlow等深度学习框架，指定显卡运行

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
```

22. 远程服务器Jupiter Notebook

```shell
ssh user_name@remote_ip -N -L localhost:local_port:localhost:remote_port
```

23. numpy去除1的维度

```python
np.squeeze() 
```

24. Pandas DataFrame多条件筛选

```python
raw_data[(raw_data.call_record_no == tmp_phone_call_no) & (raw_data.channel_id == 0)]
```

25. Numpy 多个ndarray合并

```python
np.append(ndarray1, ndarray2)
```

26. Jupyter notebook中pandas dataframe显示不全

```python
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
```

27. tar分包与合并

```shell
# 分包
tar czf - test.pdf | split -b 500m - test.tar.gz
# 合并
cat test.tar.gz* > test.tar.gz
```

28. conda源修改（清华源）

```shell
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```

29. pip修改aliyun源

```shell
vim ~/.pip/pip.conf
# 追加
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/

[install]
trusted-host=mirrors.aliyun.com
```

30. Centos7 添加新用户

```shell
sudo useradd username
# 将新增用户添加进指定组
sudo usermod -a -G groupname username
# 强制用户在下次登录时修改密码
sudo passwd -e username
```

