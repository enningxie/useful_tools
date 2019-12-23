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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir, os.path.pardir)))
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

31. 将多维列表转为一维列表

```python
from itertools import chain
a = [[1, 2], [3]]
b = list(chain.from_iterable(a))  # [1, 2, 3]
```

32. keras中Lambda进行去除维度1

```python
x = Lambda(lambda x: x[:, 0])(x)  # shape (3, 1, 2) -> shape (3, 2)
```

33. 服务器开放/删除端口：

```shell
/sbin/iptables -I INPUT -p tcp --dport 8080 -j ACCEPT  # 开放
/sbin/iptables -D INPUT 1  # 删除
/sbin/iptables -L -n  # 查看
```

34. Flask框架外网访问：

```python
app.run(host="0.0.0.0")
```

35. cURL服务器post交互：

```shell
curl -X POST -F "image=@dog.jpg" http://localhost:5000/predict
```

36. centos7安装操作Apache2

```
yum install httpd  # 安装
systemctl enable httpd.service  # 服务器开机自启
systemctl start httpd.service  # 手动启动
systemctl restart httpd.service  # 手动重启
systemctl stop httpd.service  # 手动停止
```

37. NLP任务中，char2indices/indices2char字典构建

```python
char2indices = dict((c, i) for i, c in enumerate(chars))
indices2char = dict((i, c) for i, c in enumerate(chars))
```

38. 使用numpy进行shuffle操作

```python
import numpy as np
indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]
```

39. Keras 中将y进行one-hot操作：

```python
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
```

40. 实时看log文件：

```shell
tail -f bert.log
```

41. 查看某一目录下的存储使用详情：

```shell
sudo du -h -d 1 /home
```

42. conda在指定目录创建虚拟环境，并且所创建的虚拟环境不会出现在`conda info -e`中：

```shell
conda create -p ENV python=3.6 
# To activate this environment, use:
conda activate path_to_env
```

43. 使用conda安装指定版本的cudnn：

```shell
conda install cudnn=7.3 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/
# tf2.0
conda install cudnn=7.6.0=cuda10.0_0 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/
```

44. Python3读取json文件，bytes数据转dict

```python
JSON_PATH = 'data/intent_detect_01.json'
with open(JSON_PATH, 'r') as f:
    data = json.load(f)

origin_json = eval(str(data, encoding="utf-8"))

# json 存储
with open('data.json', 'w') as f:
    json.dump(a, f)
```

45. 遍历DataFrame中的每一行

```python
for _, tmp_row in makeup_labels.iterrows():
    print(tmp_row.AspectTerms)
    break
```

46. DataFrame: Count the number of different modalities in each column.

```python
df.nunique()
# Reported Job Title    7174
# SOC minor group         94
# dtype: int64
df.value_counts()
```

47. Cool train/test set split op with DataFrame.

```python
test_set = df.groupby('SOC minor group', as_index=False)['Reported Job Title'].first()
train_set = df[~df['Reported Job Title'].isin(test_set['Reported Job Title'])]

x_train, y_train = train_set['Reported Job Title'], train_set['SOC minor group']
x_test, y_test = test_set['Reported Job Title'], test_set['SOC minor group']
```

48. Gensim 读入word2vector 

```python
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = '../data/glove.6B.100d.txt'
word2vec_output_file = '../data/glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)

model = KeyedVectors.load_word2vec_format('../data/glove.6B.100d.txt.word2vec', binary=False)
vector = model.wv('computer')
```

49. Allowing for multiple models to be loaded (once) and used in multiple threads. Note that the `Graph` is stored in the `Session` object, which is a bit more convenient. From [here](https://github.com/tensorflow/tensorflow/issues/28287) .

```python
# on thread 1
session = tf.Session(graph=tf.Graph())
with session.graph.as_default():
    k.backend.set_session(session)
    model = k.models.load_model(filepath)

# on thread 2
with session.graph.as_default():
    k.backend.set_session(session)
    model.predict(x, **kwargs)
```

50. 给脚本增加日志功能

```
# logconfig.ini
[loggers]
keys=root

[handlers]
keys=defaultHandler

[formatters]
keys=defaultFormatter

[logger_root]
level=INFO
handlers=defaultHandler
qualname=root

[handler_defaultHandler]
class=StreamHandler
formatter=defaultFormatter
# args=('app_.log', 'a')

[formatter_defaultFormatter]
format=%(levelname)s:%(asctime)s:%(message)s
```

```python
import logging
import logging.config

def main():
    # Configure the logging system
    logging.config.fileConfig('logconfig.ini')

    # Variables (to make the calls that follow work)
    hostname = 'www.python.org'
    item = 'spam'
    filename = 'data.csv'
    mode = 'r'

    # Example logging calls (insert into your program)
    logging.critical('Host %s unknown', hostname)
    logging.error("Couldn't find %r", item)
    logging.warning('Feature is deprecated')
    logging.info('Opening file %r, mode=%r', filename, mode)
    logging.debug('Got here')
```

51. Multi-model load and predict.

```python
import tensorflow as tf
import keras.backend as K
# multi-models load
# bert
self.session_bert = tf.Session(graph=tf.Graph())
with self.session_bert.graph.as_default():
    K.set_session(self.session_bert)
    self.model_bert = Bert()
# albert
self.session_albert = tf.Session(graph=tf.Graph())
with self.session_albert.graph.as_default():
    K.set_session(self.session_albert)
    self.model_albert = Albert()
    
# logic op
if mode == 'fast':
    self.session = self.session_albert
    self.model = self.model_albert
else:
    self.session = self.session_bert
    self.model = self.model_bert
    
# predict step
with self.session.graph.as_default():
    K.set_session(self.session)
    sim_scores = self.model.predict(user_responses, intents_item)
```

52. python Web请求示例 REST API：

```python
import requests
import json

KERAS_REST_API_URL = "URL"

headers = {
    'Authorization': '****',
}

JSON_PATH = 'data/tmp_data.json'
with open(JSON_PATH, 'r') as f:
    data = json.load(f)

# submit the request
r = requests.post(KERAS_REST_API_URL, json=data, headers=headers).content

return_json = eval(str(r, encoding="utf-8"))
```

```python
import requests
import json

KERAS_REST_API_URL = "http://127.0.0.1:8080/test"

JSON_PATH = 'data/tmp_data.json'
with open(JSON_PATH, 'r') as f:
    data = json.load(f)
    
r = requests.post(KERAS_REST_API_URL, json=data).content

return_json = eval(str(r, encoding="utf-8"))
```

53. conda/pip 使用

```shell
pip install [package-name]              # 安装名为[package-name]的包
pip install [package-name]==X.X         # 安装名为[package-name]的包并指定版本X.X
pip install [package-name] --proxy=代理服务器IP:端口号         # 使用代理服务器安装
pip install [package-name] --upgrade    # 更新名为[package-name]的包
pip uninstall [package-name]            # 删除名为[package-name]的包
pip list                                # 列出当前环境下已安装的所有包
```

```shell
conda install [package-name]        # 安装名为[package-name]的包
conda install [package-name]=X.X    # 安装名为[package-name]的包并指定版本X.X
conda update [package-name]         # 更新名为[package-name]的包
conda remove [package-name]         # 删除名为[package-name]的包
conda list                          # 列出当前环境下已安装的所有包
conda search [package-name]         # 列出名为[package-name]的包在conda源中的所有可用版本
conda create --name [env-name]      # 建立名为[env-name]的Conda虚拟环境
conda activate [env-name]           # 进入名为[env-name]的Conda虚拟环境
conda deactivate                    # 退出当前的Conda虚拟环境
conda env remove --name [env-name]  # 删除名为[env-name]的Conda虚拟环境
conda env list                      # 列出所有Conda虚拟环境
```

54. 各种方式增加/降低张量维度

```python
# numpy
image = np.expand_dims(image, axis=-1)  
# tensorflow
# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
# tensorflow
tf.expand_dims(encoded_sample_pred_text, 0)
tf.squeeze(predictions, 0)

```

55. TensorFlow恢复与保存变量的代码框架`tf.train.Checkpoint`

```python
# train.py 模型训练阶段

model = MyModel()
# 实例化Checkpoint，指定保存对象为model（如果需要保存Optimizer的参数也可加入）
checkpoint = tf.train.Checkpoint(myModel=model)
# ...（模型训练代码）
# 模型训练完毕后将参数保存到文件（也可以在模型训练过程中每隔一段时间就保存一次）
checkpoint.save('./save/model.ckpt')

# test.py 模型使用阶段

model = MyModel()
checkpoint = tf.train.Checkpoint(myModel=model)             # 实例化Checkpoint，指定恢复对象为model
checkpoint.restore(tf.train.latest_checkpoint('./save'))    # 从文件恢复模型参数
# 模型使用代码
```

56. TF2中查看当前环境下GPU/CPU，并指定GPU运行

```python
# 显示可用硬件资源
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus, cpus)

# 指定GPU
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(devices=gpus[0:2], device_type='GPU')

# 使用环境变量CUDA_VISIBLE_DEVICES也可以控制
export CUDA_VISIBLE_DEVICES=2,3
or
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
```

57. 设置显存使用策略

默认情况下，TensorFlow将使用几乎所有可用的显存，以避免内存碎片化所带来的性能损失。TensorFlow提供了两种显存使用策略，能够更灵活地控制程序的显存使用方式：

- 仅在需要时申请显存空间（程序初始运行时消耗很少的显存，随着程序的运行而动态申请显存）；

可以通过 `tf.config.experimental.set_memory_growth` 将 GPU 的显存使用策略设置为 “仅在需要时申请显存空间”。

```python
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, True)
```

- 限制消耗固定大小的显存（程序不会超出限定的显存大小，若超出的报错）

通过 `tf.config.experimental.set_virtual_device_configuration` 选项并传入 `tf.config.experimental.VirtualDeviceConfiguration` 实例，设置 TensorFlow 固定消耗 `GPU:0` 的 1GB 显存（其实可以理解为建立了一个显存大小为 1GB 的 “虚拟 GPU”）。

```python
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
```

58. 单GPU模拟多GPU环境

在实体 GPU `GPU:0` 的基础上建立了两个显存均为 2GB 的虚拟 GPU。

```python
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048),
     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
```

59. 指定GPU并配置GPU

```python
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    used_gpus = physical_devices[:1]
    tf.config.experimental.set_visible_devices(devices=used_gpus, device_type='GPU')
    for tmp_gpu in used_gpus:
        tf.config.experimental.set_memory_growth(device=tmp_gpu, enable=True)
```

60. 有关时间的操作，`datetime`

```python
import datetime

# from str
tmp_date = datetime.datetime.strptime(tmp_data['create_date'], "%Y-%m-%d %H:%M:%S")
# format
tmp_create_time = '{}{:02d}{:02d}'.format(str(tmp_date.year)[-2:], tmp_date.month, tmp_date.day)
# +/-
tmp_date_ = tmp_date + datetime.timedelta(days=int(res['repay_time']))
# datetime format
localtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

61. 将path中包含的`~`和`~user`转换成用户目录

```python
import os

print(os.path.expanduser(path))
```

62. `logging`模块

```python
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
```

63. 正则去除文本中的`HTML标签`及`\n`等

```python
import re
html = '<body>hello\n</body>'
re.compile(r'<[^>]+>|\n',re.S).sub('', html)
```

64. `pyhanlp`附加数据存放路径：

```python
/home/xen/envs/tf2/lib/python3.6/site-packages/pyhanlp/static/data-for-1.7.5.zip
```

65. 当更新完你的`github`密码时，原先本地那些repo再次push的时候就会报错，解决方法：

```shell
# macos
git config --global credential.helper osxkeychain
# windows
git config --global credential.helper wincred

# 下次push的时候，再次输入username/password就可以正常使用了
```

66. Convert TensorFlow 1.0's checkpoints to PyTorch's and TensorFlow 2.0's checkpoints.

```python
from transformers.convert_bert_original_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch
from transformers.convert_pytorch_checkpoint_to_tf2 import convert_pt_checkpoint_to_tf
import logging
import shutil
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Converter(object):
    def __init__(self, origin_checkpoints_path, config_file, dump_path):
        self.model_type = 'bert'
        self.origin_checkpoints_path = origin_checkpoints_path
        self.config_file = config_file
        self.dump_path = dump_path

    def convert_to_pytorch(self):
        convert_tf_checkpoint_to_pytorch(
            self.origin_checkpoints_path, self.config_file, os.path.join(self.dump_path, 'pytorch_model.bin')
        )
        shutil.copyfile(self.config_file, os.path.join(self.dump_path, 'config.json'))
        shutil.copyfile(os.path.join(os.path.dirname(self.config_file), 'vocab.txt'),
                        os.path.join(self.dump_path, 'vocab.txt'))
        logging.info("Convert to PyTorch's checkpoints finished.")

    def convert_to_tf2(self):
        self.convert_to_pytorch()
        convert_pt_checkpoint_to_tf(
            model_type=self.model_type,
            pytorch_checkpoint_path=os.path.join(self.dump_path, 'pytorch_model.bin'),
            config_file=os.path.join(self.dump_path, 'config.json'),
            tf_dump_path=os.path.join(self.dump_path, 'tf_model.h5'),
            compare_with_pt_model=True
        )
        logging.info("Convert to TensorFlow 2.0's checkpoints finished.")


if __name__ == '__main__':
    tmp_converter = Converter(
        '/Data/public/Bert/chinese_wwm_L-12_H-768_A-12/bert_model.ckpt',
        '/Data/public/Bert/chinese_wwm_L-12_H-768_A-12/bert_config.json',
        '/Data/public/transformers/chinese_wwm_L-12_H-768_A-12'
    )

    # start converting...
    tmp_converter.convert_to_tf2()
```

