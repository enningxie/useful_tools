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
