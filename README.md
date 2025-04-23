# 运行：

环境配置（ Ubuntu 22.04、NVIDIA-SMI 550.67、Driver Version: 550.67、CUDA Version: 12.4）：
`conda create -n CLGADN python=3.8`

`conda activate CLGADN`

安装库

`pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple`

`
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple`

`
pip install pandas -i https://mirrors.aliyun.com/pypi/simple`

`
pip install tqdm`

`
pip install matplotlib`

`pip install scikit-learn`

先运行perpare_data.py，生成训练集以及测试集（在data/courses目录下），再运行main.py文件

已生成好的训练集和测试集可以从此下载：https://drive.google.com/drive/folders/1LzVuhAsXQyYek5zn4rEz9WLkjieSq_a1?usp=drive_link

#  文件结构：

[BarlowTwinsLoss.py](BarlowTwinsLoss.py)  对比学习模块

 [CLGADN.py](CLGADN.py)  模型

 [CourseDataset.py](CourseDataset.py)  数据类

 [Evaluation.py](Evaluation.py) 评价指标

  [load_data.py](load_data.py)  加载数据

 [main.py](main.py)  主函数

 [Parse.py](Parse.py)  配置类

 [perpare_data.py](perpare_data.py)  将原始数据准备成模型数据

* all_ratings.csv 所有点击数据
* courses_info_with_pre.csv 课程信息数据
* train_data.csv 训练集所有数据
* train_data_transformed_01.csv 模型可以直接用的训练集数据
* test_ans.csv 测试集
* test_pos.csv 测试集所包含的用户的历史记录
* test_data_transformed.csv 模型可以直接用的测试集数据

 [Utils.py](Utils.py) 工具类