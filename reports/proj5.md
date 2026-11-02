# CS188 Fall 2026 Project 5 Machine Learning

官方当期machinelearning.zip地址、hash已追加SOURCE.json。分支edidada/proj5。仅填充models.py/train.py/losses.py，保留原导入集合、backend、数据、autograder与评分门槛。

实现在线Perceptron错分更新直至收敛；两隐藏层ReLU网络拟合回归和数字分类；字符级ReLU RNN累计词表示并忽略零填充；卷积以Tensor.unfold创建窗口再tensordot，不调用现成conv2d；数字卷积模型训练完整参数。Attention计算缩放QK转置、因果mask、softmax后乘V，使用题目原mask和三套指定Linear，保持batch维度。

回归使用均方损失，分类使用支持one-hot软目标的交叉熵；模型使用真实训练数据和Adam优化，验证集仅作为提前停止指标，不读取测试标签或生成硬编码预测。P5原小卷积数据集按官方代码从训练前200个样本取训练和部分复用测试，原数据划分未改，但这个公开测试不能证明独立MNIST泛化。RNN设定梯度裁剪，所有训练有epoch上限。

独立proj5.yml在CPU runner训练，限制BLAS线程为2，执行原全部autograder，要求总分及逐题至少满分。原数据和全部六题实际运行；本地未训练，未向课程平台提交。最终质量以完整Actions的真实误差和准确率为依据。
