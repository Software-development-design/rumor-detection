import os
import json
import seaborn
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_curve, auc, f1_score, confusion_matrix

# 0:真话 1：谣言


# 简述BILSTM长短期记忆神经网络的原理
# 双向长短期记忆网络（Bidirectional LSTM）是一种循环神经网络（Recurrent Neural Network, RNN）的变体，用于处理序列数据。
# 与传统的单向 LSTM 不同，双向 LSTM 在每个时间步上都有两个 LSTM 单元，分别按正向和反向顺序处理输入序列。
# 它能够捕捉到序列数据中前后依赖关系的特征，从而更好地理解和表示序列数据。

# 双向 LSTM 的原理如下：
# 输入序列：首先，将输入序列按时间步拆分为一个个单词或符号，并将其作为 LSTM 的输入。
# 正向传播：正向 LSTM 单元按照时间步的顺序逐个处理输入序列。在每个时间步，正向 LSTM 单元接收当前时间步的输入和前一时间步的隐藏状态，
#   并计算当前时间步的输出和隐藏状态。输出可以是当前时间步的隐藏状态、当前时间步的预测结果或其他自定义的输出。
# 反向传播：反向 LSTM 单元按照时间步的逆序逐个处理输入序列。在每个时间步，反向 LSTM 单元接收当前时间步的输入和后一时间步的隐藏状态，并计算当前时间步的输出和隐藏状态。
# 合并结果：正向 LSTM 和反向 LSTM 的输出在每个时间步上都有一个值，可以将这两个值按时间步进行连接或合并，形成一个更丰富的表示。
# 输出结果：可以使用合并结果进行进一步的处理，如添加额外的全连接层、进行分类、回归或其他任务。

# 双向 LSTM 的优点在于它能够同时考虑到过去和未来的信息，捕捉到输入序列中的长期依赖关系。
# 它通过正向和反向传播的结合，可以在每个时间步上利用过去和未来的上下文信息，从而更好地建模序列数据。
# 在文本分类、情感分析、命名实体识别等自然语言处理任务中，双向 LSTM 常被应用于对输入文本的建模，能够有效地捕捉上下文信息，提升模型的性能和表达能力。


# 从两个目录（true_dir和fake_dir）中读取JSON文件，并将文件中的文本数据提取出来。
# 然后将文本数据分为训练集和测试集，并将它们分别写入到"train.txt"和"test.txt"两个文件中。

# 定义一个名为`get_train_val_txt`的函数，该函数有两个参数`true_dir`和`fake_dir`，默认值分别为"non-rumor-repost"和"rumor-repost"。
def get_train_val_txt(true_dir="non-rumor-repost", fake_dir="rumor-repost"):
    # 创建了两个空列表`datasets`和`labels`，用于存储文本数据和对应的标签。
    datasets, labels = [], []
    # 使用`enumerate`函数遍历`[true_dir, fake_dir]`列表，其中`i`是索引，`json_dir`是目录路径。
    for i, json_dir in enumerate([true_dir, fake_dir]):
        # 使用`os.listdir`函数获取`json_dir`目录下的所有文件名，并使用`tqdm`函数进行迭代显示进度。
        for json_name in tqdm(os.listdir(json_dir)):
            # 构建JSON文件的完整路径`json_path`。
            json_path = json_dir + "/" + json_name
            # 使用`open`函数打开JSON文件，并指定编码为"utf-8"。
            f = open(json_path, encoding="utf-8")
            # 使用`json.load`函数加载JSON文件的内容，并将结果存储在`json_list`变量中。
            json_list = json.load(f)
            # 对于`json_list`中的每个JSON对象，提取其中的"text"字段，并进行一系列的文本处理操作，包括去除空格、换行符、方括号、省略号和"@"符号。
            for json_obj in json_list:
                text = json_obj.get("text", "")
                text = text.strip().replace("\n", "").replace("[", "").replace("]", "").replace("…", "").replace("@",
                                                                                                                 "")
                # 如果文本长度大于10个字符，则将文本添加到`datasets`列表中，将`i`添加到`labels`列表中作为标签。
                if len(text) > 10:
                    datasets.append(text)
                    labels.append(i)
            # 关闭打开的JSON文件。
            f.close()
    # 使用`train_test_split`函数将数据集`datasets`和标签`labels`划分为训练集和测试集，其中训练集占总数据的80%，并进行随机洗牌（shuffle=True）。
    datasets_train, datasets_test, labels_train, labels_test = train_test_split(datasets, labels, train_size=0.8,
                                                                                shuffle=True)
    # 使用`open`函数创建两个文件对象，分别用于写入训练集和测试集的数据。
    with open("train.txt", "w", encoding="utf-8") as f1, open("test.txt", "w", encoding="utf-8") as f2:
        # 使用循环遍历训练集的数据，将文本和标签以制表符分隔的形式写入"train.txt"文件中。
        for i in range(len(datasets_train)):
            f1.write(datasets_train[i] + "\t" + str(labels_train[i]) + "\n")

        # 使用循环遍历测试集的数据，将文本和标签以制表符分隔的形式写入"test.txt"文件中。
        for i in range(len(datasets_test)):
            f2.write(datasets_test[i] + "\t" + str(labels_test[i]) + "\n")
# 综上所述，该代码的主要功能是将两个目录中的JSON文件提取出文本数据，并将其划分为训练集和测试集，然后将数据写入到两个文本文件中。
# 这些文本文件可以作为机器学习模型的输入数据，用于进行文本分类或其他相关任务。


# 构建一个单词字典，用于将单词映射为索引。
# 根据提供的两个文件路径（root1和root2）中的文本数据进行统计，并选择出现频率最高的n_common个单词作为字典的一部分。

# 定义了一个名为`get_word_dict`的函数，该函数有三个参数`root1`、`root2`和`n_common`，分别表示两个文件路径和要选择的常见单词的数量
# 默认值分别为"train.txt"、"test.txt"和3000。
def get_word_dict(root1="train.txt", root2="test.txt", n_common=3000):
    # 创建一个空的`Counter`对象`word_count`，用于统计单词出现的次数。
    word_count = Counter()
    # 使用`for`循环遍历`[root1, root2]`列表，其中`root`表示文件路径。
    for root in [root1, root2]:
        # 使用`open`函数打开文件，并指定编码为"utf-8"。
        with open(root, "r", encoding="utf-8") as f:
            # 使用`readlines`方法读取文件中的所有行，并使用`strip`方法去除行末尾的换行符。
            for line in f.readlines():
                # 使用`split`方法将每行按制表符分割为单词和标签（在该代码中未使用标签）。
                line_split = line.strip().split("\t")
                # 遍历每个单词，将其添加到`word_count`的计数中。
                for word in line_split[0]:
                    word_count[word] += 1
    # 使用`most_common`方法从`word_count`中选择出现频率最高的`n_common`个单词，并将结果存储在`most_common`变量中。
    most_common = word_count.most_common(n_common)
    # 创建一个字典`word2index_dict`，用于将单词映射为索引。其中，单词的索引从2开始，前两个保留给特殊标记（"UNK"表示未知单词，"PAD"表示填充）。
    # 使用`enumerate`函数遍历`most_common`列表，其中`index`表示索引，`(word, count)`表示单词和计数。
    # 将单词和对应的索引加入到`word2index_dict`中，索引加2是为了给保留的特殊标记留出索引位置。
    word2index_dict = {word: index + 2 for index, (word, count) in enumerate(most_common)}
    # 将"UNK"的索引设为1，表示未知单词。
    word2index_dict["UNK"] = 1
    # 将"PAD"的索引设为0，表示填充。
    word2index_dict["PAD"] = 0

    # 返回构建的`word2index_dict`字典。
    return word2index_dict
# 综上所述，该代码的主要功能是从给定的文件中统计单词出现的次数，并构建一个单词字典，将常见的单词映射为索引。
# 这个字典可以用于将文本数据转换为数字序列，以便在机器学习模型中进行处理。


# 定义了一个名为`DataGenerator`的数据集类，用于生成训练或测试数据。
# 继承自`Dataset`类，通过重写`__init__`、`__getitem__`、`__len__`等方法来实现数据集的功能。

# 定义了一个名为`DataGenerator`的类，该类继承自`Dataset`。
class DataGenerator(Dataset):
    # 在`__init__`方法中，初始化了一些属性，包括`root`表示数据文件路径，`max_len`表示文本的最大长度。
    # `word2index_dict`表示单词到索引的映射字典，`datasets`和`labels`表示数据集和标签。
    def __init__(self, word2index_dict, root="train.txt", max_len=50):
        super(DataGenerator, self).__init__()
        self.root = root
        self.max_len = max_len
        self.word2index_dict = word2index_dict
        self.datasets, self.labels = self.get_datasets()

    # 在`__getitem__`方法中，根据索引`item`获取数据集中的一条数据和对应的标签。
    # 如果数据长度小于`max_len`，则使用0进行填充；如果数据长度超过`max_len`，则截取前`max_len`个单词。
    def __getitem__(self, item):
        dataset = self.datasets[item]
        label = self.labels[item]
        if len(dataset) < self.max_len:
            dataset += [0] * (self.max_len - len(dataset))
        else:
            dataset = dataset[:self.max_len]

        # 返回一个由`torch.LongTensor`类型的数据集和`torch.LongTensor`类型的标签组成的元组。
        return torch.LongTensor(dataset), torch.from_numpy(np.array(label)).long()

    # 在`__len__`方法中，返回数据集的长度，即标签的数量。
    def __len__(self):
        return len(self.labels)

    # 定义了一个名为`get_datasets`的方法，用于从文件中读取数据集和标签。
    def get_datasets(self):
        datasets, labels = [], []
        # 使用`open`函数打开文件，并指定编码为"utf-8"。
        with open(self.root, "r", encoding="utf-8") as f:
            # 使用`readlines`方法读取文件中的所有行，并使用`strip`方法去除行末尾的换行符。
            for line in f.readlines():
                # 使用`split`方法将每行按制表符分割为单词和标签。
                line_split = line.strip().split("\t")
                # 将每个单词根据`word2index_dict`进行索引转换，并将结果存储在`datasets`列表中。
                # 将每个标签转换为整数，并存储在`labels`列表中。
                datasets.append([self.word2index_dict.get(word, 1) for word in list(line_split[0])])
                labels.append(int(line_split[1]))

        # 返回`datasets`和`labels`作为数据集的结果。
        return datasets, labels
# 综上所述，该代码定义了一个数据集类`DataGenerator`，该类从文件中读取数据集和标签，并在需要时对数据进行填充或截断。
# 通过使用这个数据集类，可以方便地生成适用于训练或测试的数据样本。


# 该代码定义了一个名为`BiLSTMModel`的双向LSTM模型类，用于文本分类任务。
# 该模型继承自`nn.Module`类，通过重写`__init__`和`forward`方法来定义模型的结构和前向计算过程。

# 定义了一个名为`BiLSTMModel`的类，该类继承自`nn.Module`。
class BiLSTMModel(nn.Module):

    # 在`__init__`方法中，初始化了一些模型的组件。
    def __init__(self, num_vocab):
        super(BiLSTMModel, self).__init__()
        # `embedding`是一个`nn.Embedding`层，用于将输入的词索引转换为词向量表示，其中`num_vocab`表示词汇表的大小，`embedding_dim`表示词向量的维度为128。
        self.embedding = nn.Embedding(num_embeddings=num_vocab, embedding_dim=128)
        # `lstm`是一个`nn.LSTM`层，用于对输入的词向量进行双向LSTM处理，其中`input_size`表示输入的特征维度为128，`hidden_size`表示LSTM隐藏状态的维度为256，
        # `bidirectional=True`表示使用双向LSTM，`batch_first=True`表示输入的数据格式为(batch_size, sequence_length, input_size)，
        # `num_layers=2`表示LSTM层的层数为2。
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, bidirectional=True, batch_first=True, num_layers=2)
        # `fc1`是一个包含一个线性层和ReLU激活函数的序列，用于进行非线性变换，其中线性层的输入维度为512，输出维度为512。
        self.fc1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True)
        )
        # `fc2`是一个线性层，用于将非线性变换的结果映射为输出类别的分数，其中输入维度为512，输出维度为2（假设有2个类别）。
        self.fc2 = nn.Linear(512, 2)

    # 在`forward`方法中，定义了模型的前向计算过程。
    def forward(self, x):
        # 将输入`x`通过嵌入层`embedding`转换为词向量表示，得到`out`。
        out = self.embedding(x)
        # 将词向量输入到LSTM层`lstm`中，得到输出`outputs`和最后一个时刻的隐藏状态`h`和细胞状态`c`。
        outputs, (h, c) = self.lstm(out)
        # 将正向和反向的最后一个时刻的隐藏状态`h`拼接起来，形成一个512维的表示，通过`torch.cat`函数和`dim=-1`实现，得到`out`。
        # 将`out`输入到非线性变换层`fc1`中，经过ReLU激活函数进行非线性变换。
        out = torch.cat([h[-1, :, :], h[-2, :, :]], dim=-1)
        # 将非线性变换后的结果`out`输入到线性层`fc2`中，得到最终的输出结果。
        out = self.fc1(out)

        # 返回最终的输出结果`fc2(out)`。
        return self.fc2(out)
# 综上所述，该代码定义了一个包含嵌入层、双向LSTM层和两个线性层的双向LSTM模型。该模型接收词索引作为输入，经过嵌入层和双向LSTM层的处理，最终输出分类结果的分数。


# 该代码实现了一个训练函数`train()`，用于训练`BiLSTMModel`模型并输出训练过程中的准确率和测试准确率。
def train():
    # 通过`torch.device("cuda")`将设备设置为CUDA。
    device = torch.device("cuda")
    # 调用`get_word_dict()`函数获取单词到索引的字典`word2index_dict`。
    word2index_dict = get_word_dict()
    # 根据`word2index_dict`的长度创建一个`BiLSTMModel`模型，并将模型移动到设备上。
    model = BiLSTMModel(len(word2index_dict)).to(device)
    # 使用`optim.Adam`优化器来优化模型的参数，学习率设置为1e-3。
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # 使用`optim.lr_scheduler.StepLR`学习率调度器，设置每2000个步骤衰减学习率，衰减率为0.9。
    schedule = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.9)
    # 使用`nn.CrossEntropyLoss()`作为损失函数。
    loss_func = nn.CrossEntropyLoss()
    # 创建训练数据集的`DataLoader`，使用`DataGenerator`生成训练数据集，设置`shuffle=True`进行数据随机洗牌，`batch_size`为64。
    train_loader = DataLoader(DataGenerator(word2index_dict, root="train.txt"), shuffle=True, batch_size=64)
    # 创建测试数据集的`DataLoader`，同样使用`DataGenerator`生成测试数据集，设置`shuffle=False`不进行数据随机洗牌，`batch_size`为64。
    test_loader = DataLoader(DataGenerator(word2index_dict, root="test.txt"), shuffle=False, batch_size=64)
    # 进行30个epoch的训练，每个epoch都执行以下操作：
    for epoch in range(31):
        # 调用`train_one_epoch`函数对模型进行一轮训练，传入模型、训练数据集、损失函数、优化器、学习率调度器、设备和当前的epoch。
        train_accuracy = train_one_epoch(model, train_loader, loss_func, optimizer, schedule, device, epoch)
        # 调用`get_test_result`函数对模型进行测试，传入模型、测试数据集和设备。
        test_accuracy = get_test_result(model, test_loader, device)
        # 打印当前epoch的训练准确率和测试准确率。
        print(f"epoch:{epoch + 1},train accuracy:{train_accuracy},test accuracy:{test_accuracy}")
        # 如果当前epoch是10的倍数，将模型保存到文件中，文件名为`bilstm_model_epochX.pth`，其中X为当前epoch数。
        if (epoch + 1) % 10 == 0:
            torch.save(model, f"bilstm_model_epoch{epoch + 1}.pth")
# 综上所述，该代码实现了一个完整的训练过程，包括模型的初始化、数据加载、模型训练、测试和保存模型等步骤。
# 在训练过程中，每个epoch会输出训练准确率和测试准确率，并在每10个epoch时保存模型。


# 该代码实现了一个函数`train_one_epoch()`，用于对模型进行一轮训练，并返回训练准确率。
def train_one_epoch(model, train_loader, loss_func, optimizer, schedule, device, epoch):
    # 将模型设置为训练模式，即调用`model.train()`。
    model.train()
    # 使用`tqdm`包装训练数据加载器`train_loader`，用于显示进度条。
    data = tqdm(train_loader)
    # 初始化空的`labels_true`和`labels_pred`，用于存储真实标签和预测标签。
    labels_true, labels_pred = np.array([]), np.array([])
    # 对于每个批次的数据，使用`enumerate`函数遍历`train_loader`。
    for batch, (x, y) in enumerate(data):
        # 将当前批次的预测标签`labels_pred`与真实标签`y.numpy()`进行拼接，更新`labels_true`。
        labels_true = np.concatenate([labels_pred, y.numpy()], axis=-1)
        # 将输入数据`x`和标签数据`y`移动到设备`device`上。
        datasets_train, labels_train = x.to(device), y.to(device)
        # 将输入数据传入模型`model`进行前向计算，得到预测结果`prob`。
        prob = model(datasets_train)
        # 使用`torch.argmax`函数找到预测结果中最大值所在的索引，得到预测标签`pred`，并将其转换为NumPy数组。
        pred = torch.argmax(prob, dim=-1).cpu().numpy()
        # 将当前批次的预测标签`labels_pred`与`pred`进行拼接，更新`labels_pred`。
        labels_pred = np.concatenate([labels_pred, pred], axis=-1)
        # 计算损失函数`loss`，将预测结果`prob`和真实标签`labels_train`传入损失函数`loss_func`。
        loss = loss_func(prob, labels_train)
        # 清除优化器中的梯度信息，调用`optimizer.zero_grad()`。
        optimizer.zero_grad()
        # 反向传播，调用`loss.backward()`。
        loss.backward()
        # 更新模型参数，调用`optimizer.step()`。
        optimizer.step()
        # 更新学习率，调用`schedule.step()`。
        schedule.step()
        # 使用`tqdm`的`set_description_str`函数更新进度条的描述信息，包括当前epoch、批次、损失值和学习率。
        data.set_description_str(
            f"epoch:{epoch + 1},batch:{batch + 1},loss:{loss.item()},lr:{schedule.get_last_lr()[0]}")
    # 计算准确率，将`labels_pred`与`labels_true`进行比较，将比较结果转换为整数类型后取平均值。
    accuracy = np.mean(np.array(labels_pred == labels_true).astype(int))

    # 返回准确率`accuracy`。
    return accuracy
# 综上所述，该代码实现了模型的一轮训练过程，包括将模型设置为训练模式、遍历训练数据集、计算损失、更新模型参数、更新学习率和计算训练准确率等步骤。
# 在每个批次的训练中，会输出当前的训练损失和学习率，并在最后返回训练准确率。


# 该代码实现了一个函数`get_test_result()`，用于对模型进行测试并输出测试结果。
def get_test_result(model, test_loader, device):
    # 将模型设置为评估模式，即调用`model.eval()`。
    model.eval()
    # 使用`tqdm`包装测试数据加载器`test_loader`，用于显示进度条。
    data = tqdm(test_loader)
    # 初始化空的`labels_true`和`labels_pred`，用于存储真实标签和预测标签。
    labels_true, labels_pred = np.array([]), np.array([])
    # 初始化空的`labels_prob`，用于存储预测结果的概率。
    labels_prob = []
    # 使用`torch.no_grad()`上下文管理器，表示不需要计算梯度。
    with torch.no_grad():
        # 对于每个批次的测试数据，使用`for x, y in data:`遍历`test_loader`。
        for x, y in data:
            # 将当前批次的真实标签`y.numpy()`与`labels_true`进行拼接，更新`labels_true`。
            labels_true = np.concatenate([labels_true, y.numpy()], axis=-1)
            # 将输入数据`x`移动到设备`device`上。
            datasets_test = x.to(device)
            # 将输入数据传入模型`model`进行前向计算，得到预测结果`prob`。
            prob = model(datasets_test)
            # 使用`torch.argmax`函数找到预测结果中最大值所在的索引，得到预测标签`pred`，并将其转换为NumPy数组。
            pred = torch.argmax(prob, dim=-1).cpu().numpy()
            # 将当前批次的预测标签`pred`与`labels_pred`进行拼接，更新`labels_pred`。
            labels_pred = np.concatenate([labels_pred, pred], axis=-1)
            # 将预测结果的概率`prob`转换为NumPy数组后，将其添加到`labels_prob`中。
            labels_prob.append(prob.cpu().numpy())
    # 将`labels_prob`进行拼接，按行连接，得到完整的预测结果概率矩阵`labels_prob`。
    labels_prob = np.concatenate(labels_prob, axis=0)
    # 计算准确率`accuracy`，将`labels_pred`与`labels_true`进行比较，将比较结果转换为整数类型后取平均值。
    precision = precision_score(labels_true, labels_pred)
    # 打印测试准确率`accuracy`、精确率`precision`、召回率`recall`和F1值`f1`。
    recall = recall_score(labels_true, labels_pred)
    f1 = f1_score(labels_true, labels_pred)
    accuracy = np.mean(np.array(labels_pred == labels_true).astype(int))
    print(f"accuracy:{accuracy},precision:{precision},recall:{recall},f1:{f1}")

    # 计算ROC曲线的假正率FPR和真正率TPR，使用`roc_curve`函数，其中`labels_prob[:,-1]`表示取预测结果的概率矩阵的最后一列，即正类的概率。
    fpr, tpr, _ = roc_curve(labels_true, labels_prob[:, -1])

    # 绘制ROC曲线，包括绘制对角线和绘制实际曲线，计算并显示AUC值。
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "r--")
    plt.plot(fpr, tpr, "green", label=f"AUC:{auc(fpr, tpr)}")
    plt.legend()
    plt.title("BiLSTM roc_curve")
    # 保存ROC曲线图像到文件`roc_curve.png`。
    plt.savefig("roc_curve.png")
    # 计算混淆矩阵，使用`confusion_matrix`函数，其中`normalize="true"`表示将混淆矩阵进行归一化，即计算每个类别的分类准确率。
    matrix = confusion_matrix(labels_true, labels_pred, normalize="true")

    # 绘制混淆矩阵，使用`seaborn.heatmap`函数绘制热力图，标注每个单元格的值，并保存图像到文件`confusion_matrix.png`。
    plt.figure(figsize=(8, 8))
    seaborn.heatmap(matrix, annot=True, cmap="GnBu")
    plt.title("confusion_matrix")
    plt.savefig("confusion_matrix.png")

    # 返回测试准确率`accuracy`。
    return accuracy
# 综上所述，该代码实现了对模型进行测试，并输出测试结果，包括测试准确率、精确率、召回率、F1值、ROC曲线和混淆矩阵。
# 在测试过程中，会将预测结果的概率进行保存，并根据真实标签和预测标签计算各种评估指标，并将ROC曲线和混淆矩阵进行绘制和保存。


# 该代码实现了一个函数`predict()`，用于对输入的句子进行预测并输出预测结果。
def predict(sentence, model_path="bilstm_model_epoch10.pth"):
    # 打印输入句子和预测结果的提示信息。
    print(f"{sentence}  的预测结果为:", end=" ")
    # 定义标签列表`labels`，包含"真话"和"谣言"两个标签。
    labels = ["真话", "谣言"]
    # 将模型加载到设备`device`上。
    device = torch.device("cuda")
    # 调用`get_word_dict()`函数获取词典`word2index_dict`。
    word2index_dict = get_word_dict()
    # 将输入句子中的每个词转换为对应的索引，使用`word2index_dict.get(word, 1)`进行查找，若找不到则使用索引1表示未知词。
    sentence = [word2index_dict.get(word, 1) for word in sentence]
    # 如果句子长度小于50，则在句子末尾填充0，使其长度为50。
    if len(sentence) < 50:
        sentence += [0] * (50 - len(sentence))
    # 如果句子长度超过50，则截取前50个词。
    else:
        sentence = sentence[:50]
    # 将处理后的句子转换为张量，并添加一维作为批次维度，使用`torch.unsqueeze`函数。
    datasets = torch.unsqueeze(torch.LongTensor(sentence), dim=0).to(device)
    # 将张量移动到设备`device`上。
    model = torch.load(model_path).to(device)
    # 加载训练好的模型`model`，并设置为评估模式，即调用`model.eval()`。
    model.eval()
    # 使用`torch.no_grad()`上下文管理器，表示不需要计算梯度。
    with torch.no_grad():
        # 对输入句子进行前向计算，得到预测结果`labels_pred`。
        # 使用`torch.argmax`函数找到预测结果中最大值所在的索引，得到预测标签的索引。
        # 根据预测标签的索引从标签列表`labels`中获取预测结果的文本表示。
        labels_pred = torch.argmax(model(datasets), dim=-1).cpu().numpy()[0]
    # 打印预测结果。
    print(f"{labels[labels_pred]}")

    # 返回预测结果的文本表示。
    return labels[labels_pred]
# 综上所述，该代码实现了对输入句子进行预测，并输出预测结果的文本表示。


# 该代码实现了一个函数`module_evaluation()`，用于加载训练好的模型，并对测试数据进行评估。
def module_evaluation():
    # 使用`torch.load`函数加载训练好的模型，并将其移动到GPU上（设备为"cuda"）。
    model = torch.load("bilstm_model_epoch10.pth").to("cuda")
    # 使用`get_word_dict()`函数获取词典`word2index_dict`。
    # 创建`test_loader`，使用`DataLoader`将测试数据集包装成数据加载器。其中，使用`DataGenerator`构建数据集，并传入词典和测试数据的路径。
    # 设置`shuffle`参数为`False`，表示不对数据进行洗牌。
    # 设置`batch_size`参数为64，表示每个批次包含64个样本。
    test_loader = DataLoader(DataGenerator(get_word_dict(), root="test.txt"), shuffle=False, batch_size=64)
    # 调用`get_test_result`函数，传入加载的模型、测试数据加载器和设备"cuda"，对模型在测试数据上进行评估。
    get_test_result(model, test_loader, "cuda")

    return
# 综上所述，该代码实现了加载训练好的模型，并在测试数据上进行评估。它首先加载模型，然后使用数据加载器加载测试数据，最后调用`get_test_result`函数计算评估指标并输出结果。


if __name__ == '__main__':
    # get_train_val_txt()
    # train()
    # module_evaluation()
    rumor = input("请输入待检测话题\n")
    predict(rumor)