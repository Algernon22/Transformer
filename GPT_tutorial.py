#%%
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.nn.functional as F
import random
import textwrap
from torch.ao.nn.quantized import Softmax
from torch.onnx.symbolic_opset9 import masked_fill

# 超参数
# 文件目录
file_name = r"./hong_lou_meng.txt"

# 共有几组
BATCH_SIZE = 8

# 一组几个token
BLOCK_SIZE = 32

# 每个token对应的向量维度
EMBEDDING_SIZE = 32

NUM_HEADS = 8

HEAD_SIZE = EMBEDDING_SIZE // NUM_HEADS

# 最大生成下文数量
MAX_NEW_TOKENS = 128

# 学习率
LEARNING_RATE = 1e-3

#
NUM_LAYER = 6

# 训练次数
MAX_ITERS = 5000

# loss输出频率
EVAL_INTERVAL = int(MAX_ITERS // 10)

# 每EVAL_INTERVAL词，就会做EVAL_ITERS次损失测评
EVAL_ITERS = 200

DROPOUT_RATE = 0.2

WARP_WIDTH = 50

device = "cuda" if torch.cuda.is_available() else "cpu"

# 读取训练材料
with open(file_name, "r", encoding="utf-8") as f:
    text = f.read()

# 形成字典
# 设置集合实现去重；设置列表从而应用排序；创建一个有序不重复的字典
chars = sorted(list(set(text)))

# 词汇表的大小
# 嵌入层将输入字符转换为向量表示，输出层将模型预测的向量转换回字符。vocab_size 确保模型处理的输入和输出都在定义的字符集范围内。
vocab_size = len(chars)

# 形成token（token就是给唯一存在的字符编号） --- “分词器”
# stoi 是一个字典，其中每个键是字符 (ch)，每个值是该字符的索引 (i)。
# 例如，如果 chars 是 ['a', 'b', 'c']，那么 stoi 将是 {'a': 0, 'b': 1, 'c': 2}
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# lambda：匿名函数，相当于快速定义一个小函数 等价于
# def encode(str1):
#    return [stoi[ch] for ch in str1]
# 遍历字符串 str1 的每个字符 ch，并用 stoi 字典将其转换为整数
encode = lambda str1: [stoi[ch] for ch in str1]

# "".join(...)：将列表中的多个字符拼接成一个字符串
# 遍历整数列表 list1，用 itos 字典将每个整数转换为字符。
decode = lambda list1: "".join([itos[i] for i in list1])

# 把数据分为训练数据和验证数据，如果训练数据和验证数据一致，则会导致 过拟合（over fitting），它意味着模型过度记忆了训练数据的细节，导致泛化能力差 --- 学而不思
# 将整数序列转换为PyTorch张量，并指定数据类型为长整型
data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]

# 分批；将文本序列切分为固定长度的片段，组成批次
def get_batch(split):
    data = train_data if split == "train" else test_data

    # 随机生成 BATCH_SIZE 个起始索引，确保每个索引 + BLOCK_SIZE 后仍不越界
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))

    # torch.stack():合并张量：将多个序列（如BATCH_SIZE个[BLOCK_SIZE]的序列）合并为一个形状为[BATCH_SIZE, BLOCK_SIZE]的张量。
    # 从 data 中取出 BATCH_SIZE 个长度为 BLOCK_SIZE 的连续序列
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])

    # 从 data 中取出对应 x 的下一个字符组成的序列作为目标 y（右移一位）
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])

    x, y = x.to(device), y.to(device)
    return x, y

# 损失测评
@torch.no_grad()
def evaluate_loss(model):
    out = {}

    # 把模型转化为evaluate模式（默认模式为train）
    model.eval()

    for split in ["train", "test"]:

        # 建立一个初始值为0的容器，用于存储loss值
        losses = torch.zeros(EVAL_ITERS)

        for k in range(EVAL_ITERS):

            # split是一个字符串，用来控制get_batch（）函数的行为
            X, Y = get_batch(split)

            logits, loss = model(X, Y)
            losses[k] = loss.item()

        # out是包含两个元素的字典一个是train，一个是test，每个元素对应一个loss的平均值
        out[split] = losses.mean()

    # 再转化为训练模式
    model.train()

    return out

 # B. self attention
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        # B.a. 通过三个线性变换(输入EMBEDDING_SIZE维向量，输出head_size维向量)分别得到 Query、Key 和 Value
        # key = A_k*x、query = A_q*x value = A_v*x，其中A_i代表了权重矩阵
        self.key = nn.Linear(EMBEDDING_SIZE, head_size, bias=False)
        self.query = nn.Linear(EMBEDDING_SIZE, head_size, bias=False)
        self.value = nn.Linear(EMBEDDING_SIZE, head_size, bias=False)

        # 生成不可训练的矩阵（矩阵内容固定）
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x):
        B, T, C = x.shape

        # B.a.
        k = self.key(x)
        q = self.query(x)

        # B.b. 生成注意力矩阵
        # 注意力得分反映了一个token对其他tokens的"关注"程度。这个得分通过计算 Query 和 Key 的矩阵点积来得到,在数学运算中,向量A dot product 越是相近的向量,结果要大于dot product距离更远的向量
        # / (k.shape[-1] ** 0.5) 其作用是缩放因子
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5

        # B.c. 通过掩码矩阵乘法来让神经网络利用上文信息，见图片img.png
        wei = wei.masked_fill(self.tril == 0, float("-inf"))
        # 行优先做softmax,形成概率
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)

        # B.c. Softmax 矩阵和V相乘，得到最终的输出
        # 做矩阵乘法，实现前文利用
        v = self.value(x)
        out = wei @ v

        return out

# C. Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()

        # C.a. 将输入x分别传递到 num_heads 个不同的 Self-Attention 中
        # 创建多个注意力头
        self.heads = nn.ModuleList(
            # C.b. 调用self attention
            [Head(head_size) for _ in range(num_heads)]
        )

        self.proj = nn.Linear(num_heads * head_size, EMBEDDING_SIZE)
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x):
        # C.c. 拼接多头的生成向量
        out = torch.cat(
            [h(x) for h in self.heads],
            dim = -1
        )

        # C.d. 把拼接向量放入linear层,得到最终结果
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.net = nn.Sequential(
            # 通过非线性变换增强模型表达能力
            nn.Linear(embedding_size, embedding_size * 4),
            nn.ReLU(),
            # 先扩展维度再压缩，帮助学习更复杂的特征交互
            nn.Linear(embedding_size * 4, embedding_size),
            # 正则化
            nn.Dropout(DROPOUT_RATE),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(num_heads, HEAD_SIZE)
        self.feed_forward = FeedForward(EMBEDDING_SIZE)
        self.norm1 = nn.LayerNorm(EMBEDDING_SIZE)
        self.norm2 = nn.LayerNorm(EMBEDDING_SIZE)

    def forward(self, x):
        # D.a. Add & Norm
        x = x + self.self_attn(self.norm1(x))

        # E.a. Feed Forward + Add & Norm
        x = x + self.feed_forward(self.norm2(x))

        return x

# 模型设计
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()

        # A. 获取输入句子的每一个单词的表示向量 X
        # A.a. 为字典中的每个token做特征编码
        # 嵌入层：把这小于vocab_size的数字编码为EMBEDDING_SIZE维的向量
        self.token_embedding_table = nn.Embedding(vocab_size, EMBEDDING_SIZE)

        # A.b. 为BLOCK中的每个token做位置编码
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, EMBEDDING_SIZE)

        self.blocks = nn.Sequential(
            *[Block(EMBEDDING_SIZE, NUM_HEADS) for _ in range(NUM_LAYER)],
        )
        self.ln_f = nn.LayerNorm(EMBEDDING_SIZE)
        self.lm_head = nn.Linear(EMBEDDING_SIZE, vocab_size)

    def forward(self, idx, target = None):
        # - idx: 当前输入的token序列，形状为 (B, T)，B是batch大小，T是BLOCK长度
        # - target: 真实标签（此处未使用）

        B, T = idx.shape

        TE = self.token_embedding_table(idx)
        PE = self.position_embedding_table(torch.arange(T, device = device))

        # A.c 把Batch*Block个token编码为带有位置和特征的向量，形成三维向量矩阵，形状：B*T*EMBEDDING_SIZE
        x = PE + TE

        X = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if target is None:
            loss = None
        else:
            # B, T, C = logits.shape
            logits = logits.view(B * T, -1)
            target = target.view(B * T)
            loss = F.cross_entropy(logits, target)
        return logits, loss

    def generate(self, token_sequ, max_new_token):
        # 输入参数：
        # - token_suqe: 初始的token序列，形状为 (B, T)
        # - max_new_token: 需要生成的新token数量

        # 循环生成max_new_token个token预测值
        for _ in range(max_new_token - 1):

            # 截取当前序列的最后 BLOCK_SIZE 个token，保持输入形状为 (B, BLOCK_SIZE)
            # 最后用：其目的是保持输入形状仍然是平面，要不然就变成线
            token_input = token_sequ[:, -BLOCK_SIZE:]

            # 调用forward方法，得到预测的logits（形状为 (B, BLOCK_SIZE, vocab_size)）
            logits, loss = self.forward(token_input)

            # 只取最后一个时间步（即最新预测的位置）的logits
            # 结果形状变为 (B, vocab_size)
            logits = logits[:, -1, :]

            # 将logits转换为概率分布（所有词汇的概率和为1）
            probs = F.softmax(logits, dim = -1)

            # 根据概率分布采样，得到下一个token的编号
            token_next = torch.multinomial(probs, num_samples = 1)

            # 将新生成的token拼接到原序列中，再T维上，扩展序列长度
            token_sequ = torch.cat((token_sequ, token_next), dim = 1)

        new_token = token_sequ[:, -max_new_token:]
        return new_token


# 3. 测试运行

def main():
    print(f"训练内容{file_name}")
    model = DummyModel()
    model = model.to(device)

    # 打印模型参数量
    print(sum(p.numel() for p in model.parameters()) / 1e9, 'B parameters')

    # 设置优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 使用 StepLR 作为学习率调度器
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # 循环训练
    for i in range(MAX_ITERS):

        if i % EVAL_INTERVAL == 0 or i == MAX_ITERS - 1:
            losses = evaluate_loss(model)
            print(f"step {i}, train loss: {losses['train']:.4f}, test loss: {losses['test']:.4f}")

        # 获取批次数据,xb：原始数据，yb：next data
        xb, yb = get_batch("train")

        # 输入xb，计算预测值logits； 通过交叉熵损失函数比较 logits 和 yb，计算标量损失值 loss
        logits, loss = model.forward(xb, yb)

        # 将模型参数的梯度缓冲区置零，避免梯度累积
        optimizer.zero_grad(set_to_none = True)

        # 根据损失 loss，通过反向传播计算模型参数的梯度
        loss.backward()

        # 根据优化器规则和当前梯度，更新模型参数
        optimizer.step()

        # 更新学习率
        scheduler.step()

    print("训练结束")

    start_idx = random.randint(0, len(test_data) - BLOCK_SIZE - MAX_NEW_TOKENS)

    # 上文内容
    # 初始化一个形状为 (1, BLOCK_SIZE) 的全零张量
    context = torch.zeros((1, BLOCK_SIZE), dtype = torch.long, device = device)

    # 从测试数据中截取 BLOCK_SIZE 长度的内容作为输入
    context[0, :] = test_data[start_idx: start_idx + BLOCK_SIZE]

    # 解码并格式化
    context_str = decode(context[0].tolist())
    wrapped_context_str = textwrap.fill(context_str, width = WARP_WIDTH)

    # 真实下文
    real_next_tokens = torch.zeros((1, MAX_NEW_TOKENS), dtype = torch.long, device = device)
    real_next_tokens[0, :] = test_data[start_idx + BLOCK_SIZE: start_idx + BLOCK_SIZE + MAX_NEW_TOKENS]
    real_next_tokens_str = decode(real_next_tokens[0].tolist())
    wrapped_real_next_tokens_str = textwrap.fill(real_next_tokens_str, width = WARP_WIDTH)

    # 生成下文
    generated_tokens = model.generate(context, MAX_NEW_TOKENS)
    generated_str = decode(generated_tokens[0].tolist())
    wrapped_generated_str = textwrap.fill(generated_str, width = WARP_WIDTH)

    print("上文内容：")
    print(wrapped_context_str)
    print("\n")
    print("下文内容：")
    print(wrapped_real_next_tokens_str)
    print("\n")
    print("生成内容：")
    print(wrapped_generated_str)
    print("\n")

main()