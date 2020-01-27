# 模型结构功能图解
![Image text](https://github.com/dodobill/Dependency_parser/blob/master/img-folder/module.png)
## 输入层
1. self.wlookup = nn.Embedding(len(vocab) + 3, self.wdims)  # 词嵌入层
2. self.plookup = nn.Embedding(len(pos) + 3, self.pdims)  # 词性嵌入层
3. self.vocab = {word:ind+3 for (word,ind) in w2i.items()}  # 词典
4. self.pos = {pos:ind+3 for ind,pos in enumerate(pos)}  # 词性字典
5. finalinput = torch.cat((wordvecs,posvecs),dim=-1)

## 注意力层
1. 模型
  - self.encoder_layer = nn.TransformerEncoderLayer(d_model=(self.wdims + self.pdims)*2,hhead=4)
  - self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
2. 说明
单个encoder接受输入的维度为finalinput_dim\*2，总共两层encoder叠加

## 循环神经网络层
1. 模型（双向lstm）
  - self.builders = \[nn.LSTMCell(self.wdims + self.pdims, self.ldims),nn.LSTMCell(self.wdims + self.pdims, self.ldims)]
  - self.bbuilders = \[nn.LSTMCell(self.ldims * 2, self.ldims),nn.LSTMCell(self.ldims * 2, self.ldims)]
2. 计算类
  - Class RNNSate
  - next()，返回下一个状态的RNNState实例
  - state()， 返回上个状态的隐藏输入
  - __call__()， 返回上一个状态的输出
3. 备注
  - 需要将bbuilder，builder列表中的单元加入到module中，使用self.add_module
  - 经过循环神经网络后，对一个句子，每一个词单体，entry.lstms\[lstm_forward,lstm_backward]

## 依存关系计算（不预测标签类型）
1. __evaluate(sentence)
  - 调用__getExpr(sentence,i,j)
  - 计算sentence中所有实体之间的依赖值
2. __getExpr(sentence,i,j)
  - 计算sentence\[i]的headfov，作为头的权值
  - 计算sentence\[j]的modfov，作为依赖的权值
  - activation(headfov+modfov)，输出sentence\[i],sentence\[j]之间的关系值
## 依存关系计算（预测标签类型）
1. __evaluateLabel(sentence,i,j)
  - 计算sentence\[i]的rheadfov，作为头的权值
  - 计算sentence\[j]的rmodfov，作为依赖的权值
  - activation(rheadfov+rmodfov)，输出sentence\[i],sentence\[j]之间的关系值
  - 输出标签类型概率分布
  
## Class WrapperParser
  - 封装了设计的module，进行给类操作
  1. predict
  2. save
  3. load
  4. train
  
  
