import os

class RNNState():
    '''
    保存循环神经网络每一步的状态
    优化：
    创建多个类，尝试归并到一个类
    '''
    def __init__(self, cell, hidden=None):
        '''
        若为序列的第一个状态，则初始化hidden为0
        调用next，返回新状态，tuple
        调用class（），返回上一个点的输出hidden[0]
        :param cell:
        :param hidden:
        '''
        self.cell = cell#计算单元
        self.hidden = hidden#循环神经网络的中间态
        if not hidden:#如果为None，则创建一个tuple,[0]-输出,[1]-状态
            self.hidden = Variable(torch.zeros(1, self.cell.hidden_size)), \
                          Variable(torch.zeros(1, self.cell.hidden_size))

    def next(self, input):#传递至下一个cell进行计算
        return RNNState(self.cell, self.cell(input, self.hidden))

    def state(self):
        return self.hidden[1]

    def __call__(self):#获取最后的output
        return self.hidden[0]
      
      
class LSTMModule(nn.Module):
      def __init__(self,vocab,pos,w2i,options):
        super(MSTParserLSTMModel, self).__init__()  # 调用父构造函数
        random.seed(1)
        
        #激活方法区
        self.activations = {'tanh': F.tanh, 'sigmoid': F.sigmoid, 'relu': F.relu,
                            # Not yet supporting tanh3
                            # 'tanh3': (lambda x: nn.Tanh()(cwise_multiply(cwise_multiply(x, x), x)))
                            }
        self.activation = self.activations[options.activation]
        #
        #标记参数
        self.blstmFlag = options.blstmFlag
        self.labelsFlag = options.labelsFlag
        self.costaugFlag =options.costaugFlag
        self.bibiFlag = options.bibiFlag
        #
        #词嵌入参数
        self.ldims = options.lstm_dims
        self.wdims = options.wembedding_dims
        self.pdims = options.pembedding_dims
        #
        #层数
        self.layers = options.lstm_layers
        #字典
        self.wordsCount = vocab#统计词语出现的次数
        self.vocab = {word:ind+3 for (word,ind) in w2i.items()}
        self.pos = {pos:ind+3 for ind,pos in enumerate(pos)}
        #
        #是否采用双层双向LSTM模型
        if self.bibiFlag:
          self.builders = [nn.LSTMCell(self.wdims + self.pdims, self.ldims),#第一层
                             nn.LSTMCell(self.wdims + self.pdims, self.ldims)]#第二层
          self.bbuilders = [nn.LSTMCell(self.ldims * 2, self.ldims),
                              nn.LSTMCell(self.ldims * 2, self.ldims)]
        else:
          assert self.layers ==1,#单层双向lstm
          self.builders = [  # 构造前向和反向网络
                nn.LSTMCell(self.wdims + self.pdims + self.edim, self.ldims),
                nn.LSTMCell(self.wdims + self.pdims + self.edim, self.ldims)]
        #
        
        
        
        
        
