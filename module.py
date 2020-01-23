import os

'''
firstlayer-lstmcell: (wdim+pdim,ldim)
secondlayer-lstmcell: (ldim*2, ldim)
紧接lstm后的头特征: (ldim*2,hidden_units)
紧接lstm后的依赖特征: (ldim*2,hidden_units)
hiddenlayer2: (hidden_units,hidden2_units) 
'''
get_data = (lambda x:x.data.cpu()) if use_gpu else (lambda x:x.data)

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
      def __init__(self,vocab,pos,rels,w2i,options):
        super(LSTMModule, self).__init__()  # 调用父构造函数
        random.seed(2100)
        
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
        self.rels = rels
        self.bibiFlag = options.bibiFlag
        #
        #词嵌入层
        self.ldims = options.lstm_dims
        self.wdims = options.wembedding_dims
        self.pdims = options.pembedding_dims
        self.wlookup = nn.Embedding(len(vocab) + 3, self.wdims)  # 词嵌入层
        self.plookup = nn.Embedding(len(pos) + 3, self.pdims)  # 词性嵌入层
        #
        #层数
        self.layers = options.lstm_layers
        #字典
        self.wordsCount = vocab#统计词语出现的次数
        self.vocab = {word:ind+3 for (word,ind) in w2i.items()}
        self.pos = {pos:ind+3 for ind,pos in enumerate(pos)}
        self.vocab['*PAD*'] = 1  # 句子的pad
        self.pos['*PAD*'] = 1
        self.vocab['*INITIAL*'] = 2
        self.pos['*INITIAL*'] = 2
        #
        #是否采用双层双向LSTM模型
        if self.bibiFlag:
          self.builders = [nn.LSTMCell(self.wdims + self.pdims, self.ldims),#第一层
                             nn.LSTMCell(self.wdims + self.pdims, self.ldims)]
          self.bbuilders = [nn.LSTMCell(self.ldims * 2, self.ldims),#第二层
                              nn.LSTMCell(self.ldims * 2, self.ldims)]
        else:
          assert self.layers ==1,#单层双向lstm
          self.builders = [  # 构造前向和反向网络
                nn.LSTMCell(self.wdims + self.pdims, self.ldims),
                nn.LSTMCell(self.wdims + self.pdims, self.ldims)]
        
        for i,b in enumerate(self.builders):
            self.add_module('builder%i'%i,b)
        if hasattr(self, 'bbuilders'):
            for i,b in enumerate(self.bbuilders):
                self.add_module('bbuilder%i'%i,b)
        #
        #注意力层模型
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=(self.wdims + self.pdims)*2,hhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        #
        #隐藏层设置
        self.hidden_units = options.hidden_units#第一个隐含层的输出维度
        self.hidden2_units = options.hidden2_units#第二个隐含层的输出维度
        self.hidLayerFOH = Parameter((self.ldims * 2, self.hidden_units))  # 紧接lstm后面
        self.hidLayerFOM = Parameter((self.ldims * 2, self.hidden_units))  # 可训练参数层
        self.hidBias = Parameter((1, self.hidden_units))  # 可训练参数层
        if self.hidden2_units>0:#多一层隐藏层
            self.hid2Layer = Parameter((self.hidden_units, self.hidden2_units))
            self.hid2Bias = Parameter((1,self.hidden2_units))
        if self.labelsFlag:# 需要预测依赖的类型
            self.rhidLayerFOH = Parameter((2 * self.ldims, self.hidden_units))
            self.rhidLayerFOM = Parameter((2 * self.ldims, self.hidden_units))
            self.rhidBias = Parameter((1,self.hidden_units))
            if self.hidden2_units>0:#多一层隐藏层
                self.rhid2Layer = Parameter((self.hidden_units, self.hidden2_units))
                self.rhid2Bias = Parameter((1,self.hidden2_units))
            self.routLayer = Parameter(#预测依赖种类
                (self.hidden2_units if self.hidden2_units > 0 else self.hidden_units, len(self.irels)))
            self.routBias = Parameter((1,len(self.irels)))#输出维度为预测种类的个数
        #
        #输出层
        self.outLayer = Parameter(  # 将输出的维度压缩成1,预测依赖关系
            (self.hidden2_units if self.hidden2_units > 0 else self.hidden_units, 1))
        #
        
      def __getExpr(self, sentence, i, j):
        if sentence[i].headfov is None:# sentence[i]作为头部的特征值
            sentence[i].headfov = torch.mm(cat([sentence[i].lstms[0], sentence[i].lstms[1]]),
                                           self.hidLayerFOH)
        if sentence[j].modfov is None:# sentence[j]作为依赖的特征值
            sentence[j].modfov = torch.mm(cat([sentence[j].lstms[0], sentence[j].lstms[1]]),
                                          self.hidLayerFOM)
        if self.hidden2_units>0:
            output = torch.mm(
                self.activation(
                    self.hid2Bas+
                    torch.mm(self.activation(sentence[i].headfov + sentence[j].modfov + self.hidBias),
                             self.hid2Layer)
                ),self.outLayer
            )
        else:
            outpur = torch.mm(
                self.activation(
                    sentence[i].headfov + sentence[j].modfov + self.hidBias),
                self.outLayer)  # + self.outBias
        return output#返回sentence[i],sentence[j]依赖特征值
    
      def __evaluate(self,sentence):
            exprs = [[self.__getExpr(sentence, i, j)#NxN的依赖矩阵
                  for j in range(len(sentence))]
                 for i in range(len(sentence))]
            scores = np.array([[get_data(output).numpy()[0, 0] for output in exprsRow] for exprsRow in exprs])#exprs的numpy形式
            return scores, exprs
      def __evaluateLabel(self,sentence,i,j):
        if sentence[i].rheadfov is None:
            sentence[i].rheadfov = torch.mm(cat([sentence[i].lstms[0], sentence[i].lstms[1]]),
                                            self.rhidLayerFOH)

        if sentence[j].rmodfov is None:
            sentence[j].rmodfov = torch.mm(cat([sentence[j].lstms[0], sentence[j].lstms[1]]),
                                           self.rhidLayerFOM)
        if self.hidden2_units > 0:
            output = torch.mm(
                self.activation(
                    self.rhid2Bias +
                    torch.mm(
                        self.activation(sentence[i].rheadfov + sentence[j].rmodfov + self.rhidBias),
                        self.rhid2Layer
                    ) +
                    self.routBias),self.routLayer
            )
        else:
            output = torch.mm(
                self.activation(sentence[i].rheadfov + sentence[j].rmodfov + self.rhidBias),
                self.routLayer

            ) + self.routBias
        return get_data(output).numpy()[0], output[0]#返回numpy数值
        
      def predict(self,sentence):
        '''
        修改处：
        1. 增加注意力机制
        '''
        
        wordints = [int(self.vocab.get(entry.word,0)) for entry in sentence]
        posints = [int(self.pos[entry.pos]) for entry in sentence]
        wordvecs = self.wlookup(wordints)
        posvecs = self.plookup(posints)
        inputs =  torch.cat((wordvecs,posvecs),dim=-1)
        attention_res = torch.squeeze(self.transformer_encoder(inputs))
        
        for i,entry in enumerate(sentence):
            #wordvec = self.wlookup(scalar(int(self.vocab.get(entry.word,0)))) if self.wdims>0 else None
            #posvec = self.plookup(scalar(int(self.pos[entry.pos]))) if self.pdims > 0 else None
            #entry.vec = cat([wordvec,posvec])
            entry.vec = attention_res[i]
            entry.lstms = [entry.vec,entry.vec]
            entry.headfov = None
            entry.modfov = None
        if self.blstmFlag:#加入lstm层,该算法此处只使用了一层层lstm
            
            '''
            此时的lstm[0],lstm[1]只是经过了一层lstm的特征
            '''
            lstm_forward = RNNState(self.builders[0])
            lstm_backward = RNNState(self.builders[1])

            for entry,rentry in zip(sentence,reversed(sentence)):
                lstm_forward = lstm_forward.next(entry.vec)
                lstm_backward = lstm_backward.next(rentry.vec)
                entry.lstms[1] = lstm_forward()
                rentry.lstms[0] = lstm_backward()
                
        if self.bibiFlag:
            lstm_forward = RNNState(self.bbuilders[0])
            lstm_backward = RNNState(self.bbuilders[1])
            
            for entry in sentence:
                entry.vec = torch.cat((entry.lstm[1],entry.lstm[0]),dim=-1)
                
            for entry,rentry in zip(sentence,reversed(sentence)):
                lstm_forward = lstm_forward.next(entry.vec)
                lstm_backward = lstm_backward.next(rentry.vec)
                entry.lstms[1] = lstm_forward()
                rentry.lstms[0] = lstm_backward()
                
                
        
        scores,exprs = self.__evaluate(sentence)
        heads = decoder.parse_proj(scores)
        for entry,head in zip(sentence,heads):
            entry.pred_parent_id = head
            entry.pred_relation = '_'
        if self.labelsFlag:
            for modifier,head in enumerate(heads[1:]):
                scores, exprs = self.__evaluateLabel(sentence, head, modifier + 1)
                sentence[modifier+1].pred_relation = self.rels[enumerate(scores),key=itemgetter(1)[0]]
            
      def forward(self,sentence,errs,lerrs):
        
          wordints = [int(self.vocab.get(entry.word,0)) for entry in sentence]
          posints = [int(self.pos[entry.pos]) for entry in sentence]
          wordvecs = self.wlookup(wordints)
          posvecs = self.plookup(posints)
          inputs =  torch.cat((wordvecs,posvecs),dim=-1)
          attention_res = torch.squeeze(self.transformer_encoder(inputs))
        
          for i,entry in enumerate(sentence):
            #wordvec = self.wlookup(scalar(int(self.vocab.get(entry.word,0)))) if self.wdims>0 else None
            #posvec = self.plookup(scalar(int(self.pos[entry.pos]))) if self.pdims > 0 else None
            #entry.vec = cat([wordvec,posvec])
            entry.vec = attention_res[i]
            entry.lstms = [entry.vec,entry.vec]
            entry.headfov = None
            entry.modfov = None
            entry.rheadfov = None
            entry.rmodfov = None
            
          if self.blstmFlag:
            lstm_forward = RNNState(self.builders[0])
            lstm_backward = RNNState(self.builders[1])
            for entry,rentry in zip(sentence,reversed(sentence)):
                lstm_forward = lstm_forward.next(entry.vec)
                lstm_backward = lstm_backward.next(rentry.vec)
                entry.lstms[1] = lstm_forward()
                rentry.lstms[0] = lstm_backward()
           
          if self.bibiFlag:
            lstm_forward = RNNState(self.bbuilders[0])
            lstm_backward = RNNState(self.bbuilders[1])
            
            for entry in sentence:
                entry.vec = torch.cat((entry.lstm[1],entry.lstm[0]),dim=-1)
                
            for entry,rentry in zip(sentence,reversed(sentence)):
                lstm_forward = lstm_forward.next(entry.vec)
                lstm_backward = lstm_backward.next(rentry.vec)
                entry.lstms[1] = lstm_forward()
                rentry.lstms[0] = lstm_backward()
        
        
          scores,exprs = self.__evaluate(sentence,True)#先经过双向lstm，在➡由多层感知机输出结果
          heads =decoder.parse_proj(scores)  
          gold = [entry.parent_id for entry in sentence] #事先标注好的父节点下标
            
          if self.labelsFlag:
            for modifier, head in enumerate(gold[1:]):
                rscores, rexprs = self.__evaluateLabel(sentence, head, modifier + 1)
                goldLabelInd = self.rels[sentence[modifier + 1].relation]
                wrongLabelInd = \#不是正确依赖种类中概率最大的索引
                max(((l, scr) for l, scr in enumerate(rscores) if l != goldLabelInd), key=itemgetter(1))[0]
                #此处依赖种类的损失值规则有待调整
                if rscores[goldLabelInd] < rscores[wrongLabelInd] + 1:#如果错误的分数加1大于正确的得分
                    lerrs += [rexprs[wrongLabelInd] - rexprs[goldLabelInd]]
          e = sum([1 for h,g in zip(gold[1:],heads[1:]) if h!=g])#父节点预测出错的个数
          constant_zero = torch.tensor([[0.]],requires_grad=False)
          if e>0:
            errs+=[torch.max((exprs[h][i]-exprs[g][i]),constant_zero)[0][0] for i,(h,g) in enumerate(heads,gold) if h!=g]#errs是每一句话的真实损失值
          return e#返回错误的个数

def get_optim(opt,parameters):
  if opt =='sgd':
      return optim.SGD(parameters,lr=opt.lr)
  elif opt =='adam':
      return optim.Adam(parameters)  

class WrapperParser:
      def __init__(self,vocab,pos,w2i,options):
        model = LSTMModule(vocab,pos,rels,w2i,options)
        self.model = model.cuda() if use_gpu else model
        self.trainer = get_optim(options.optim,self.model.parameters())
        
      def predict(self,conll_path):
        with open(conll_path,'r') as r:
            for i,sentence in enumerate(read_conll(r)):
                conll_sentence = [entry for entry in sentence if isinstance(entry,utils.ConllEntry)]
                self.model.predict(conll_sentence)
                yield conll_sentence
      def save(self,fn):
        tmp = fn + '.tmp'
        torch.save(self.model.state_dict(), tmp)
        shutil.move(tmp, fn)
        
      def load(self, fn):
        self.model.load_state_dict(torch.load(fn))  
        
      def train(self, conll_path):
        print(torch.__version__)
        batch = 1
        eloss = 0
        etotal = 0
        start = time.time()
        with open(conll_path, 'r') as r:
            data = read_conll(r)
            errs = []
            lerrs = []
            #lerrs = []#标签预测的损失值，暂时不用
            for i,sentence in enumerate(data):
                if i%100==0 and i!=0:
                    print('='*20)
                    print('Processing sentence number: {}'.format(i))
                    print('Errors: {}'.format(eloss/etotal))
                    print('Time: {}'.format(time.time()-start))
                    start = time.time()
                    eloss=0
                    etotal=0
                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                e = self.model.forward(conll_sentence,errs,lerrs)
                eloss+=e
                etotal+=len(conll_sentence)
                
                if i%batch==0 and i!=0:
                    self.trainer.zero_grad()
                    backerrs = sum(errs)+sum(lerrs)
                    backerrs.backward()
                    self.trainer.step()
                    errs = []
                    lerrs= []
            if len(errs)>0:
                self.trainer.zero_grad()
                backerrs = sum(errs)+sum(lerrs)
                backerrs.backward()
                self.trainer.step()
                errs = []
                lerrs = []
                
            print('One epoch finished!!!')
            print('Accuracy:{}'.format(1-eloss/etotal))
                
                
          
        
        
