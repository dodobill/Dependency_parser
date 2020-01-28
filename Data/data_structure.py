import os


tool1 = lambda x:str(x) if type(x)!=str else x

class ConllEntry:
  def __init__(self,id,word,pos,parent_id,relation):
    self.id = int(id)
    self.word = word#str
    self.norm = normalize(word)#经过变化的word
    self.pos = pos.upper()
    assert type(parent_id)==int
    self.parent_id = parent_id
    self.relation = relation
    
    self.pred_parent_id = None
    self.pred_relation = None
  def __str__(self):  
    values = map(tool1,[self.id, self.word, self.norm, self.pos, self.parent_id])
    print("编号\t原词\t新词\t词性\t父亲编号\n")
    return '\t'.join(['_' if v is None else v for v in values])
  
  
def read_conll(fh):
    root = utils.ConllEntry(-1, '*root*','ROOT-POS', -1,None)
    tokens = [root]
    for line in fh:
        tok = line.strip().split('\t')
        if not tok or line.strip() == '':
            if len(tokens)>1: yield tokens
            tokens = [root]
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                tokens.append(line.strip())
            else:
                tokens.append(utils.ConllEntry(int(tok[0]), tok[1], tok[3], tok[6],tok[7]))
    if len(tokens) > 1:
        yield tokens
     
    
    
    
