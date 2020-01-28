import os
fn = '_data/train.txt'

'''
以下仅规范化一句话
改进：模块化，将大段代码转化成函数
'''
class Entry:
    def __init__(self,id,old_id,word,pos,parent_id,entity,relation='_'):
        self.old_id=old_id
        self.id = id
        self.word = word
        self.pos = pos
        self.parent_id =parent_id
        self.entity = entity
        self.relation = relation
    def __str__(self):
        values = [str(self.id),self.word,self.pos,str(self.parent_id),self.entity,self.relation]
        return '\t'.join(values)

def _readline(fn:str,code="utf-8")->str:
    mode = rmode(fn)
    with open(fn,mode) as fr:
        for i in fr:
            if mode=="rb":
                i = i.decode(encoding=code)
            yield i

def rmode(fn:str)->str:
    if fn.endswith(".txt"):
        return "r"
    else:
        return 'rb'

if __name__ == '__main__':
    old_id=""
    id = -1
    word = ""
    pos = ""
    parent_id = ""
    entity = ""
    sentence = []
    for line in _readline(fn):
        if line.strip()=="":
            for i in sentence:
                if i.parent_id!='_':
                    try:
                        tar = next(filter(lambda x:x.old_id==i.parent_id,sentence))
                    except:
                        print('没有找到对应序号的实例')
                    i.parent_id=tar.id
            print("完成第一句")
            break
    
        items = line.strip().split('\t')#[item,...]
        assert len(items)==6,items
        if items[2]=="pr":
            if id!=-1:
                sentence.append(Entry(id,old_id,word,pos,parent_id,entity))
            id += 1
            old_id=items[0]
            entity = items[3]
            word = items[1]
            pos = items[5]
            parent_id = items[4]
        else:
            word+=items[1]

