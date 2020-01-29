import os
import pickle
fn = '_data/train2.txt'
tofile = '_data/train2.pk'
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
        values = [str(self.id),str(self.old_id),self.word,self.pos,str(self.parent_id),self.entity,self.relation]
        return '\t'.join(values)

def _readline(fn:str,code="utf-8")->str:
    #mode = rmode(fn)
    with open(fn,"rb") as fr:
        for i in fr:
            #if mode=="rb":
            i = i.decode(encoding=code)
            yield i

def rmode(fn:str)->str:
    if fn.endswith(".txt"):
        return "r"
    else:
        return 'rb'

def wpk(fn,tofn,code="utf-8"):
    tofile = open(tofn,'wb')
    old_id = ""
    id = -1
    word = ""
    pos = ""
    parent_id = ""
    entity = ""
    sentence = []
    for idx,line in enumerate(_readline(fn,code)):
        if line.strip() == "":
            sentence.append(Entry(id, old_id, word, pos, parent_id, entity))#将最后一个字加入
            for i in sentence:
                if i.parent_id != '_':
                    try:
                        tar = next(filter(lambda x: x.old_id == i.parent_id, sentence))
                    except:
                        print('没有找到对应序号的实例')
                        print(idx)

                    i.parent_id = tar.id
            pickle.dump(sentence,tofile)
            old_id = ""
            id = -1
            word = ""
            pos = ""
            parent_id = ""
            entity = ""
            sentence = []
            continue

        items = line.strip().split('\t')  # [item,...]
        assert len(items) == 6, idx
        if items[2] == "pr":
            if id != -1:
                sentence.append(Entry(id, old_id, word, pos, parent_id, entity))
            id += 1
            old_id = items[0]
            entity = items[3]
            word = items[1]
            pos = items[5]
            parent_id = items[4]
        else:
            word += items[1]
    tofile.close()
def rpk(pk):
    with open(pk,'rb') as fr:
        while(True):
            try:
                data = pickle.load(fr)
            except:
                print('read over')
                break
            yield data


if __name__ == '__main__':
    pass






