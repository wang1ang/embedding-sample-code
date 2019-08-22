import os
from collections import defaultdict

tier0 = {'en'}
tier1 = tier0.union({'es', 'pt', 'ja', 'fr', 'de', 'it', 'cht', 'chs'})
tier2 = tier1.union({'ru', 'ar', 'ko', 'tr', 'hi', 'pl', 'nl', 'ro', 'sv'})
tier4 = tier2.union({'no'})
#tier = defaultdict(lambda: 3)
tier = defaultdict(lambda: 3, {k:0 if k in tier0 else 1 if k in tier1 else 2 for k in tier2})
tier['no'] = 4
def get_files(level=1):
    root = 'f:\\embedding_data'
    d = {}
    for f in os.listdir(root):
        if f.endswith('.txt'):
            suffix = f.split('_')[-1][:-4]
            if not suffix in d:
                d[suffix] = []
            d[suffix].append(os.path.join(root, f))
    if level > 1:
        root = 'f:\\embedding_data\\top24'
        for f in os.listdir(root):
            if f.endswith('.txt'):
                suffix = f.split('_')[-1][:-4]
                if not suffix in d:
                    d[suffix] = []
                d[suffix].append(os.path.join(root, f))
    # for lan in d:
    #     s = 0
    #     for f in d[lan]:
    #         statinfo = os.stat(f)
    #         s += statinfo.st_size
    #     print (lan, s/1024/1024/1024)
    ret = [d[lan] for lan in d if tier[lan] <= level]
    print ('lang_tier='+str(level), 'lang_num=' + str(len(ret)))
    return ret
if __name__ == '__main__':
    files = get_files(1)
    d = {}
    for lan in files:
        for f in lan:
            suffix = f.split('_')[-1][:-4]
            if not suffix in d:
                d[suffix] = 0
            statinfo = os.stat(f)
            d[suffix] += statinfo.st_size
    for k, v in sorted(d.items(), key=lambda item: -item[1]):
        print (k, tier[k], v/1024/1024/1024)
