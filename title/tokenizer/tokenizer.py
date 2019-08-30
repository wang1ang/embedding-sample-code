#!/usr/bin/env python
#coding=utf-8
import sys
from tokenizer.dictionary import Dictionary

class Tokenizer:
    def __init__(self, model_path, voc_path, max_length):
        self.pun_list = [" ", "!", "#", "$", "%", "&", "'", "(", ")", "*", \
            ",", "-", ".", "/", ":", ";", "?", "@", "[", "\"", "\\", \
            "\t", "]", "^", "_", "`", "{", "|", "}", "~", "¨", "·", "—", \
            "‘", "’", "“", "”", "…", "─", "╔", "╝", "●", "♂", "　", "～", \
            "、", "。", "《", "》", "「", "」", "『", "』", "【", "】", "〔", "〕", \
            "゛", "ゝ", "﹎", "﹖", "！", "（", "）", "，", "．", "：", "；", "？"]
        self.load_model(model_path)
        self.dico = Dictionary.read_vocab(voc_path)
        self.max_length = max_length

    def replace_pun(self, text):
        for p in self.pun_list:
            text = text.replace(p, " %s " % p)
        while text.find("  ") >= 0:
            text = text.replace("  ", " ")
        return text

    def load_model(self, pth):
        import sentencepiece as spm
        #sys.stderr.write("loading seg model: %s\n" % pth)
        #sys.stderr.flush()
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(pth)

    def get_seg_line(self, text):
        arr = self.sp.EncodeAsPieces(text)
        arr = arr[:self.max_length]
        text = " ".join(arr)
        text = text.replace("▁", " ").strip()
        #text = text.replace("▁", "_").strip()
        while text.find("  ") >= 0:
            text = text.replace("  ", " ")
        return text
    @staticmethod
    def convert_num(text):
        arr = text.split()
        for i in range(len(arr)):
            if arr[i].replace(".", "").replace("%", "").isdigit():
                arr[i] = "<num>"
        return " ".join(arr)

    def tokenize(self, text):
        lower = text.strip().lower()
        seg = self.get_seg_line(lower)
        seppun = self.replace_pun(seg)
        nonum = self.convert_num(seppun)
        return nonum

    def token_to_id(self, text):
        line = text.split(' ')[:self.max_length]
        return Dictionary.index_data(line, self.dico)[:self.max_length]


if __name__ == "__main__":
    sys.stderr.flush()
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: %s <lang_model_path>\n" % sys.argv[0])
        exit(1)
    model_path = sys.argv[1]
    model = Tokenizer(model_path)
    sys.stderr.flush()
    for line in sys.stdin:
        line = line.strip()
        print(model.tokenize)
