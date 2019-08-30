#!/usr/bin/env python
#coding=utf-8
import sys

class Simplifier:
    def __init__(self, f):
        self.OLD_TO_NEW = {}
        self.load_data(f)
    def load_data(self, f):
        lines = open(f, encoding='utf-8').readlines()
        for line in lines:
            arr = line.split("\t")
            if len(arr) == 2:
                self.OLD_TO_NEW[arr[0].strip()] = arr[1].strip()
        sys.stderr.write("len(OLD_TO_NEW): %d \n" % len(self.OLD_TO_NEW))

    def simplify(self, line):
        arr = [i for i in line]
        for i in range(len(arr)):
            if arr[i] in self.OLD_TO_NEW:
                arr[i] = self.OLD_TO_NEW[arr[i]]
        return "".join(arr)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        s = Simplifier(sys.argv[1])
    else:
        sys.stderr.write("error args\n")
        exit(1)
    
    for line in sys.stdin:
        line = line.strip()
        line = line.replace("　", " ")
        while line.find("  ") >= 0:
            line = line.replace("  ", " ")
        print(s.simplify(line))
