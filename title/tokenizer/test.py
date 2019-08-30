from simplifier import Simplifier
from tokenizer import Tokenizer
from lowercase_and_remove_accent import run_strip_accents
s = Simplifier('tokenizer/zh_mapping.txt')
t = Tokenizer('tokenizer/spiece_all_bpe/spiece.all.bpe.130000.lower.model',
    'tokenizer/lg.all.voc',
    12
)

#text = 'Python でプログラムする際にとても便利な Anaconda ですが、長く使っているとコンピュータのハードドライブの要領を圧迫し始めました。 本稿では、そんな Anaconda の不要になったファイルを整理・削除して、ハードドライブをクリーンな状態に保つ方法を解説します。'
#text = 'Gesundheitsgefährdenden Schimmel effektiv und nachhaltig bekämpfen.'
text = 'Vi håber på at se dig!'
#text = 'Sisäseinät, talon katto vai terassin kaiteet? Mieti maalia valitessasi, millaiseen pintaan ja käyttöön maali tulee. Tikkurilan laajasta valikoimasta löydät huippulaatuiset tuotteet niin kodin sisätilojen, talon ulkopintojen kuin pihakalusteidenkin maalaamiseen.'

print (text)

simple = s.simplify(text)
print (simple)

accents = run_strip_accents(simple)
print (accents)
tokens = t.tokenize(accents)
print (tokens)

tokens = t.tokenize(simple)
print (tokens)


#tokens = t.tokenize(text)
#print (tokens)

accents = run_strip_accents(tokens)
print (accents)

ids = t.token_to_id(accents)
print (ids)