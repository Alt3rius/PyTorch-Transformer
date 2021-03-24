import sentencepiece

def generator_en():
    with open('data/para_crawl/en-pl.txt', encoding='utf-8') as f:
        line = f.readline()
        while line:
            line = line.split('\t')
            yield line[0]
            line = f.readline()

gen_en = generator_en()

sentencepiece.SentencePieceTrainer.train(sentence_iterator=gen_en, model_prefix='en', vocab_size=50000, user_defined_symbols=['<s>', '</s>', '<pad>'])

def generator_pl():
    with open('data/para_crawl/en-pl.txt', encoding='utf-8') as f:
        line = f.readline()
        while line:
            line = line.split('\t')
            yield line[1]
            line = f.readline()

gen_pl = generator_pl()

sentencepiece.SentencePieceTrainer.train(sentence_iterator=gen_pl, model_prefix='pl', vocab_size=50000, user_defined_symbols=['<s>', '</s>', '<pad>'])

