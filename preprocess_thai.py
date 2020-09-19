import sys

# from transformers import AutoTokenizer
dataset = sys.argv[1]
model_name_or_path = sys.argv[2] # Not use
max_len = int(sys.argv[3])
PROJECT_DATA_PATH = '/content/drive/My Drive/Colab Notebooks/IS_NER/data/03_BERT_Thai_NER'

subword_len_counter = 0

# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, f'{PROJECT_DATA_PATH}/model/bert-master')
import tokenization

bpe_vocab_path = f'{PROJECT_DATA_PATH}/model/th_wiki_bpe/th.wiki.bpe.op25000.vocab'
bpe_model_path = f'{PROJECT_DATA_PATH}/model/th_wiki_bpe/th.wiki.bpe.op25000.model'
tokenizer = tokenization.ThaiTokenizer(vocab_file=bpe_vocab_path, spm_file=bpe_model_path)
max_len -= 2  # my set , must recheck

with open(dataset, "rt") as f_p:
    for line in f_p:
        line = line.rstrip()

        if not line:
            print(line)
            subword_len_counter = 0
            continue

        token = line.split()[0]

        current_subwords_len = len(tokenizer.tokenize(token))

        # Token contains strange control characters like \x96 or \x95
        # Just filter out the complete line
        if current_subwords_len == 0:
            continue

        if (subword_len_counter + current_subwords_len) > max_len:
            print("")
            print(line)
            subword_len_counter = current_subwords_len
            continue

        subword_len_counter += current_subwords_len

        print(line)