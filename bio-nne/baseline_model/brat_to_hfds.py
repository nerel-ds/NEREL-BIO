import json
import os
import argparse


from nltk.data import load
from nltk.tokenize import NLTKWordTokenizer


from tqdm.auto import tqdm



import nltk
nltk.download('punkt')



# Download tokenizer for Russian
ru_tokenizer = load("tokenizers/punkt/russian.pickle") 
word_tokenizer = NLTKWordTokenizer()



brat2mrc_parser = argparse.ArgumentParser(description = "Brat to hfds-json formatter script.")

brat2mrc_parser.add_argument('--brat_dataset_path', type = str, required = True, help = "Path to brat dataset (with train, dev, test dirs).")

brat2mrc_parser.add_argument('--tags_path', type = str, required = True, help = 'Path to tags file with format ["CLASS1", "CLASS2", ...].')

brat2mrc_parser.add_argument('--hfds_output_path', type = str, default = None, help = "Path, where formatted dataset would be stored. By default, same path as in --brat_dataset_path would be used.")



args = brat2mrc_parser.parse_args()



brat_dataset_path = args.brat_dataset_path



hfds_output_path = args.hfds_output_path

if hfds_output_path is None:
    hfds_output_path = brat_dataset_path

tags_path = args.tags_path

with open(tags_path, "r") as tags_file:
    tags = json.loads(tags_file.read())

print(tags)

sets = ["train", "dev", "test"]

for ds in sets:

    print(ds + " set:")
    
    jsonpath = os.path.join(hfds_output_path, ds + ".json")
    dataset_path = os.path.join(brat_dataset_path, ds)
    jsondir = os.path.dirname(jsonpath)

    if not os.path.exists(jsondir):

        os.makedirs(jsondir)
        
    jsonfile = open(jsonpath, "w", encoding='UTF-8')
    doc_count = 0
    entities_count = 0
    doc_ids = []

    for ad, dirs, files in os.walk(dataset_path):
        for f in tqdm(files):
            if f[-4:] == '.ann':
                try:
                    if os.stat(dataset_path + '/' + f).st_size == 0:
                        continue
                    
                    # Read all Named Entities from the file.
                    annfile = open(dataset_path + '/' + f, "r", encoding='UTF-8')
                    txtfile = open(dataset_path + '/' + f[:-4] + ".txt", "r", encoding='UTF-8')
                    txtdata = txtfile.read()

                    entity_types = []
                    entity_start_chars = []
                    entity_end_chars = []

                    for line in annfile:
                        line_tokens = line.split()
                        if len(line_tokens) > 3 and len(line_tokens[0]) > 1 and line_tokens[0][0] == 'T':
                            try:
                                if line_tokens[1] in tags:
                                    entity_type = line_tokens[1]
                                    start_char = int(line_tokens[2])
                                    end_char = int(line_tokens[3])
             
                                    entity_types.append(entity_type)
                                    entity_start_chars.append(start_char)
                                    entity_end_chars.append(end_char)

                            except ValueError:

                                continue 

                        try:
                            assert len(entity_types) == len(entity_start_chars) == len(entity_end_chars)

                        except AssertionError:
                            raise AssertionError


                    annfile.close()
                    txtfile.close()

                    # In each file, select the contexts separately.

                    offset_mapping = []
                    text = ''

                    sentence_spans = ru_tokenizer.span_tokenize(txtdata)

                    for span in sentence_spans:
                        start, end = span
                        context = txtdata[start : end]
                        word_spans = word_tokenizer.span_tokenize(context)
                        offset_mapping.extend([(s + start, e + start) for s, e in word_spans])

                    try:
                        assert len(entity_types) == len(entity_start_chars) == len(entity_end_chars)
                    except AssertionError:

                        print(f[:-4])
                        print(txtdata)
                        print(entity_types)
                        print(len(entity_types))
                        print(entity_start_chars)
                        print(len(entity_start_chars))
                        print(entity_end_chars)
                        print(len(entity_end_chars))

                        for s, e, t in zip(entity_start_chars, entity_end_chars, entity_types):

                            print(t, txtdata[s : e])

                        raise AssertionError

                    # Compile entities into the required format

                    start_words, end_words = zip(*offset_mapping)
                    doc_entities = {

                        'text': txtdata,
                        'entity_types': entity_types,
                        'entity_start_chars': entity_start_chars,
                        'entity_end_chars': entity_end_chars,
                        'id': f[:-4],
                        'word_start_chars': start_words,
                        'word_end_chars': end_words
                    }

                    # Save the enitites into hfds format.

                    doc_count += 1
                    doc_ids.append(f[:-4])
                    entities_count += len(doc_entities["entity_types"])

                    jsonfile.write(json.dumps(doc_entities, ensure_ascii = False) + '\n')

                except FileNotFoundError:
                    pass



    # Print the results

    print(f"Entities: {entities_count}")
    print(f"Docs: {doc_count}")


    jsonfile.close()