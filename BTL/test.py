import pandas as pd
from sklearn.model_selection import train_test_split
import spacy
from spacy.tokens import DocBin

# Đọc dữ liệu
file_path = 'ner_datasetreference.csv'
dataset = pd.read_csv(file_path, encoding='ISO-8859-1')

# Điền các giá trị NaN trong cột "Sentence #" với giá trị trước đ
dataset['Sentence #'] = dataset['Sentence #'].ffill()

# Tách dữ liệu thành các câu và các nhãn tương ứng
sentences = dataset.groupby("Sentence #")["Word"].apply(list).values
tags = dataset.groupby("Sentence #")["Tag"].apply(list).values

def filter_invalid(sentences, tags):
    filtered_sentences = []
    filtered_tags = []
    for sent, tag in zip(sentences, tags):
        if all(isinstance(word, str) for word in sent):
            filtered_sentences.append(sent)
            filtered_tags.append(tag)
    return filtered_sentences, filtered_tags

sentences, tags = filter_invalid(sentences, tags)

train_sentences, test_sentences, train_tags, test_tags = train_test_split(
    sentences, tags, test_size=0.2, random_state=42
)
def convert_to_spacy_format(sentences, tags):
    nlp = spacy.blank("en")
    db = DocBin()
    for sent, tag in zip(sentences, tags):
        doc = nlp.make_doc(" ".join(sent))
        ents = []
        for i, label in enumerate(tag):
            if label != "O":
                start = len(" ".join(sent[:i])) + (1 if i > 0 else 0)
                end = start + len(sent[i])
                span = doc.char_span(start, end, label=label)
                if span is not None:
                    ents.append(span)
        doc.ents = ents
        db.add(doc)
    return db
train_data_spacy = convert_to_spacy_format(train_sentences, train_tags)
test_data_spacy = convert_to_spacy_format(test_sentences, test_tags)
