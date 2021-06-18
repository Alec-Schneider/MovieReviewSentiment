import sentiment_utils as sent
from sentiment_utils import MovieReviews
import torch
from torchtext.data import get_tokenizer
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

movie_train = MovieReviews("./movie_reviews/", train=True, transform=None)
movie_test = MovieReviews("./movie_reviews/", train=False, transform=None)

sentiment_map = {
    0: "negative",
    1: "somewhat negative",
    2: "neutral",
    3: "somewhat positive",
    4: "positive"
}


tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(yield_tokens(movie_train))

text_pipeline = lambda x: vocab.lookup_indices(tokenizer(x))
label_pipeline = lambda x: int(x)

def yield_tokens(data_iter):
    for text, _ in data_iter:
        yield tokenizer(text)


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_text, _label) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64).to(device)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64).to(device)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0).to(device)
    text_list = torch.cat(text_list).to(device)
    return label_list.to(device), text_list.to(device), offsets.to(device)


num_classes = len(sentiment_map.keys())
vocab_size = len(vocab)
embed_size = 64
model = sent.TextClassificationModel(vocab_size, embed_size, num_classes).to(device)



def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()
    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predited_label = model(text, offsets)
        loss = criterion(predited_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predited_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predited_label = model(text, offsets)
            loss = criterion(predited_label, label)
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count