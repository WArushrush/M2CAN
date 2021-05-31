from image_process import *
from text_process import *


yelp_root = "/home/data/Yelp"
twitter_root = "/home/data/Twitter"
mvsa_root = "/home/data/MVSA"
f = open(yelp_root + "/train.json", 'r')
yelp_train_data = json.load(f)
f.close()
f = open(yelp_root + "/test.json", 'r')
yelp_test_data = json.load(f)
f.close()
f = open(twitter_root + "/train.json", 'r')
twitter_train_data = json.load(f)
f.close()
f = open(twitter_root + "/test.json", 'r')
twitter_test_data = json.load(f)
f.close()
f = open(mvsa_root + "/train.json", 'r')
mvsa_train_data = json.load(f)
f.close()
f = open(mvsa_root + "/test.json", 'r')
mvsa_test_data = json.load(f)
f.close()
yelp_train_len = len(yelp_train_data)
twitter_train_len = len(twitter_train_data)
mvsa_train_len = len(mvsa_train_data)
yelp_test_len = len(yelp_test_data)
twitter_test_len = len(twitter_test_data)
mvsa_test_len = len(mvsa_test_data)


def pack_image_batch(image_batch, resnet50):
    new_image_batch = []
    num_batch = [1 for idx in image_batch]
    for idx in image_batch:
        new_image_batch += [sum(list(process_image_batch(torch.stack(idx).to(device), resnet50)))/len(idx)]
    return torch.stack(new_image_batch), num_batch


def shuffle_dataset():
    random.shuffle(yelp_train_data)
    random.shuffle(twitter_train_data)
    random.shuffle(mvsa_train_data)
    random.shuffle(yelp_test_data)
    random.shuffle(twitter_test_data)
    random.shuffle(mvsa_test_data)


def get_train_data(batch, bert_model, resnet50):
    # get yelp train data
    yelp_start_idx = (batch * batch_size) % yelp_train_len
    yelp_end_idx = ((batch + 1) * batch_size) % yelp_train_len
    yelp_batch = {}
    yelp_batch['text'] = [yelp_train_data[idx]['text'] for idx in range(yelp_start_idx, yelp_end_idx)]
    yelp_batch['augmented_text'] = text_augmentation_transform2(yelp_batch['text'], bert_model)
    yelp_batch['text'] = text_transform2(yelp_batch['text'], bert_model)
    yelp_batch['image'] = []
    for idx in range(yelp_start_idx, yelp_end_idx):
        temp = []
        for img in yelp_train_data[idx]['image']:
            try:
                temp.append(image_transform(yelp_root + "/photos/" + img[:2].lower() + "/" + img + ".jpg"))
            except Exception as e:
                continue
        yelp_batch['image'].append(temp)
        del temp
    yelp_batch['augmented_image'] = []
    for idx in range(yelp_start_idx, yelp_end_idx):
        temp = []
        for img in yelp_train_data[idx]['image']:
            try:
                temp.append(image_augmentation_transform(yelp_root + "/photos/" + img[:2].lower() + "/" + img + ".jpg"))
            except Exception as e:
                continue
        yelp_batch['augmented_image'].append(temp)
        del temp
    yelp_batch['label'] = [int(yelp_train_data[idx]['sentiment']) for idx in range(yelp_start_idx, yelp_end_idx)]

    # get twitter data
    twitter_start_idx = (batch * batch_size) % twitter_train_len
    twitter_end_idx = ((batch + 1) * batch_size) % twitter_train_len
    twitter_batch = {}
    twitter_batch['text'] = [twitter_train_data[idx]['text'] for idx in range(twitter_start_idx, twitter_end_idx)]
    twitter_batch['augmented_text'] = text_augmentation_transform2(twitter_batch['text'], bert_model)
    twitter_batch['text'] = text_transform2(twitter_batch['text'], bert_model)
    twitter_batch['image'] = []
    for idx in range(twitter_start_idx, twitter_end_idx):
        img = twitter_train_data[idx]['image'][0]
        temp = []
        cur = 1
        while True:
            dir = img.split('//')[0]
            file = img.split('//')[1].split('.')[0] + '-' + str(cur)
            if os.path.exists(twitter_root + "/data/" + dir + "/" + dir + file + '.jpg'):
                temp.append(image_transform(twitter_root + "/data/" + dir + "/" + dir + file + '.jpg'))
                cur += 1
            else:
                break
        twitter_batch['image'].append(temp)
        del temp
    twitter_batch['augmented_image'] = []
    for idx in range(twitter_start_idx, twitter_end_idx):
        img = twitter_train_data[idx]['image'][0]
        temp = []
        cur = 1
        while True:
            dir = img.split('//')[0]
            file = img.split('//')[1].split('.')[0] + '-' + str(cur)
            if os.path.exists(twitter_root + "/data/" + dir + "/" + dir + file + '.jpg'):
                temp.append(image_augmentation_transform(twitter_root + "/data/" + dir + "/" + dir + file + '.jpg'))
                cur += 1
            else:
                break
        twitter_batch['augmented_image'].append(temp)
        del temp
    twitter_batch['label'] = [int(twitter_train_data[idx]['sentiment']) for idx in range(twitter_start_idx, twitter_end_idx)]


    # get mvsa data
    mvsa_start_idx = (batch * batch_size) % mvsa_train_len
    mvsa_end_idx = ((batch + 1) * batch_size) % mvsa_train_len
    mvsa_batch = {}
    mvsa_batch['text'] = [mvsa_train_data[idx]['text'] for idx in range(mvsa_start_idx, mvsa_end_idx)]
    mvsa_batch['augmented_text'] = text_augmentation_transform2(mvsa_batch['text'], bert_model)
    mvsa_batch['text'] = text_transform2(mvsa_batch['text'], bert_model)
    mvsa_batch['image'] = [[image_transform(mvsa_root + "/data/" + img)
                            for img in mvsa_train_data[idx]['image']] for idx in
                           range(mvsa_start_idx, mvsa_end_idx)]
    mvsa_batch['augmented_image'] = [[image_augmentation_transform(mvsa_root + "/data/" + img)
                            for img in mvsa_train_data[idx]['image']] for idx in
                           range(mvsa_start_idx, mvsa_end_idx)]
    mvsa_batch['label'] = [int(mvsa_train_data[idx]['sentiment']) for idx in range(mvsa_start_idx, mvsa_end_idx)]

    yelp_batch['image'], yelp_batch['images_per_sample'] = pack_image_batch(yelp_batch['image'], resnet50)
    twitter_batch['image'], twitter_batch['images_per_sample'] = pack_image_batch(twitter_batch['image'], resnet50)
    mvsa_batch['image'], mvsa_batch['images_per_sample'] = pack_image_batch(mvsa_batch['image'], resnet50)
    yelp_batch['augmented_image'], yelp_batch['augmented_images_per_sample'] = pack_image_batch(yelp_batch['augmented_image'], resnet50)
    twitter_batch['augmented_image'], twitter_batch['augmented_images_per_sample'] = pack_image_batch(twitter_batch['augmented_image'], resnet50)
    mvsa_batch['augmented_image'], mvsa_batch['augmented_images_per_sample'] = pack_image_batch(mvsa_batch['augmented_image'], resnet50)
    return yelp_batch, twitter_batch, mvsa_batch


def get_test_data(batch, bert_model, resnet50):
    # get yelp test data
    yelp_start_idx = (batch * batch_size) % yelp_test_len
    yelp_end_idx = ((batch + 1) * batch_size) % yelp_test_len
    yelp_batch = {}
    yelp_batch['text'] = [yelp_test_data[idx]['text'] for idx in range(yelp_start_idx, yelp_end_idx)]
    yelp_batch['text'] = text_transform2(yelp_batch['text'], bert_model)
    yelp_batch['image'] = []
    for idx in range(yelp_start_idx, yelp_end_idx):
        temp = []
        for img in yelp_test_data[idx]['image']:
            try:
                temp.append(image_transform(yelp_root + "/photos/" + img[:2].lower() + "/" + img + ".jpg"))
            except Exception as e:
                continue
        yelp_batch['image'].append(temp)
    yelp_batch['label'] = [int(yelp_test_data[idx]['sentiment']) for idx in range(yelp_start_idx, yelp_end_idx)]

    # get twitter data
    twitter_start_idx = (batch * batch_size) % twitter_test_len
    twitter_end_idx = ((batch + 1) * batch_size) % twitter_test_len
    twitter_batch = {}
    twitter_batch['text'] = [twitter_test_data[idx]['text'] for idx in range(twitter_start_idx, twitter_end_idx)]
    twitter_batch['text'] = text_transform2(twitter_batch['text'], bert_model)
    twitter_batch['image'] = []
    for idx in range(twitter_start_idx, twitter_end_idx):
        img = twitter_test_data[idx]['image'][0]
        temp = []
        cur = 1
        while True:
            dir = img.split('//')[0]
            file = img.split('//')[1].split('.')[0] + '-' + str(cur)
            if os.path.exists(twitter_root + "/data/" + dir + "/" + dir + file + '.jpg'):
                temp.append(image_transform(twitter_root + "/data/" + dir + "/" + dir + file + '.jpg'))
                cur += 1
            else:
                break
        twitter_batch['image'].append(temp)
    twitter_batch['label'] = [int(twitter_test_data[idx]['sentiment']) for idx in range(twitter_start_idx, twitter_end_idx)]


    # get mvsa data
    mvsa_start_idx = (batch * batch_size) % mvsa_test_len
    mvsa_end_idx = ((batch + 1) * batch_size) % mvsa_test_len
    mvsa_batch = {}
    mvsa_batch['text'] = [mvsa_test_data[idx]['text'] for idx in range(mvsa_start_idx, mvsa_end_idx)]
    mvsa_batch['text'] = text_transform2(mvsa_batch['text'], bert_model)
    mvsa_batch['image'] = [[image_transform(mvsa_root + "/data/" + img)
                            for img in mvsa_test_data[idx]['image']] for idx in
                           range(mvsa_start_idx, mvsa_end_idx)]
    mvsa_batch['label'] = [int(mvsa_test_data[idx]['sentiment']) for idx in range(mvsa_start_idx, mvsa_end_idx)]
    yelp_batch['image'], yelp_batch['images_per_sample'] = pack_image_batch(yelp_batch['image'], resnet50)
    twitter_batch['image'], twitter_batch['images_per_sample'] = pack_image_batch(twitter_batch['image'], resnet50)
    mvsa_batch['image'], mvsa_batch['images_per_sample'] = pack_image_batch(mvsa_batch['image'], resnet50)
    return yelp_batch, twitter_batch, mvsa_batch
