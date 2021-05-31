from model.seq2seq import TextFeatureExtractor, TaskClassifier, MMDiscriminator, MLB
from model.image2image import ImageEncoder
from data_loader import *
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from copy import deepcopy

print("start")

ae_criterion = nn.MSELoss()
GAN_criterion = F.nll_loss
task_criterion = F.nll_loss


def get_real_labels(idx):
    return torch.tensor([0] * idx[0] + [1] * idx[1] + [2] * idx[2], requires_grad=False).long().to(device)


def train(epoch, target_idx, bert_model, resnet50, bert_optimizer, resnet50_optimizer):
    print("training")
    bert_model.train()
    resnet50.train()
    textfeatureextractor.train()
    imageencoder.train()
    mmdiscriminator1.train()
    mmdiscriminator2.train()
    mlb.train()
    taskclassifier.train()
    mean_task_loss = 0
    batch_cnt = 0
    for it in range(14170 // batch_size):
        try:
            torch.cuda.empty_cache()
            batch = it

            train_data = get_train_data(batch, bert_model, resnet50)
            temp = [train_data[idx] for idx in range(3) if idx != target_idx]
            temp.append(train_data[target_idx])
            train_data = temp
            del temp

            # text feature extraction
            text_batches = [data['text'] for data in train_data]
            augmented_text_batches = [data['augmented_text'] for data in train_data]
            image_batches = [data['image'] for data in train_data]
            augmented_image_batches = [data['augmented_image'] for data in train_data]
            text_feature_batches = []
            augmented_text_feature_batches = []
            projected_text_feature_batches = []
            projected_augmented_text_feature_batches = []
            for domain in range(3):
                text_batch = text_batches[domain]
                augmented_text_batch = augmented_text_batches[domain]
                text_feature_batch, projected_text_feature_batch = textfeatureextractor(text_batch)
                text_feature_batches.append(text_feature_batch)
                projected_text_feature_batches.append(projected_text_feature_batch)
                augmented_text_feature_batch, projected_augmented_text_feature_batch = textfeatureextractor(
                    augmented_text_batch)
                augmented_text_feature_batches.append(augmented_text_feature_batch)
                projected_augmented_text_feature_batches.append(projected_augmented_text_feature_batch)

            # image feature extraction
            image_feature_batches = []
            augmented_image_feature_batches = []
            projected_image_feature_batches = []
            projected_augmented_image_feature_batches = []
            for domain in range(3):
                image_batch = image_batches[domain]
                augmented_image_batch = augmented_image_batches[domain]
                image_feature_batch, projected_image_feature_batch = imageencoder(image_batch)
                image_feature_batches.append(image_feature_batch)
                projected_image_feature_batches.append(projected_image_feature_batch)
                augmented_image_feature_batch, projected_augmented_image_feature_batch = imageencoder(
                    augmented_image_batch)
                augmented_image_feature_batches.append(augmented_image_feature_batch)
                projected_augmented_image_feature_batches.append(projected_augmented_image_feature_batch)

            # cross-modal contrastive loss
            contrastive_loss_1 = 0
            for domain in range(3):
                pos_sim = torch.exp(
                    torch.sum(projected_text_feature_batches[domain] * projected_image_feature_batches[domain],
                              dim=-1)) + \
                          torch.exp(torch.sum(
                              projected_text_feature_batches[domain] * projected_augmented_image_feature_batches[
                                  domain], dim=-1)) + \
                          torch.exp(torch.sum(
                              projected_augmented_text_feature_batches[domain] * projected_image_feature_batches[
                                  domain], dim=-1)) + \
                          torch.exp(torch.sum(projected_augmented_text_feature_batches[domain] *
                                              projected_augmented_image_feature_batches[domain], dim=-1))
                sim_matrix = torch.sum(
                    torch.exp(torch.mm(projected_text_feature_batches[domain],
                                       projected_image_feature_batches[domain].t().contiguous())) + \
                    torch.exp(torch.mm(projected_text_feature_batches[domain],
                                       projected_augmented_image_feature_batches[domain].t().contiguous())) + \
                    torch.exp(torch.mm(projected_augmented_text_feature_batches[domain],
                                       projected_image_feature_batches[domain].t().contiguous())) + \
                    torch.exp(torch.mm(projected_augmented_text_feature_batches[domain],
                                       projected_augmented_image_feature_batches[domain].t().contiguous()))
                    , dim=-1
                )
                contrastive_loss_1 += (- torch.log(pos_sim / sim_matrix)).mean()

            # cross-domain contrastive loss
            contrastive_loss_2 = 0
            for s in range(3):
                for t in range(s + 1, 3):
                    contrastive_loss_2 += -torch.mm(text_feature_batches[s],
                                                    text_feature_batches[t].t().contiguous()).mean()
                    contrastive_loss_2 += -torch.mm(text_feature_batches[s],
                                                    augmented_text_feature_batches[t].t().contiguous()).mean()
                    contrastive_loss_2 += -torch.mm(augmented_text_feature_batches[s],
                                                    text_feature_batches[t].t().contiguous()).mean()
                    contrastive_loss_2 += -torch.mm(augmented_text_feature_batches[s],
                                                    augmented_text_feature_batches[t].t().contiguous()).mean()
                    contrastive_loss_2 += -torch.mm(image_feature_batches[s],
                                                    image_feature_batches[t].t().contiguous()).mean()
                    contrastive_loss_2 += -torch.mm(image_feature_batches[s],
                                                    augmented_image_feature_batches[t].t().contiguous()).mean()
                    contrastive_loss_2 += -torch.mm(augmented_image_feature_batches[s],
                                                    image_feature_batches[t].t().contiguous()).mean()
                    contrastive_loss_2 += -torch.mm(augmented_image_feature_batches[s],
                                                    augmented_image_feature_batches[t].t().contiguous()).mean()
            del projected_augmented_image_feature_batches, projected_augmented_text_feature_batches, \
                projected_image_feature_batches, projected_text_feature_batches

            # fusion
            mm_features = [mlb(text_feature_batches[domain], image_feature_batches[domain]) for domain in range(3)]
            mm_features2 = [mlb(text_feature_batches[domain], augmented_image_feature_batches[domain]) for domain in
                            range(3)]
            mm_features3 = [mlb(augmented_text_feature_batches[domain], image_feature_batches[domain]) for domain in
                            range(3)]
            mm_features4 = [mlb(augmented_text_feature_batches[domain], augmented_image_feature_batches[domain]) for
                            domain in range(3)]
            del text_feature_batches, image_feature_batches

            # cross-domain adversarial loss
            mm_D1_loss = 0
            mm_D2_loss = 0

            predict_domain_s1 = mmdiscriminator1(mm_features[0])
            predict_domain_t1 = mmdiscriminator1(mm_features[2])
            mm_D1_loss += GAN_criterion(predict_domain_s1, torch.ones(len(predict_domain_s1),
                                                                      requires_grad=False).long().to(device))
            mm_D1_loss += GAN_criterion(predict_domain_t1, torch.zeros(len(predict_domain_t1),
                                                                       requires_grad=False).long().to(device))

            predict_domain_s2 = mmdiscriminator2(mm_features[1])
            predict_domain_t2 = mmdiscriminator2(mm_features[2])
            mm_D2_loss += GAN_criterion(predict_domain_s2, torch.ones(len(predict_domain_s2),
                                                                      requires_grad=False).long().to(device))
            mm_D2_loss += GAN_criterion(predict_domain_t2, torch.zeros(len(predict_domain_t2),
                                                                       requires_grad=False).long().to(device))
            del predict_domain_s1, predict_domain_s2, predict_domain_t1, predict_domain_t2
            mm_D_loss = mm_D1_loss + mm_D2_loss

            # task
            task_loss = 0
            for idx in range(2):
                label = torch.tensor(train_data[idx]['label'], requires_grad=False).long().to(device)
                predict_label = taskclassifier(mm_features[idx])
                predict_label2 = taskclassifier(mm_features2[idx])
                predict_label3 = taskclassifier(mm_features3[idx])
                predict_label4 = taskclassifier(mm_features4[idx])
                task_loss += (task_criterion(predict_label, label) + task_criterion(predict_label2, label) +
                              task_criterion(predict_label3, label) + task_criterion(predict_label4, label)) / 4
            if batch % 180 == 0:
                print("target: ", target_idx, "epoch: ", epoch, " batch: ", batch, " cross-modal contrastive loss: ",
                      str(round(float(contrastive_loss_1), 5)), " cross-domain contrastive loss: ",
                      str(round(float(contrastive_loss_2), 5)), " cross-domain adversarial loss 1: ", str(round(float(mm_D1_loss), 5)),
                      " cross-domain adversarial loss 2: ", str(round(float(mm_D2_loss), 5)), " task_loss: ",
                      str(round(float(task_loss), 5)))
            del train_data, mm_features
            bert_optimizer.zero_grad()
            resnet50_optimizer.zero_grad()
            textfeatureextractor_optimizer.zero_grad()
            imageencoder_optimizer.zero_grad()
            mlb_optimizer.zero_grad()
            mmdiscriminator1_optimizer.zero_grad()
            mmdiscriminator2_optimizer.zero_grad()
            taskclassifier_optimizer.zero_grad()
            total_loss = task_loss + 0.02 * contrastive_loss_1 + 0.02 * contrastive_loss_2 + 0.05 * mm_D_loss
            total_loss.backward(retain_graph=False)
            bert_optimizer.step()
            resnet50_optimizer.step()
            textfeatureextractor_optimizer.step()
            imageencoder_optimizer.step()
            mlb_optimizer.step()
            mmdiscriminator1_optimizer.step()
            mmdiscriminator2_optimizer.step()
            taskclassifier_optimizer.step()

            mean_task_loss += task_loss.item()
            batch_cnt += 1
            del task_loss
            torch.cuda.empty_cache()

        except Exception as e:
            # print(e)
            torch.cuda.empty_cache()
            continue

    print("target: ", target_idx, "epoch: ", epoch, "mean task loss: ", round(float(mean_task_loss / batch_cnt), 5))


def test(target_idx, bert_model, resnet50, bert_optimizer, resnet50_optimizer):
    print("testing")
    bert_model.eval()
    resnet50.eval()
    textfeatureextractor.eval()
    imageencoder.eval()
    mlb.eval()
    mmdiscriminator1.eval()
    mmdiscriminator2.eval()
    taskclassifier.eval()
    acc = 0
    cnt = 0
    for it in range(1500 // batch_size):
        try:
            # print(it)
            batch = it
            test_data = get_test_data(batch, bert_model, resnet50)
            temp = [test_data[idx] for idx in range(3) if idx != target_idx]
            temp.append(test_data[target_idx])
            test_data = temp

            text_batches = [data['text'] for data in test_data]
            text_feature_batches = []
            for domain in range(3):
                text_batch = text_batches[domain]
                temp0, temp1 = textfeatureextractor(text_batch)
                text_feature_batches.append(temp0)

            image_batches = [data['image'] for data in test_data]
            image_feature_batches = []
            for domain in range(3):
                image_batch = image_batches[domain]
                temp0, temp1 = imageencoder(image_batch)
                image_feature_batches.append(temp0)

            # fusion
            mm_features = [mlb(text_feature_batches[domain], image_feature_batches[domain]) for domain in range(3)]

            label = torch.tensor(test_data[2]['label'], requires_grad=False).long().to(device)
            predict_label = taskclassifier(mm_features[2])
            for jdx in range(batch_size):
                temp = predict_label[jdx].tolist()
                pred = temp.index(max(temp))
                if pred == label.tolist()[jdx]:
                    acc += 1
                cnt += 1

        except Exception as e:
            # print(e)
            continue
    return acc / cnt


domain_name = ['Y', 'T', 'M']


top_acc = [0, 0, 0]
top_epoch = [-1, -1, -1]

for target_idx in range(1):
    print("refreshing bert")
    bert_model = BertModel.from_pretrained('bert-base-cased')
    bert_model = bert_model.to(device)
    bert_optimizer = torch.optim.Adam(bert_model.parameters(), lr=0.00002, eps=1e-8)
    print("refreshing resnet50")
    resnet50 = models.resnet50(pretrained=True)
    resnet50.fc = nn.Linear(2048, 2048)
    torch.nn.init.eye(resnet50.fc.weight)
    resnet50 = resnet50.to(device)
    resnet50_optimizer = torch.optim.Adam(resnet50.parameters(), lr=0.00002, eps=1e-8)
    textfeatureextractor = TextFeatureExtractor().to(device)

    imageencoder = ImageEncoder().to(device)
    mlb = MLB().to(device)
    mmdiscriminator1 = MMDiscriminator().to(device)
    mmdiscriminator2 = MMDiscriminator().to(device)

    taskclassifier = TaskClassifier().to(device)
    textfeatureextractor_optimizer = torch.optim.Adam(textfeatureextractor.parameters(), lr=0.0005, betas=(0.9, 0.999))

    imageencoder_optimizer = torch.optim.Adam(imageencoder.parameters(), lr=0.0005, betas=(0.9, 0.999))
    mlb_optimizer = torch.optim.Adam(mlb.parameters(), lr=0.0005, betas=(0.9, 0.999))
    mmdiscriminator1_optimizer = torch.optim.Adam(mmdiscriminator1.parameters(), lr=0.0005, betas=(0.9, 0.999))
    mmdiscriminator2_optimizer = torch.optim.Adam(mmdiscriminator2.parameters(), lr=0.0005, betas=(0.9, 0.999))
    taskclassifier_optimizer = torch.optim.Adam(taskclassifier.parameters(), lr=0.0005, betas=(0.9, 0.999))

    for epoch in range(num_epoch):
        shuffle_dataset()
        train(epoch, target_idx, bert_model, resnet50, bert_optimizer, resnet50_optimizer)
        cur_acc = test(target_idx, bert_model, resnet50, bert_optimizer, resnet50_optimizer)
        if cur_acc > top_acc[target_idx]:
            top_acc[target_idx] = cur_acc
            top_epoch[target_idx] = epoch
            bert_model.save_pretrained('pretrain/bert3_' + str(target_idx) + '.pth')
            torch.save(resnet50.state_dict(), 'pretrain/resnet50_' + str(target_idx) + '.pth')
        print("target_idx", target_idx, "epoch: ", epoch, "acc: ", cur_acc, "top acc: ", top_acc, "top epoch: ",
              top_epoch)
