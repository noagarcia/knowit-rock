import torch
import torch.nn as nn
from torchvision import models


class TableModule(nn.Module):
    def __init__(self):
        super(TableModule, self).__init__()

    def forward(self, x, dim):
        y = torch.cat(x, dim)
        return y


class VR_ImageFeatures(nn.Module):
    def __init__(self, args):
        super(VR_ImageFeatures, self).__init__()

        # Load pre-trained visual model
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        # Image projection
        self.img_emb = nn.Sequential(
            nn.Linear(2048*args.numframes, args.img_space),
        )

        # Concatenation
        self.concat = TableModule()

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(args.img_space + args.bert_emb_size, 1)
        )

    def forward(self, img0, img1, img2, img3, img4, bert_text1, bert_text2, bert_text3, bert_text4):

        # Visual embeddings
        vis_emb1 = torch.squeeze(self.resnet(img0))
        vis_emb2 = torch.squeeze(self.resnet(img1))
        vis_emb3 = torch.squeeze(self.resnet(img2))
        vis_emb4 = torch.squeeze(self.resnet(img3))
        vis_emb5 = torch.squeeze(self.resnet(img4))
        vis_emb = self.img_emb(self.concat([vis_emb1, vis_emb2, vis_emb3, vis_emb4, vis_emb5], 1))

        #  Concat
        emb1 = torch.squeeze(self.concat([vis_emb, bert_text1], 1), 1)
        emb2 = torch.squeeze(self.concat([vis_emb, bert_text2], 1), 1)
        emb3 = torch.squeeze(self.concat([vis_emb, bert_text3], 1), 1)
        emb4 = torch.squeeze(self.concat([vis_emb, bert_text4], 1), 1)

        # Classifier
        scoreAnswer1 = self.classifier(emb1)
        scoreAnswer2 = self.classifier(emb2)
        scoreAnswer3 = self.classifier(emb3)
        scoreAnswer4 = self.classifier(emb4)

        # Concat 4 outputs
        out = torch.squeeze(self.concat([scoreAnswer1, scoreAnswer2, scoreAnswer3, scoreAnswer4], 1), 1)
        return out


class VR_ImageBOW(nn.Module):
    def __init__(self, args, num_words):
        super(VR_ImageBOW, self).__init__()

        # Image projection
        self.imgmap = nn.Sequential(
            nn.Linear(num_words, args.img_space),
        )

        # Concatenation
        self.concat = TableModule()

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(args.img_space + args.bert_emb_size, 1),
        )

    def forward(self, bow, bert_text1, bert_text2, bert_text3, bert_text4):

        vis_emb = self.imgmap(bow)

        #  Concat
        emb1 = torch.squeeze(self.concat([vis_emb, bert_text1], 1), 1)
        emb2 = torch.squeeze(self.concat([vis_emb, bert_text2], 1), 1)
        emb3 = torch.squeeze(self.concat([vis_emb, bert_text3], 1), 1)
        emb4 = torch.squeeze(self.concat([vis_emb, bert_text4], 1), 1)

        # Classifier
        scoreAnswer1 = self.classifier(emb1)
        scoreAnswer2 = self.classifier(emb2)
        scoreAnswer3 = self.classifier(emb3)
        scoreAnswer4 = self.classifier(emb4)

        # Concat 4 outputs
        out = torch.squeeze(self.concat([scoreAnswer1, scoreAnswer2, scoreAnswer3, scoreAnswer4], 1), 1)
        return out