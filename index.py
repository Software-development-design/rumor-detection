import os
import json
import seaborn
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,recall_score,roc_curve,auc,f1_score,confusion_matrix

# 0:真话 1：谣言


def get_train_val_txt(true_dir="non-rumor-repost",fake_dir="rumor-repost"):
    datasets,labels=[],[]
    for i,json_dir in enumerate([true_dir,fake_dir]):
        for json_name in tqdm(os.listdir(json_dir)):
            json_path=json_dir+"/"+json_name
            f=open(json_path,encoding="utf-8")
            json_list=json.load(f)
            for json_obj in json_list:
                text=json_obj.get("text","")
                text=text.strip().replace("\n", "").replace("[", "").replace("]", "").replace("…","").replace("@","")
                if len(text)>10:
                    datasets.append(text)
                    labels.append(i)
            f.close()
    datasets_train,datasets_test,labels_train,labels_test=train_test_split(datasets,labels,train_size=0.8,shuffle=True)
    with open("train.txt","w",encoding="utf-8") as f1,open("test.txt","w",encoding="utf-8") as f2:
        for i in range(len(datasets_train)):
            f1.write(datasets_train[i]+"\t"+str(labels_train[i])+"\n")

        for i in range(len(datasets_test)):
            f2.write(datasets_test[i]+"\t"+str(labels_test[i])+"\n")


def get_word_dict(root1="train.txt",root2="test.txt",n_common=3000):
    word_count=Counter()
    for root in [root1,root2]:
        with open(root,"r",encoding="utf-8") as f:
            for line in f.readlines():
                line_split=line.strip().split("\t")
                for word in line_split[0]:
                    word_count[word]+=1
    most_common=word_count.most_common(n_common)
    word2index_dict={word:index+2 for index,(word,count) in enumerate(most_common)}
    word2index_dict["UNK"]=1
    word2index_dict["PAD"]=0

    return word2index_dict


class DataGenerator(Dataset):

    def __init__(self,word2index_dict,root="train.txt",max_len=50):
        super(DataGenerator, self).__init__()
        self.root=root
        self.max_len=max_len
        self.word2index_dict=word2index_dict
        self.datasets,self.labels=self.get_datasets()

    def __getitem__(self, item):
        dataset=self.datasets[item]
        label=self.labels[item]
        if len(dataset)<self.max_len:
            dataset+=[0]*(self.max_len-len(dataset))
        else:
            dataset=dataset[:self.max_len]

        return torch.LongTensor(dataset),torch.from_numpy(np.array(label)).long()

    def __len__(self):
        return len(self.labels)

    def get_datasets(self):
        datasets,labels=[],[]
        with open(self.root,"r",encoding="utf-8") as f:
            for line in f.readlines():
                line_split=line.strip().split("\t")
                datasets.append([self.word2index_dict.get(word,1) for word in list(line_split[0])])
                labels.append(int(line_split[1]))

        return datasets,labels


class BiLSTMModel(nn.Module):

    def __init__(self,num_vocab):
        super(BiLSTMModel, self).__init__()
        self.embedding=nn.Embedding(num_embeddings=num_vocab,embedding_dim=128)
        self.lstm=nn.LSTM(input_size=128,hidden_size=256,bidirectional=True,batch_first=True,num_layers=2)
        self.fc1=nn.Sequential(
            nn.Linear(512,512),
            nn.ReLU(inplace=True)
        )
        self.fc2=nn.Linear(512,2)

    def forward(self,x):
        out=self.embedding(x)
        outputs,(h,c)=self.lstm(out)
        out=torch.cat([h[-1,:,:],h[-2,:,:]],dim=-1)
        out=self.fc1(out)

        return self.fc2(out)


def train():
    device=torch.device("cuda")
    word2index_dict=get_word_dict()
    model=BiLSTMModel(len(word2index_dict)).to(device)
    optimizer=optim.Adam(model.parameters(),lr=1e-3)
    schedule=optim.lr_scheduler.StepLR(optimizer,step_size=2000,gamma=0.9)
    loss_func=nn.CrossEntropyLoss()
    train_loader=DataLoader(DataGenerator(word2index_dict,root="train.txt"),shuffle=True,batch_size=64)
    test_loader=DataLoader(DataGenerator(word2index_dict,root="test.txt"),shuffle=False,batch_size=64)
    for epoch in range(31):
        train_accuracy=train_one_epoch(model,train_loader,loss_func,optimizer,schedule,device,epoch)
        test_accuracy=get_test_result(model,test_loader,device)
        print(f"epoch:{epoch+1},train accuracy:{train_accuracy},test accuracy:{test_accuracy}")
        if (epoch+1) % 10 == 0:
            torch.save(model,f"bilstm_model_epoch{epoch+1}.pth")


def train_one_epoch(model,train_loader,loss_func,optimizer,schedule,device,epoch):
    model.train()
    data=tqdm(train_loader)
    labels_true,labels_pred=np.array([]),np.array([])
    for batch,(x,y) in enumerate(data):
        labels_true=np.concatenate([labels_pred,y.numpy()],axis=-1)
        datasets_train,labels_train=x.to(device),y.to(device)
        prob=model(datasets_train)
        pred=torch.argmax(prob,dim=-1).cpu().numpy()
        labels_pred=np.concatenate([labels_pred,pred],axis=-1)
        loss=loss_func(prob,labels_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        data.set_description_str(f"epoch:{epoch+1},batch:{batch+1},loss:{loss.item()},lr:{schedule.get_last_lr()[0]}")
    accuracy=np.mean(np.array(labels_pred==labels_true).astype(int))

    return accuracy


def get_test_result(model,test_loader,device):
    model.eval()
    data=tqdm(test_loader)
    labels_true,labels_pred=np.array([]),np.array([])
    labels_prob=[]
    with torch.no_grad():
        for x,y in data:
            labels_true=np.concatenate([labels_true,y.numpy()],axis=-1)
            datasets_test=x.to(device)
            prob=model(datasets_test)
            pred=torch.argmax(prob,dim=-1).cpu().numpy()
            labels_pred=np.concatenate([labels_pred,pred],axis=-1)
            labels_prob.append(prob.cpu().numpy())
    labels_prob=np.concatenate(labels_prob,axis=0)
    precision=precision_score(labels_true,labels_pred)
    recall=recall_score(labels_true,labels_pred)
    f1=f1_score(labels_true,labels_pred)
    accuracy=np.mean(np.array(labels_pred==labels_true).astype(int))
    print(f"accuracy:{accuracy},precision:{precision},recall:{recall},f1:{f1}")

    fpr,tpr,_=roc_curve(labels_true,labels_prob[:,-1])

    plt.figure(figsize=(8,8))
    plt.plot([0,1],[0,1],"r--")
    plt.plot(fpr,tpr,"green",label=f"AUC:{auc(fpr,tpr)}")
    plt.legend()
    plt.title("BiLSTM roc_curve")
    plt.savefig("roc_curve.png")
    matrix=confusion_matrix(labels_true,labels_pred,normalize="true")

    plt.figure(figsize=(8,8))
    seaborn.heatmap(matrix,annot=True,cmap="GnBu")
    plt.title("confusion_matrix")
    plt.savefig("confusion_matrix.png")

    return accuracy


def predict(sentence,model_path="bilstm_model_epoch10.pth"):
    print(f"{sentence}  的预测结果为:",end=" ")
    labels=["真话","谣言"]
    device=torch.device("cuda")
    word2index_dict=get_word_dict()
    sentence=[word2index_dict.get(word,1) for word in sentence]
    if len(sentence)<50:
        sentence+=[0]*(50-len(sentence))
    else:
        sentence=sentence[:50]
    datasets=torch.unsqueeze(torch.LongTensor(sentence),dim=0).to(device)
    model=torch.load(model_path).to(device)
    model.eval()
    with torch.no_grad():
        labels_pred=torch.argmax(model(datasets),dim=-1).cpu().numpy()[0]
    print(f"{labels[labels_pred]}")

    return labels[labels_pred]

def module_evaluation():
    model = torch.load("bilstm_model_epoch10.pth").to("cuda")
    test_loader = DataLoader(DataGenerator(get_word_dict(), root="test.txt"), shuffle=False, batch_size=64)
    get_test_result(model, test_loader, "cuda")

    return

if __name__ == '__main__':
    # get_train_val_txt()
    # train()
    module_evaluation()
    rumor = input("请输入待检测话题\n")
    predict(rumor)










































