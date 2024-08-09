#encoding=utf-8

import numpy as np
import scipy.spatial.distance as spd
import torch
import libmr
import torch.nn.functional as F
from dataloader import RadarDataSet
from torch.utils.data import DataLoader

#训练和测试数据的文件名需要修改
trainset = RadarDataSet("../Records/Train316_P11.txt") 
testsetinv = RadarDataSet("../Records/Test316_P11Intruder_14_1.txt")
trainloader = DataLoader(trainset,batch_size=256,num_workers=4,pin_memory=True)  
testloaderinv = DataLoader(testsetinv,batch_size=256,num_workers=4,pin_memory=True)

#预训练模型路径需要改，自己定
model = torch.load("../TimeRidgeRes/resulttransMoco316_P11.pth")

def calc_distance(query_score, mcv, eu_weight, distance_type='eucos'):
    if distance_type == 'eucos':
        query_distance = spd.euclidean(mcv, query_score) * eu_weight + \
            spd.cosine(mcv, query_score)
    elif distance_type == 'euclidean':
        query_distance = spd.euclidean(mcv, query_score)
    elif distance_type == 'cosine':
        query_distance = spd.cosine(mcv, query_score)
    else:
        print("distance type not known: enter either of eucos, euclidean or cosine")
    return query_distance


#拟合模型
def fit_weibull(means, dists, categories, tailsize=20, distance_type='eucos'):
    """
    Input:
        means (C, channel, C)
        dists (N_c, channel, C) * C
    Output:
        weibull_model : Perform EVT based analysis using tails of distances and save
                        weibull model parameters for re-adjusting softmax scores
    """
    weibull_model = {}
    for mean, dist, category_name in zip(means, dists, categories):
        weibull_model[category_name] = {}
        weibull_model[category_name]['distances_{}'.format(distance_type)] = dist[distance_type]
        print("dist shape: ",dist[distance_type].shape)
        weibull_model[category_name]['mean_vec'] = mean
        weibull_model[category_name]['weibull_model'] = []
        for channel in range(mean.shape[0]):
            mr = libmr.MR()
            tailtofit = np.sort(dist[distance_type][channel, :])[-tailsize:]
            mr.fit_high(tailtofit, len(tailtofit))
            weibull_model[category_name]['weibull_model'].append(mr)

    return weibull_model


def query_weibull(category_name, weibull_model, distance_type='eucos'):
    return [weibull_model[category_name]['mean_vec'],
            weibull_model[category_name]['distances_{}'.format(distance_type)],
            weibull_model[category_name]['weibull_model']]

def compute_openmax_prob(scores, scores_u):
    prob_scores, prob_unknowns = [], []
    for s, su in zip(scores, scores_u):
        channel_scores = np.exp(s)
        channel_unknown = np.exp(np.sum(su))

        total_denom = np.sum(channel_scores) + channel_unknown
        prob_scores.append(channel_scores / total_denom)
        prob_unknowns.append(channel_unknown / total_denom)

    # Take channel mean
    scores = np.mean(prob_scores, axis=0)
    unknowns = np.mean(prob_unknowns, axis=0)
    modified_scores = scores.tolist() + [unknowns]
    return modified_scores

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def openmax(weibull_model, categories, input_score, eu_weight, alpha=7, distance_type='eucos'):
    """Re-calibrate scores via OpenMax layer
    Output:
        openmax probability and softmax probability
    """
    nb_classes = len(categories)

    ranked_list = input_score.argsort().ravel()[::-1][:alpha]
    alpha_weights = [((alpha + 1) - i) / float(alpha) for i in range(1, alpha + 1)]
    
    omega = np.zeros(nb_classes)
    omega[ranked_list] = alpha_weights

    scores, scores_u = [], []
    for channel, input_score_channel in enumerate(input_score):
        score_channel, score_channel_u = [], []
        for c, category_name in enumerate(categories):
            mav, dist, model = query_weibull(category_name, weibull_model, distance_type)
            channel_dist = calc_distance(input_score_channel, mav[channel], eu_weight, distance_type)
            wscore = model[channel].w_score(channel_dist)
            modified_score = input_score_channel[c] * (1 - wscore * omega[c])
            #print("modified_score: ",modified_score)
            score_channel.append(modified_score)
            score_channel_u.append(input_score_channel[c] - modified_score)

        scores.append(score_channel)
        scores_u.append(score_channel_u)

    scores = np.asarray(scores)
    scores_u = np.asarray(scores_u)

    openmax_prob = np.array(compute_openmax_prob(scores, scores_u))
    softmax_prob = softmax(np.array(input_score.ravel()))
    return openmax_prob, softmax_prob


def compute_channel_distances(mavs, features, eu_weight=0.5):
    """
    Input:
        mavs (channel, C)
        features: (N, channel, C)
    Output:
        channel_distances: dict of distance distribution from MAV for each channel.
    """
    eucos_dists, eu_dists, cos_dists = [], [], []
    for channel, mcv in enumerate(mavs):  # Compute channel specific distances
        eu_dists.append([spd.euclidean(mcv, feat[channel]) for feat in features])
        cos_dists.append([spd.cosine(mcv, feat[channel]) for feat in features])
        eucos_dists.append([spd.euclidean(mcv, feat[channel]) * eu_weight +
                            spd.cosine(mcv, feat[channel]) for feat in features])

    return {'eucos': np.array(eucos_dists), 'cosine': np.array(cos_dists), 'euclidean': np.array(eu_dists)}


def compute_train_score_and_mavs_and_dists(train_class_num,trainloader,device,net):
    scores = [[] for _ in range(train_class_num)]
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # this must cause error for cifar
            outputs,_ = net.encoder_q(inputs)
            for score, t in zip(outputs, targets):
                # print(f"torch.argmax(score) is {torch.argmax(score)}, t is {t}")
                if torch.argmax(score) == t:
                    scores[t].append(score.unsqueeze(dim=0).unsqueeze(dim=0))
    scores = [torch.cat(x).cpu().numpy() for x in scores]  # (N_c, 1, C) * C
    mavs = np.array([np.mean(x, axis=0) for x in scores])  # (C, 1, C)
    dists = [compute_channel_distances(mcv, score) for mcv, score in zip(mavs, scores)]
    return scores, mavs, dists

def getMavs(train_class_num,trainloader,device,net):
    scores = [[] for _ in range(train_class_num)]
    hiddensall = [[] for _ in range(train_class_num)]
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, hiddens = net.encoder_q(inputs)
            #outputs = F.softmax(outputs,dim=1)
            i = 0
            for score,t in zip(outputs,targets):
                if torch.argmax(score) == t:
                    scores[t].append(score.unsqueeze(dim=0).unsqueeze(dim=0))
                    hiddensall[t].append(hiddens[i].unsqueeze(dim=0).unsqueeze(dim=0))
                i+= 1
    scores = [torch.cat(x).cpu().numpy() for x in scores]
    mavs = np.array([np.mean(x, axis=0) for x in scores])  # (C, 1, C)
    #mavshiddens = [torch.cat(x).cpu().numpy() for x in hiddensall] # C,1,hidden
    #mavshiddens = np.array([np.mean(x, axis=0) for x in mavshiddens])
    dists = [compute_channel_distances(mcv, score) for mcv, score in zip(mavs, scores)]
    return scores, mavs, dists#, mavshiddens

def Process():
    conunt = 0
    scores,mavs,dists = getMavs(11,trainloader,device='cuda',net=model)
    testall = 0
    testacc = 0
    testallnoin = 0
    TP = 0
    testallin = 0
    testcount  =0
    TN = 0
    FN = 0
    FP = 0
    FPc = 0
    categories = [i for i in range(11)]
    weibullmodels = fit_weibull(mavs,dists,categories,distance_type='cosine')
    print("Weibullmodel Success-----------------")
    #mavsTensor = torch.from_numpy(mavhiddens).cuda()
    #mavsTensor = mavsTensor.squeeze(1)
    DynaMicResultALL = []
    DynaMicResultALLLable = []
    DynaMicResultTrue = []
    DynaMicResultTrueLable = []
    with torch.no_grad():
        for batchidx,(testdata,testlabel) in enumerate(testloaderinv):
            print("current batchidx: ",batchidx)
            testdata = testdata.cuda()
            testlabel = testlabel.numpy()
            outputs,hiddens = model.encoder_q(testdata)
            for i in range(outputs.shape[0]):
                #引入Weibull分布进行处理
                input_score = outputs[i].unsqueeze(0).cpu().detach().numpy()
                openmaxprob,softmaxprob = openmax(weibullmodels,categories,input_score,eu_weight=0.5,distance_type='cosine')
                testall += 1
                #if testlabel[i]  != 11  and np.max(openmaxprob) < 0.7 and np.argmax(openmaxprob) != testlabel[i]:
                #    print("Open: ",openmaxprob)
                #    print("Softmax: ",softmaxprob)
                
                #our Model Based 
                if testlabel[i] != 11:
                    testallnoin += 1
                else:
                    testallin += 1
                if np.max(openmaxprob) > 0.40:
                    predictLable = np.argmax(openmaxprob)
                else:
                    predictLable = 11
               
                ## Softmax Based 
                #if np.max(softmaxprob) > 0.4:
                #    predictLable = np.argmax(softmaxprob)
                #else:
                #    predictLable = 11

                ## Weibull Model
                #predictLable = np.argmax(openmaxprob)

                if predictLable == testlabel[i]:
                    testacc += 1
                    #DynaMicResultALL.append(input_score)
                    #DynaMicResultALLLable.append(testlabel[i])
                if predictLable == testlabel[i] and testlabel[i] != 11:
                    TP += 1
                    #print(openmaxprob)
                elif predictLable == testlabel[i] and testlabel[i] == 11:
                    DynaMicResultTrue.append(hiddens[i].cpu().detach().numpy())
                    DynaMicResultTrueLable.append(conunt)
                    TN += 1

                elif predictLable == 11 and testlabel[i] != 11:  #FN: Known Sample is misclassified to UnKnon
                    FN += 1
                #FP 两个部分，一个部分是未知样本被划分到已知样本上，另外一个部分是已知样本划分错误
                elif testlabel[i] == 11 and predictLable != 11:
                    print(openmaxprob)
                    FP += 1
                    #print(openmaxprob)
                elif testlabel[i] != 11 and predictLable != 11 and testlabel[i] != predictLable: #样本已知类,预测已知类,但已知类之间预测错误
                    FPc += 1
                conunt += 1
    precision = TP / (TP + FP + FPc)
    recall = TP/(TP+FN)
    F1Score = 2 * (precision * recall) / (precision + recall)
    print("TP: ",TP," TN: ",TN," FP: ",FP," FPc: ",FPc," FN: ",FN)
    print("The F1Score is: ",F1Score)

    #画图用的，你用不上
    #DynaMicResultALLData = np.stack(DynaMicResultALL,axis=0)
    #DynaMicResultTrueData = np.stack(DynaMicResultTrue,axis=0)
   
    #DynaMicResultALLLable = np.array(DynaMicResultALLLable)
    #DynaMicResultTrueLable = np.array(DynaMicResultTrueLable)
    #print(DynaMicResultTrueData.shape)
    #print(DynaMicResultTrueLable.shape)
    #np.save("../DynamicKNN/18.npy",DynaMicResultTrueData)
    #np.save("../DynamicKNN/18Lable.npy",DynaMicResultTrueLable)


    #print("Test Count: ",testcount)

if __name__ == "__main__":
    Process()
    

    