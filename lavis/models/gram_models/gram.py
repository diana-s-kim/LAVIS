"""
Emotion Learning pyTorch version--Finding Visual Factors For Emotional Reaction to Abstract Painting
The MIT License (MIT)                                                                                                    
Originally created in 2023, for Python 3.x                                                                                    
Copyright (c) 2023 Diana S. Kim (diana.se.kim@gmail.com)
"""

import torch
from torch import nn
from torch import optim
import numpy as np
from lavis.models.gram_models.classifier import EmotionClassifier
from lavis.models.gram_models.color_gram_classifier import EmotionColorGramClassifier
from lavis.models.gram_models.color_gram_classifier_with_transformer import EmotionColorGramTransformerClassifier



#create gram model
model_level={"resnet34_orig":{"last":{"out":(1,1,512), "drop":1, "freeze_level":7}},
             "resnet34_gram":{"conv4":{"out":(14, 14, 256), "drop":3, "freeze_level":7}, "conv3": {"out":(28,28,128),"drop":4, "freeze_level":4}},
             "resnet34_gram_transformer":{"conv4":{"out":(14, 14, 256), "drop":3, "freeze_level":7}, "conv3": {"out":(28,28,128),"drop":4, "freeze_level":4}},
}

model_config={"resnet34_orig":
                     {"classifier":EmotionClassifier,"name":"resnet34", "drop":None, "freeze_level":None, "mlp":[[512,100],[100,9]], "dropout":[0.3,0.3], "activation":['relu','relu'],"d_model":None, "nhead":None, "dim_feedforward":None, "num_layers":None},
              "resnet34_gram": 
                     {"classifier":EmotionColorGramClassifier,"name":"resnet34", "drop":None, "freeze_level":None, "mlp":None, "dropout":[0.3,0.3], "activation":['relu','relu'], "d_model":None, "nhead":None, "dim_feedforward":None, "num_layers":None},
              "resnet34_gram_transformer": 
                     {"classifier":EmotionColorGramTransformerClassifier,"name":"resnet34", "drop":None, "freeze_level":None, "mlp":None, "dropout":[0.3,0.3], "activation":['relu','relu'], "d_model":64, "nhead":4, "dim_feedforward":128, "num_layers":4}}


#fillup the model config by feature_set#
def update_model_config(specs):
    model_config[specs.model]["drop"]=model_level[specs.model][specs.feature_level]["drop"]
    model_config[specs.model]["freeze_level"]=model_level[specs.model][specs.feature_level]["freeze_level"]
    mlp_dim=0
    if specs.model=="resnet34_gram":#mlp, too
        if "texture" in specs.feature_set:
            mlp_dim+=model_level[specs.model][specs.feature_level]["out"][2]
        if "composition" in specs.feature_set:
            mlp_dim+=model_level[specs.model][specs.feature_level]["out"][0]*model_level[specs.model][specs.feature_level]["out"][1]
        if "color" in specs.feature_set:
            mlp_dim+=3
    elif specs.model=="resnet34_gram_transformer":
        if "texture" in specs.feature_set:
            mlp_dim+=model_config[specs.model]["d_model"]
        if "composition" in specs.feature_set:
            mlp_dim+=model_config[specs.model]["d_model"]
        if "color" in specs.feature_set:
            mlp_dim+=model_config[specs.model]["d_model"]
    else:
        mlp_dim=512
    model_config[specs.model]["mlp"]=[[mlp_dim,100],[100,9]]


class Model_Spec:
    def __init__(self):
        self.img_dir="/ibex/ai/home/kimds/Research/P2/data/wikiart_resize/"
        self.csv_dir="./data/"
        self.styles=None
        self.emotions=None

        self.model="resnet34_gram"#"resnet34_orig"
        self.crop_size=224

        self.learning_rate=5.0e-4
        self.epochs=25
        self.num_batches=32
        self.start_epoch=0

        self.save_model_dir="./model/"
        self.do_cllct=None
        self.resume=None
        self.do_eval=None
        self.do_blip2="/ibex/ai/home/kimds/Research/P2/withLLM/implementation/vision/Emotion_toAbstract/all_gram_model/emotion_20.pt"#"/ibex/ai/home/kimds/Research/P2/withLLM/implementation/vision/Emotion_toAbstract/all_generic_model/emotion_20.pt"
        
        self.feature_level="conv4"#"last"
        self.feature_set=["texture","composition","color"]#None
        self.version="pre_processed_v1"
        self.data="hist_emotion"

def create_gram():
    specs = Model_Spec()
    #device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model
    update_model_config(specs)
    model=model_config[specs.model]["classifier"](name=model_config[specs.model]["name"],drop=model_config[specs.model]["drop"],freeze=model_config[specs.model]["freeze_level"],mlp=model_config[specs.model]["mlp"],dropout=model_config[specs.model]["dropout"],activations=model_config[specs.model]["activation"], d_model=model_config[specs.model]["d_model"], nhead=model_config[specs.model]["nhead"], dim_feedforward=model_config[specs.model][ "dim_feedforward"], num_layers=model_config[specs.model]["num_layers"], factors=specs.feature_set, level=specs.feature_level).to(device)
    model.net.unfreeze()
    if specs.model=="resnet34_gram_transformer":
        optimizer = torch.optim.Adam([{'params': model.net.parameters()},{'params': model.fc_layers.parameters()}, {'params': model.transformer.parameters(), 'lr': 7.5e-4}],lr=specs.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.96)
    else:
        optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': specs.learning_rate}])
              
    #optimization
    if specs.data=="hist_emotion":
        model=nn.Sequential(model,nn.LogSoftmax(dim=-1))
        criterion = nn.KLDivLoss(reduction='batchmean')

    elif specs.data=="max_emotion":
        criterion = nn.CrossEntropyLoss(reduction='mean')

    if specs.do_eval:
        checkpoint=torch.load(specs.do_eval,map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"],strict=True)
        evaluate(val_dataloader, model, criterion, device)
        return 

    if specs.do_blip2:
        checkpoint=torch.load(specs.do_blip2,map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"],strict=True)
        return model[0]

    if specs.do_cllct: #cllct last hidden embedding and style representation
        model.eval()
        if specs.do_cllct[0] != "None": #from original model
            checkpoint=torch.load(specs.do_cllct[0],map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"],strict=True)

        model_to_collect=model[0]
        if specs.do_cllct[1]=="train":
           collect(train_dataloader, specs, model_to_collect, device)
        elif specs.do_cllct[1]=="val":#val
           collect(val_dataloader, specs, model_to_collect, device)
        else:#generated
           collect(generated_dataloader, specs, model_to_collect, device) 
        return

    #resume-part#
    if specs.resume:
        checkpoint=torch.load(specs.resume,map_location=device)
        optimizer.load_state_dict(checkpoint["optimizer"])
#        scheduler.load_state_dict(checkpoint["lr_scheduler"])
        specs.start_epoch = checkpoint["epoch"] + 1
    
    print("start training....")
    model.train()#training
    for epoch in range(specs.start_epoch, specs.epochs):
        train_one_epoch(train_dataloader, model, criterion, optimizer, device)
#        scheduler.step()
        if epoch%5==0:
             torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer':optimizer.state_dict()},specs.save_model_dir+"emotion_"+str(epoch)+".pt")
        evaluate(val_dataloader,model, criterion, device=device)
    return
