import numpy as np
import logging 
from tqdm import tqdm

from utils.config import *
from models.enc_vanilla import *
from models.enc_Luong import *
from models.enc_PTRUNK import *

from utils.utils_kvr import *
BLEU = True

# Configure models
avg_best,cnt,acc = 0.0,0,0.0
cnt_1 = 0   
### LOAD DATA
train, dev, test, testOOV, lang, max_len, max_r = prepare_data_seq(args['task'],batch_size=int(args['batch']),shuffle=True)

model = globals()[args['decoder']](int(args['hidden']),max_len,max_r,lang,args['path'],args['task'],lr=float(args['learn']),n_layers=int(args['layer']),dropout=float(args['drop']))

for epoch in range(300):
    logging.info("Epoch:{}".format(epoch))  
    # Run the train function
    pbar = tqdm(enumerate(train),total=len(train))
    for i, data in pbar: 
        model.train_batch(data[0], data[1], data[2], data[3],data[4],data[5],
                        len(data[1]),10.0,0.5,i==0) 
        pbar.set_description(model.print_loss())
        
    if((epoch+1) % int(args['evalp']) == 0):    
        acc = model.evaluate(dev,avg_best, BLEU)    
        if 'Mem2Seq' in args['decoder']:
            model.scheduler.step(acc)
        if(acc >= avg_best):
            avg_best = acc
            cnt=0
        else:
            cnt+=1
        if(cnt == 5): break
        if(acc == 1.0): break 


