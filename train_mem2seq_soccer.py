from models.Mem2Seq import *

BLEU = False

if (args['decoder'] == "Mem2Seq"):
    if args['dataset']=='kvr':
        from utils.utils_kvr_mem2seq import *
        BLEU = True
    else: 
        print("You need to provide the --dataset information")
else:
    if args['dataset']=='kvr':
        from utils.utils_kvr import *
        BLEU = True
    else: 
        print("You need to provide the --dataset information")


avg_best,cnt,acc = 0.0,0,0.0
cnt_1 = 0
train, dev, test, testOOV, lang, max_len, max_r = prepare_data_seq(args['task'],batch_size=int(args['batch']),shuffle=True)

if args['decoder'] == "Mem2Seq":
    model = globals()[args['decoder']](int(args['hidden']),
                                        max_len,max_r,lang,args['path'],args['task'],
                                        lr=float(args['learn']),
                                        n_layers=int(args['layer']), 
                                        dropout=float(args['drop']),
                                        unk_mask=bool(int(args['unk_mask']))
                                    )
else:
    model = globals()[args['decoder']](hidden_size=int(args['hidden']), max_r=max_r, n_words=lang.n_words, b_size=lang.n_words, emb_dim=lang.n_words, sos_tok=SOS_token, eos_tok=EOS_token, itos=5, kb_max_size=100
                                )

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


