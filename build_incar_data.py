import json
import numpy as np

def navigate_to_conv_n_kg():
    dataset_types = ["val","test","train"]
    entity_f = open("data/kvr_entities_incar.txt","w",encoding="utf-8")
    entity_list = set()
    data_dir = "data/KVR/"
    incar_kg_dir = "data/KG/incar/"
    incar_conv_dir = "conversations_incar/"
    hit_team_maps = dict()
    kg_id = 0
    for dataset_type in dataset_types:
        data_r = open(data_dir+dataset_type+"_incar.txt","r",encoding="utf-8")
        found = False
        for line in data_r:

            if line.startswith("#") or found:
                if line.startswith("#conv#") or found:
                    if not found:
                        kb_write = open(incar_kg_dir + str(kg_id) + "_kg.txt", "w", encoding="utf-8")
                        diag_write = open(incar_conv_dir+dataset_type+"/"+str(kg_id)+".json","w")
                        hit_team_maps[str(kg_id)] = str(kg_id)
                        kg_id+=1
                        found = True
                        q_id = 1

                        data = dict()
                        continue

                    if line.startswith("0") and len(line)>2:

                        triple = line.strip().split(" ")
                        if len(triple)==4:
                            kb_write.write(triple[1] + "\t" + triple[2] + "\t" + triple[3] + "\n")
                            entity_list.add(triple[1])
                            entity_list.add(triple[3])
                        elif len(triple)>4:
                            kb_write.write(triple[1] + "\t" + triple[2] + "\t" + triple[3] + "\n")
                            kb_write.write(triple[1] + "\t" + triple[2] + "\t" + triple[5] + "\n")
                            print(triple[1] + "\t" + triple[2] + "\t" + triple[5] + "\n")
                            entity_list.add(triple[1])
                            entity_list.add(triple[3])
                            entity_list.add(triple[5])

                    elif len(line)>2:
                        qas = line.strip().split("\t")
                        q = qas[0][2:len(qas[0])]
                        a = qas[1]
                        data["q"+str(q_id)] = q
                        data["a"+str(q_id)] = a
                        q_id+=1
                    else:
                        found=False
                        kb_write.close()
                        json.dump(data,diag_write,indent=4)
                        diag_write.close()

                else:
                    found = False
    np.save(incar_conv_dir+"hit_team_maps.npy",hit_team_maps)
    entity_list = list(entity_list)
    for entity in entity_list:
        entity_f.write(entity+"\n")
    entity_f.close()



navigate_to_conv_n_kg()