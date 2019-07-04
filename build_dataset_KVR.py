from os import listdir
from os.path import isfile, join
import os
import json
import re
import numpy as np
from fuzzywuzzy import process, fuzz
from spacy.lang.en.stop_words import STOP_WORDS
import unidecode

# no teams are associated with these file
blacklist = ["30Y6N4AHYQ86V0VH2YQ98LYOE31DRV","3ATYLI1PRUEG193TK2VGHOOW80DOJA","3RHLQY6EDUGZUUTJRMUW0SNEMPID4D","3CVBMEMMXCSDPXG4FGBTUPMM5J0H79","3DWNFENNE47CXGTB1IG73QIRR9D4JG","3H781YYV6UUO3LM92I6OEIHR20YET1","3MYASTQBG8N6JZGHPJ193VOJIHYQD6","3QHITW7OYPL8LSD559A7DUGDXE2QA5","3R16PJFTS43780ZTJ3GJ45I46YF4KH","3SX4X51T81L50G0PQWI3OPLEJCGOAS","3UDTAB6HH7B6SGY1W52S2TEBESM90B","3Y7LTZE0YUY1ETSUYAAB4O4CAJBZUH","3ZQX1VYFTEHRHS7ATYYPSYFISPLO8S"]

teamname2wikiname ={
    "Atletico Madrid": "Atletico Madrid",
    "BVB Dortmund":"BVB Dortmund",
    "Chelsea":"Chelsea",
    "Barcelona":"Barcelona",
    "Bayern Munich": "Bayern Munich",
    "FC. Porto": "FC. Porto",
    "Juventus": "Juventus",
    "Liverpool": "Liverpool",
    "Manchester United": "Manchester United",
    "Paris SG": "Paris SG",
    "Real Madrid": "Real Madrid",
    "Argentina":"Argentina",
    "Belgium":"Belgium",
    "Brazil":"Brazil",
    "Colombia":"Colombia",
    "Croatia":"Croatia",
    "France":"France",
    "Germany":"Germany",
    "Iceland":"Iceland",
    "Italy":"Italy",
    "Mexico":"Mexico",
    "Nigeria":"Nigeria",
    "Portugal":"Portugal",
    "Senegal":"Senegal",
    "Spain":"Spain",
    "Sweden":"Sweden",
    "Switzerland":"Switzerland",
    "Uruguay":"Uruguay",
    "Arsenal":"Arsenal"
}


def get_KG(team_wikiname):
    kg = []
    if os.path.exists("data/KG/clubs/"+team_wikiname+"_kg.txt"):
        f = open("data/KG/clubs/"+team_wikiname+"_kg.txt", encoding="utf-8")
        for line in f:
            row = line.split("\t")
            s,p,o = row[0],row[1],row[2].strip()
            kg.append(['_'.join(i for i in s.split(" ")),'_'.join(j for j in p.split(" ")),'_'.join(k for k in o.split(" "))])
        f.close()
    else:
        f = open("data/KG/country/"+team_wikiname+"_kg.txt", encoding="utf-8")
        for line in f:
            row = line.split("\t")
            s,p,o = row[0],row[1],row[2].strip()
            kg.append(['_'.join(i for i in s.split(" ")),'_'.join(j for j in p.split(" ")),'_'.join(k for k in o.split(" "))])
        f.close()
    #print(kg)
    return kg



def get_conv(file):
    f = open(file,'r',encoding="utf-8")
    f = json.load(f,strict=False)
    conv = []
    i = 1
    while True:
        if "q"+str(i) in list(f.keys()):
            conv.append([f["q"+str(i)],f["a"+str(i)]])
            i+=1
        else:
            break
    return conv, i-1


def generate_ngrams(s, n=[1, 2, 3, 4]):
    words_list = s.split()
    words_list = [w for w in words_list if w not in STOP_WORDS]
    ngrams_list = []

    for num in range(0, len(words_list)):
        for l in n:
            ngram = ' '.join(words_list[num:num + l])
            ngrams_list.append(ngram)
    return ngrams_list


def get_fuzzy_match(object, answer, threshold=0.65):
    """get phrase with highest match in answer"""
    answer_phrase = generate_ngrams(answer)
    if answer_phrase:
        best_match = [fuzz.ratio(object, phr) for phr in answer_phrase]
        if np.max(best_match) > threshold*100:
            return np.max(best_match), answer_phrase[np.argmax(best_match)]

    return 0, ''


def get_relavent_objects(knowledge, ques, ans):
    a_list = set()
    ans = ans
    ques = ques
    for k in knowledge:
        sub_old, pred_old, obj_old = k[0], k[1], k[2]

        ratio, matched_str = get_fuzzy_match((' '.join(pred_old.split("_"))).strip(),ques)
        if ratio>0:
            ques = ques.replace(matched_str,sub_old)

        ratio, matched_str = get_fuzzy_match((' '.join(pred_old.split("_"))).strip(),ques)
        if ratio>0:
            ques = ques.replace(matched_str,pred_old)

        ratio, matched_str = get_fuzzy_match((' '.join(obj_old.split("_"))).strip(),ques)
        if ratio>0:
            ques = ques.replace(matched_str,obj_old)


        ratio, matched_str = get_fuzzy_match((' '.join(sub_old.split("_"))).strip(),ans)
        if ratio>0:
            ans = ans.replace(matched_str,sub_old)

        ratio, matched_str = get_fuzzy_match((' '.join(pred_old.split("_"))).strip(),ans)
        if ratio>0:
            ans = ans.replace(matched_str,pred_old)

        ratio, matched_str = get_fuzzy_match((' '.join(obj_old.split("_"))).strip(),ans)
        if ratio>0:
            ans = ans.replace(matched_str,obj_old)

            if obj_old not in a_list:
                a_list.add(obj_old)

    return ques, ans, list(a_list)




if __name__=="__main__":

    file2team = np.load("soccer_conversations/hit_team_maps.npy",allow_pickle=True).item()


    dataset_types = ["val","train","test"]
    for dataset_type in dataset_types:
        print("Building ",dataset_type," set")
        path = "soccer_conversations/"+dataset_type
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        if dataset_type=="val":
            save2file = open("data/KVR/dev.txt","w+",encoding="utf-8")
        else:
            save2file = open("data/KVR/"+dataset_type+".txt","w+",encoding="utf-8")
        count = 0
        total_dia = 0
        total_qa = 0
        for files in onlyfiles:
            file_id = files[0:files.find(".")]
            #chiking if filename not in blacklist and team in the filename has a corresponding KG
            if file2team[file_id] and (file_id not in blacklist) and (file2team[file_id] in list(teamname2wikiname.keys())):
                knowledge = get_KG(teamname2wikiname[file2team[file_id]])
                conversation,n_qaset = get_conv(path+"/"+files)
                total_dia+=1
                if total_dia%30==0:
                    print(total_dia," so far")
                total_qa+=n_qaset
                save2file.write("#conv#\n")
                for k in knowledge:
                    k[0] = re.sub("\t",' ',k[0])
                    k[0] = re.sub(' +', ' ',k[0])
                    k[1] = re.sub("\t", ' ', k[1])
                    k[1] = re.sub(' +', ' ',k[1])
                    k[2] = re.sub("\t", ' ', k[2])
                    k[2] = re.sub(' +', ' ',k[2])

                    save2file.write("0 "+k[0].strip()+" "+k[1].strip()+" "+k[2].strip()+"\n")
                i = 1
                for a_pair in conversation:

                    modified_q,modified_ans,ans_list_from_kg = get_relavent_objects(knowledge,a_pair[0],a_pair[1])
                    if len(ans_list_from_kg)>0:
                        count+=1
                    modified_q = re.sub("\t", ' ', modified_q)
                    modified_q = re.sub(' +', ' ',modified_q)
                    modified_ans = re.sub("\t", ' ', modified_ans)
                    modified_ans = re.sub(' +', ' ',modified_ans)
                    save2file.write(str(i)+" "+modified_q.strip()+"\t"+modified_ans.strip()+"\t"+str(ans_list_from_kg).strip()+"\n")
                    i+=1
                save2file.write("\n")
        print("in ",dataset_type,total_dia," dialogues.")
        print("in ",dataset_type,total_qa," QA sets")
        print("in ",dataset_type, count," have answers")
