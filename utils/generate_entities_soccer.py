import os
import io
from unidecode import unidecode


clubs = 'data/KG/clubs/'
country = 'data/KG/country/'

cl = os.listdir(clubs)
cn = os.listdir(country)

entities = set()

for f in cl:
    fl = io.open(clubs+f,"r",encoding="utf-8")
    for line in fl:
        entities.add(unidecode('_'.join(line.split("\t")[2].strip().split())))
        entities.add(unidecode('_'.join(line.split("\t")[0].strip().split())))

    fl.close()



for f in cn:
    fl = io.open(country+f,"r",encoding="utf-8")
    for line in fl:
        entities.add(unidecode('_'.join(line.split("\t")[2].strip().split())))
        entities.add(unidecode('_'.join(line.split("\t")[0].strip().split())))
    fl.close()

unique_entities = list(entities)

with io.open("data/soccer_entities.txt","w",encoding="utf-8") as f:
    for ent in unique_entities:
        if ent!='':
            f.write(ent+"\n")

f.close()
