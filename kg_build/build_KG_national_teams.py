import csv
import io
import unidecode
import requests
from bs4 import BeautifulSoup


map_position = {
    "GK" : "goalkeeper",
    "DF" : "defender",
    "MF" : "midfielder",
    "FW" : "forward"
}

IGONRE_PLAYER_STATE = ["INJ","OTH","SUS","PRE","RET"]


def team_details(team_name):
    base_url = 'https://en.wikipedia.org/wiki/'
    wiki_link = base_url+team_name
    team_name = wiki_link[wiki_link.rfind("/") + 1:]
    url = requests.get(wiki_link)
    page_content = BeautifulSoup(url.content, 'html.parser')


    table = page_content.find_all('table', {'class': 'infobox'})[0]
    trs = table.find('tbody').find_all('tr')
    infos = [['head coach','captain','most caps','world cup appearances','world cup best results']]
    head_coach=""
    captain = ""
    appearances = ""
    best_result_type= ""
    best_result_years = ""
    for tr in trs:
        if tr.find_all('th'):
            if tr.find_all('th')[0].get_text().strip()=="Head coach":
                head_coach = tr.find_all('td')[0].get_text().strip()
                #print(head_coach)

            if tr.find_all('th')[0].get_text().strip()=="Captain":
                captain = tr.find_all('td')[0].get_text().strip()
                #print(captain)

            if tr.find_all('th')[0].get_text().strip()=="Most caps":
                most_cap = tr.find_all('td')[0].get_text().strip()
                #print(most_cap)

            if tr.find_all('th')[0].get_text().strip()=="Appearances":
                appearances = tr.find_all('td')[0].get_text().strip()
                #print(appearances)

            if tr.find_all('th')[0].get_text().strip()=="Best result":

                best_result = tr.find_all('td')[0].get_text().strip()
                best_result_type = best_result[:best_result.find("(")].strip()
                best_result_years = best_result[best_result.find("(")+1:best_result.find(")")].split(",")
                break
    #print(head_coach,captain,appearances,best_result)
    return head_coach, captain, appearances, best_result_type, best_result_years


# Create and Save knowledge graph
def create_KG(players_info,team_name):

    head_coach, captain, World_Cup_appearances, World_Cup_best_result_type, World_Cup_best_results = team_details(team_name)
    # Saving infos related to Team
    team_name= team_name[:team_name.find("_")]
    country = team_name
    file = io.open("../data/KG/country/"+team_name + "_kg.txt", "w", encoding="utf-8")
    # (country,"coach",head_coach)
    file.write(country + "\t" + "coach" + "\t" + head_coach+"\n")

    # (country,"captain",captain)
    file.write(country + "\t" + "captain" + "\t" + captain+"\n")

    # (country,"World cup appearances",appearance)
    file.write(country + "\t" + "world cup appearance" + "\t" + World_Cup_appearances+"\n")

    # (country,"world cup + result type",result years)
    for year in World_Cup_best_results:
        file.write(country+"\t" + "world cup "+World_Cup_best_result_type + "\t" + year+"\n")


    # Saving infos related to players
    for player in players_info:

        # (country,"position",name_of_player)
        file.write(country+"\t"+map_position[player[1]]+"\t"+player[2]+"\n")

        # (counry, "has player", position)
        file.write(country + "\t" + "has player" + "\t" + player[2]+"\n")

        # (name_of_player, "position", position)
        file.write(player[2] + "\t" + "position" + "\t" + map_position[player[1]]+"\n")

        # (name_of_player, "goals", goals)
        file.write(player[2] + "\t" + "goals" + "\t" + str(player[5])+"\n")

        # (name_of_player, "caps", caps)
        file.write(player[2] + "\t" + "caps" + "\t" + str(player[4])+"\n")

        # (name_of_player, "age", age)
        file.write(player[2] + "\t" + "age" + "\t" + str(player[3])+"\n")

        # Since only current squad players have jersey number
        if player[6]=="current squad":
            # (name_of_player, "goals", goals)
            if len(str(player[0]))>0:
                file.write(player[2] + "\t" + "jersey" + "\t" + str(player[0])+"\n")

    file.close()
    print(country," DONE")


def filter_name(name):
    w = name.split(" ")
    if w[len(w)-1] in IGONRE_PLAYER_STATE:
        del(w[len(w)-1])

    last_elem = w[len(w)-1]
    for i in IGONRE_PLAYER_STATE:
        if i in last_elem:
            w[len(w)-1] = last_elem[:len(last_elem)-3]
    return ' '.join(a for a in w)



def current_squad(team_name):
    base_url = 'https://en.wikipedia.org/wiki/'
    wiki_link = base_url+team_name
    url = requests.get(wiki_link)
    page_content = BeautifulSoup(url.content, 'html.parser')


    # filtered_squad format:   ['No.', 'Pos', 'Player', 'Age', 'Caps', 'Goals','type_of_player']
    #                            0      1         2       3      4        5         6
    filtered_squad = []

    current_squad_table = []
    tables = page_content.find_all('table')
    found_current_suqad = False
    for table in tables:
        # searching for Recent call-ups table
        if found_current_suqad and table.find_all('tr',{'class':'nat-fs-player'}):
            players_table = table.find_all('tr',{'class':'nat-fs-player'})
            for player_row in players_table:
                player_info = player_row.find_all('td')
                pos = player_info[0].find_all('a')[0].get_text().strip()
                pos = player_info[0].find_all('a')[0].get_text().strip()
                name = player_row.find_all('th')[0].get_text().strip()
                name = unidecode.unidecode(name)
                if "(" in name:
                    name = name[0:name.find("(")]

                name = filter_name(name)
                caps = int(player_info[2].get_text().strip())
                goals = int(player_info[3].get_text().strip())
                if caps>=50:
                    age = player_info[1].get_text().strip()
                    age = int(age[age.rfind("(age") + len("(age "):age.rfind(")")])
                    filtered_squad.append(["", pos, name, age, caps, goals,'recent call-ups'])
            break

        # searching for Current Squad
        if table.find_all('tr',{'class':'nat-fs-player'}):
            found_current_suqad = True
            players_table = table.find_all('tr',{'class':'nat-fs-player'})
            for player_row in players_table:
                player_info = player_row.find_all('td')
                num = player_info[0].get_text().strip()
                pos = player_info[1].find_all('a')[0].get_text().strip()
                pos = player_info[1].find_all('a')[0].get_text().strip()
                name = player_row.find_all('th')[0].get_text().strip()
                name = unidecode.unidecode(name)
                if "(" in name:
                    name = name[0:name.find("(")]

                name = filter_name(name)

                age = player_info[2].get_text().strip()
                age = int(age[age.rfind("(age")+len("(age "):age.rfind(")")])
                caps = int(player_info[3].get_text().strip())
                goals = int(player_info[4].get_text().strip())
                #club = player_info[5].get_text().strip()
                #club= unidecode.unidecode(club)
                if caps>=5:
                    filtered_squad.append([num,pos,name,age,caps,goals,'current squad'])

    # Creating and saving Knowledge Graph for the given team
    create_KG(filtered_squad, team_name)





national_teams = ['Brazil_national_football_team','Germany_national_football_team','Argentina_national_football_team', 'Sweden_national_football_team', 'Switzerland_national_football_team', 'Iceland_national_football_team', 'Belgium_national_football_team', 'France_national_football_team', 'Spain_national_football_team', 'Uruguay_national_football_team',
                    'Mexico_national_football_team', 'Colombia_national_football_team', 'Italy_national_football_team', 'Croatia_national_football_team', 'Senegal_national_football_team', 'Nigeria_national_football_team', 'Portugal_national_football_team']


for team in national_teams:
    current_squad(team)
