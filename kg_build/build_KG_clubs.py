import csv
import re
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

teamname_mapping = {
    "Atl%C3%A9tico_Madrid":"Atletico Madrid",
    'Arsenal_F.C.':'Arsenal',
    'FC_Barcelona':'Barcelona',
    'Real_Madrid_C.F.':"Real Madrid",
    'Juventus_F.C.':'Juventus',
    'Manchester_United_F.C.':'Manchester United',
    'Paris_Saint-Germain_F.C.':'Paris SG',
    'Liverpool_F.C.':'Liverpool',
    'Chelsea_F.C.':'Chelsea',
    'FC_Bayern_Munich':'Bayern Munich',
    'FC_Porto':'FC. Porto',
    'Borussia_Dortmund':'BVB Dortmund',
    'A.C._Milan':'AC Milan'
}



IGONRE_PLAYER_STATE = ["INJ","OTH","SUS","PRE","RET"]

#players.append(['Number', 'Position', 'Name', 'Date of birth', 'height', 'senior career caps', 'senior career goals'])
#                  0         1           2         3               4              5                      6

def create_KG(players_info,team_name,side_bar):
    team_name= team_name[team_name.rfind("/")+1:]
    team_name = teamname_mapping[team_name]
    file = io.open("data/KG/clubs/"+team_name + "_kg.txt", "w", encoding="utf-8")

    # (club_name, "ground", name_of_ground)
    file.write(team_name + "\t" + "ground" + "\t" + side_bar[0] + "\n")

    # (name_of_ground, "capacity", capacity)
    file.write(side_bar[0] +"\t"+ "capacity" + "\t"+ side_bar[3] +"\n")

    # (club_name, "chairman", name_of_chairman)
    file.write(team_name + "\t" + "chairman" + "\t" + side_bar[1] + "\n")

    # (club_name, "coach", name_of_coach)
    file.write(team_name + "\t" + "coach" + "\t" + side_bar[2] + "\n")

    # Saving infos related to players
    for player in players_info:
        if player[5]:

            if int(player[5])>5:

                name_of_player = player[2]
                if ")" in name_of_player:
                    name_of_player=name_of_player[0:name_of_player.find(")")+1].strip()

                # (club_name, "has a player, name_of_player)
                file.write(team_name + "\t" + "has player" + "\t" + name_of_player+"\n")

                # (name_of_player, "position", position)
                file.write(name_of_player + "\t" + "position" + "\t" + map_position[player[1]]+"\n")

                # (name_of_player, "goals", goals)
                if str(player[6]):
                    goals = str(player[6])
                    goals = goals[1:len(goals)-1]
                    file.write(name_of_player + "\t" + "goals" + "\t" + goals+"\n")

                # (name_of_player, "caps", caps)
                file.write(name_of_player + "\t" + "caps" + "\t" + str(player[5])+"\n")

                # (name_of_player, "date of birth", data_of_birth)
                file.write(name_of_player + "\t" + "date of birth" + "\t" + str(player[3])+"\n")

                # (name_of_player, "jersey", jersey no)
                if str(player[0]):
                    file.write(name_of_player + "\t" + "jersey" + "\t" + str(player[0])+"\n")

                # (name_of_player, "height", height)
                file.write(name_of_player + "\t" + "height" + "\t" + str(player[4])+"\n")

    file.close()
    print(team_name+" Done !")


def filter_name(name):
    w = name.split(" ")
    if w[len(w)-1] in IGONRE_PLAYER_STATE:
        del(w[len(w)-1])

    last_elem = w[len(w)-1]
    for i in IGONRE_PLAYER_STATE:
        if i in last_elem:
            w[len(w)-1] = last_elem[:len(last_elem)-3]
    return ' '.join(a for a in w)

# Getting players profile details
def player_info(profile_link,club):
    base_url = "https://en.wikipedia.org"
    url = base_url+profile_link
    url = requests.get(url)
    page_content = BeautifulSoup(url.content, 'html.parser')

    date_of_birth = ""
    height = ""
    senior_career_apps = 0
    senior_carrer_goals = 0


    tables = page_content.find_all('table',{'class':'infobox vcard'})[0]


    trs = tables.find_all('tr')
    found_senior_carrer = False
    for tr in trs:
        if tr.find_all('th'):
            if tr.find_all('th')[0].get_text().strip()=="Date of birth":
                date_of_birth = tr.find_all('span',{'class':'bday'})[0].get_text().strip()
                #print(date_of_birth)
            if tr.find_all('th')[0].get_text().strip()=="Height":
                height = tr.find_all('td')[0].get_text().strip()
                height = height[0:height.find("m")+1]

                if "ft" in height:
                    height = height[height.find("(")+1:]

                height = (' '.join(height.split("\xa0"))).strip()

            if tr.find_all('th')[0].get_text().strip()=="Senior career*":
                found_senior_carrer = True
            if "National team" in tr.find_all('th')[0].get_text().strip():
                break

            if found_senior_carrer:
                if tr.find_all('td'):
                    if tr.find_all('td')[0].get_text().strip()=="team":
                        continue
                    else:
                        if tr.find_all('td')[0].find("a"):
                            club_url = tr.find_all('td')[0].find("a")
                            #print(tr.find_all('td')[1].get_text().strip())
                            if club_url.attrs["href"] == club:
                                senior_career_apps = int(tr.find_all('td')[1].get_text().strip())
                                goals = tr.find_all('td')[2].get_text().strip()
                                senior_career_goals = int(goals[1:len(goals)-1])
                                senior_carrer_goals = goals

    return date_of_birth, height, senior_career_apps, senior_carrer_goals


def side_bar_info(wiki_link):
    url = requests.get(wiki_link)
    page_content = BeautifulSoup(url.content, 'html.parser')

    table = page_content.find_all('table', {'class': 'infobox vcard'})[0]
    trs = table.find('tbody').find_all('tr')
    head_coach = ""
    president= ""
    ground = ""
    capacity  = ""
    for tr in trs:

        if tr.find_all('th'):
            if tr.find_all('th')[0].get_text().strip()=="Manager":
                head_coach = tr.find_all('td')[0].get_text().strip()

            if tr.find_all('th')[0].get_text().strip()=="Head coach":
                head_coach = tr.find_all('td')[0].get_text().strip()

            if tr.find_all('th')[0].get_text().strip()=="President":
                president = tr.find_all('td')[0].get_text().strip()

            if tr.find_all('th')[0].get_text().strip()=="Chairman":
                president = tr.find_all('td')[0].get_text().strip()

            if tr.find_all('th')[0].get_text().strip()=="Ground":
                ground = tr.find_all('td')[0].get_text().strip()
            if tr.find_all('th')[0].get_text().strip()=="Capacity":
                capacity = tr.find_all('td')[0].get_text().strip()
                if "[" in capacity:
                    capacity = capacity[0:capacity.find("[")]

    return ground, head_coach, president, capacity

# get current squad of a football club
def fetch_current_squad(club):

    base_url = 'https://en.wikipedia.org/wiki/'
    wiki_link = base_url+club
    team_name = wiki_link[wiki_link.rfind("/") + 1:]
    url = requests.get(wiki_link)
    page_content = BeautifulSoup(url.content, 'html.parser')
    ground, head_coach, president ,capacity =  side_bar_info(wiki_link)
    side_bar = [ground,president,head_coach,capacity]
    tables = page_content.find_all('tr')

    found = False
    players = []
    #players.append(['Number','Position','Name', 'Date of birth', 'height', 'senior career caps', 'senior career goals'])
    #                  0         1        2         3               4              5                      6

    for table in tables:
        if not found:
            tbs = table.find_all('tr', {'class': 'vcard agent'})

            if tbs:
                for tb in tbs:
                    found = True
                    tds = tb.find_all('td')
                    num = tds[0].get_text().replace("\n","").strip()
                    pos = tds[2].get_text().replace("\n","").strip()
                    name = tds[3].get_text().replace("\n","").strip()

                    if "(" in name:
                        name = name[0:name.find("(")].strip()

                    tds = tds[3].find('a')
                    dob = ""
                    height = ""
                    senior_caps = ""
                    senior_goals = ""
                    if(tds):
                        if "cnote" not in tds.attrs["href"]:
                            player_link = tds.attrs["href"]
                            dob,height,senior_caps,senior_goals = player_info(player_link, "/wiki/"+club)
                    num = unidecode.unidecode(num)
                    pos = unidecode.unidecode(pos)
                    name = unidecode.unidecode(name)
                    players.append([num,pos,name, dob,height,senior_caps,senior_goals,ground])

        if found:
            break

    create_KG(players,club,side_bar)





if __name__ == "__main__":

    base_url = 'https://en.wikipedia.org/wiki/'


    club_teams = ['Paris_Saint-Germain_F.C.','Liverpool_F.C.','Atl%C3%A9tico_Madrid','Arsenal_F.C.','FC_Barcelona', 'Real_Madrid_C.F.', 'Juventus_F.C.', 'Manchester_United_F.C.','Chelsea_F.C.','FC_Bayern_Munich', 'FC_Porto', 'Borussia_Dortmund','A.C._Milan']

    for club in club_teams:
        fetch_current_squad(club)
