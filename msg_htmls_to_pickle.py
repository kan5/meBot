import os
from bs4 import BeautifulSoup
import pandas as pd
import time
start_time = time.time()


# '13 янв 2009 21:20:20' -> SECONDS_FROM_2004
def time_refractor(date_string):
    start = 2004
    date_string = date_string.split()

    day = int(date_string[0])
    mon_days = month_str_to_int_days(date_string[1])
    year = int(date_string[2])

    year = year - start
    for i in range(year-1):
        if i%4 == 0:
            day = day + 1
    if (year%4 == 0) and (mon_days>60):
        day = day + 1

    day = day + mon_days
    in_day = date_string[4].split(':')

    return year*31536000 + day*86400 + int(in_day[0])*3600 + int(in_day[1])*60 + int(in_day[2])

def month_str_to_int_days(name):
    switcher = {
        'янв': 0,
        'фев': 31,
        'мар': 59,
        'апр': 90,
        'мая': 120,
        'июн': 151,
        'июл': 181,
        'авг': 212,
        'сен': 243,
        'окт': 273,
        'ноя': 304,
        'дек': 334,
    }

    return switcher.get(name)


def get_messages():
    # signs:
    # - is private +
    # - is_club +
    # - is_event +
    # - is_conversation +
    # - chat id +
    # - chat name +
    # - sender name +
    # - date-item -
    # - date +
    # - text +
    # - links_in_text +
    # - is redacted +
    # - profile link +

    big_list = []

    # making main path to messages
    st_dir = os.getcwd()
    main_messages_dir = st_dir + "\\Archive\\messages\\"
    os.chdir(main_messages_dir)

    # check every folder in messages
    for i in os.listdir()[:]:

        # NOT need in file with paths
        pt = os.path.join(main_messages_dir, i)
        if os.path.isfile(pt):
            continue
        os.chdir(os.path.join(pt))

        # returned list
        list_dick = []
        # one element of list_dick
        dick = {}
        # is private default
        ispivat = 0
        isevent = 0
        isclub = 0
        isconvers = 0
        # opening every file in chat dir
        for j in os.listdir():
            f = open(j, 'r')
            # parser object
            soup = BeautifulSoup(f, 'html.parser')
            # chat id
            dick['chat_id'] = i
            # chat name
            dick['chat_name'] = soup.find('div', class_='ui_crumb').get_text()
            # more narrow object
            soup = soup.find(class_= "wrap_page_content")
            soup = soup.find_all(class_="message")
            # loop for each message
            for k in soup:
                # list that give name, date, is_redacted
                ob = k.find('div', class_='message__header').get_text().split(',')
                ob[1] = ob[1][1:]
                # is redacted
                dick['is_redacted'] = 0
                if len(ob[1])>22:
                    dick['is_redacted'] = 1
                # date in seconds from 2004
                dick['date'] = time_refractor(ob[1])
                # sender name
                dick['sender_name'] = ob[0]
                # date item
                dick['date_item'] = int(k.get_attribute_list('data-id')[0])
                # is private
                if (ispivat==0)and(ob[0] == dick['chat_name']):
                    ispivat = 1
                # profile_link
                dick['profile_link'] = ''
                if (dick['sender_name'] != 'Вы')and(k.find('a') is not None):
                    dick['profile_link'] = k.find('a').get_attribute_list('href')[0]
                # is club
                if (isclub == 0)and('club' in dick['profile_link'][15:]):
                    isclub = 1
                elif (isevent == 0)and('event' in dick['profile_link'][15:]):
                    isevent = 1
                # text
                dick['text'] = ''
                dick['text'] = k.find('div', class_='').get_text()
                # links in text
                dick['links_in_text'] = []
                at = k.find_all(class_='attachment__link')
                if at is not None:
                    for h in at:
                        dick['links_in_text'].append(h.get_attribute_list('href')[0])

                list_dick.append(dick.copy())
            f.close()
        if (ispivat == 1):
            if isclub==0:
                if isevent==1:
                    ispivat = 0
            else:
                ispivat = 0
                isevent = 0
        else:
            isconvers = 1
        for c in list_dick:
            c['is_private'] = ispivat
            c['is_event'] = isevent
            c['is_conversation'] = isconvers
            c['is_club'] = isclub
            big_list.append(c)

        os.chdir(main_messages_dir)
    os.chdir(st_dir)
    df = pd.DataFrame(big_list)
    return df


df = get_messages()
df.sort_values(by=['chat_name'], inplace=True, ascending=True)
df = df.reset_index(drop=True)
df.to_pickle('msg1.pickle')


print("--- %s seconds ---" % (time.time() - start_time))