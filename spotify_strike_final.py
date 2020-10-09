import requests
import secrets
from random import random

# token generated from website:
# https://developer.spotify.com/console/get-users-available-devices/
token = secrets.token

def generate_random(l):
    return random()*l

AUTH_URL = 'https://accounts.spotify.com/api/token'



def get_trackid(mood):
    trackid=""
    if(mood=="Angry"):
        #don't back in anger
        trackid = "0UvCh63URrLFcPkKt99hHd"
    elif(mood=="Disgust"):
        #ignorance
        trackid="7pdvxSHmfXKRgXKTCcs5Hs"
    elif(mood=="Fear"):
        #back in my body
        trackid="0DlMSBJh9uo52DieTYVXLw"
    elif(mood=="Neutral"):
        #light on
        trackid="6UnCGAEmrbGIOSmGRZQ1M2"
    elif(mood=="Happy"):
        #havana
        trackid="1rfofaqEpACxVEHIZBJe6W"
    elif(mood=="Surprise"):
        #only the young
        trackid="2slqvGLwzZZYsT4K4Y1GBC"
    elif(mood=="Sad"):
        #dog years - we will be alright
        trackid="5RRNZFyOi17nTh2bPEKPtp"
    return trackid


def play_song(mood):
    trackid = get_trackid(mood)
    print(trackid)

    # authorizing spotify
    token = secrets.token
    headers = {
    'Authorization': 'Bearer {token}'.format(token=token)
    }

    # base URL of all Spotify API endpoints
    BASE_URL = 'https://api.spotify.com/v1/'

    #placing track in the play queue
    post_trackid = requests.post("https://api.spotify.com/v1/me/player/queue?uri=spotify:track:"+
              trackid,headers=headers)
    print(post_trackid)
    #playing next
    put_next = requests.post("https://api.spotify.com/v1/me/player/next",
                             headers=headers)
    print(put_next)
    
    #playing the song
    play_resume = requests.put("https://api.spotify.com/v1/me/player/play",
                               headers=headers)

    print(play_resume)


# play_song('Sad')
    

    
    

    
        
        

        
    
