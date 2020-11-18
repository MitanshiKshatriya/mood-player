# mood-player
Uses CNN to detect your mood (facial expression) and plays songs from spotify according to mood detected.

### generating api key

- go to link - https://developer.spotify.com/console/post-queue/

- click on get token

- inside the popup select scopes click on all checkboxes , then click request token

- copy the token

- create a new file name it secrets.py. Add the newly generated token in the file as - 

~~~
token = "your copied token"
~~~

### setup spotify

- play any one track on your spotify webplayer/app

### run the python program

- open a command prompt and type the following
~~~
cd mood-player
python moodeDetection.py webcam 25
~~~

- press space bar to make change the music according to your mood.
