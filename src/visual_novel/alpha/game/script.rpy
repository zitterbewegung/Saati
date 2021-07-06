init python:
     import urllib2

# The script of the game goes in this file.

# Declare characters used by this game. The color argument colorizes the
# name of the character.

define pov = Character("[povname]")

define e = Character("Eileen")


# The game starts here.

label start:

    # Show a background. This uses a placeholder by default, but you can
    # add a file (named either "bg room.png" or "bg room.jpg") to the
    # images directory to show it.

    scene bg room

    # This shows a character sprite. A placeholder is used, but you can
    # replace it by adding a file named "eileen happy.png" to the images
    # directory.

    show eileen happy

    # These display lines of dialogue.
    e "You've created a new Ren'Py game."
    python:
        povname = renpy.input("What is your name?", length=32)
        povname = povname.strip()
        if not povname:
            povname = "Pat Smith"
        url = "http://localhost:8000/engines/completions?prompt=My%20name%20is%20{}&max_tokens=16&temperature=1&top_p=1&top_k=40&n=1&stream=false&echo=false&presence_penalty=0.0001&frequency_penalty=0.0001&best_of=1&recursive_depth=0&recursive_refresh=0".format(povname)
        try:
           urllib2.urlopen(url)
           connected = "yes"
        except:
           connected = "no"
    if connected is "yes":
        pov "My name is [povname]!"
    e "Once you add a story, pictures, and music, you can release it to the world!" 
    # This ends the game.

    return
