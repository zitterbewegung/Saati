import urllib
import urllib2

url = "http://........."
opener = urllib2.build_opener(urllib2.HTTPHandler(debuglevel=1))
data = urllib.urlencode({'name' : playerName,'score' : playerScore})
userList = str(opener.open(url, data=data).read())
