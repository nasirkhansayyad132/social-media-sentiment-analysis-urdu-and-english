
#import Facebook_scraper class from facebook_page_scraper
from facebook_page_scraper import Facebook_scraper
import json
from langdetect import detect

def fetch_page_posts(page_name,posts_count = 4):
    all_posts = []
    browser = "firefox"
    proxy = "IP:PORT" #if proxy requires authentication then user:password@IP:PORT
    timeout = 600 #600 seconds
    headless = True
    #instantiate the Facebook_scraper class
    #Below is raw data saved of 10 posts from a certain page
    meta_ai = Facebook_scraper(page_name, posts_count, browser, proxy=proxy, timeout=timeout, headless=headless)
   
    json_data = meta_ai.scrap_to_json() #All data about posts is converted to json
    dataDic = json.loads(json_data)#json data is converted to dictionary
    for i in range(len(dataDic)): #Iterat over dictionary
        
        if((detect(dataDic[list(dataDic.keys())[i]]['content']))=='en'):
            all_posts.append((dataDic[list(dataDic.keys())[i]]['content'])) #dic keys are converted to list to iterate over dictionary
        i=i+1   #only posts in dictionary are stored in list
    
    return all_posts
# mypost= fetch_page_posts('ImranKhanOfficial')
# print(mypost)