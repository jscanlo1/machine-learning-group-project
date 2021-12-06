#this file aims to create a workable csv or excel file by scraping
#the Facebook comment section in the BuzzFeed facebook dataset.
import argparse
import time
import json
import csv
import sys

from selenium import webdriver
#from selenium.webdriver.common.keys import Keys

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup as bs
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import Scraper_Functions as scrape_functions
import pandas as pd

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Web Scraper for Facebook Comments" + \
    #        "posts, reactions, and (optionally) comments on a public " + \
    #        "Facebook group or page.")
    #

    parser = argparse.ArgumentParser(description="Facebook Page Scraper")
    required_parser = parser.add_argument_group("required arguments")
    required_parser.add_argument('-page', '-p', help="The Facebook Public Page you want to scrape", required=True)
    required_parser.add_argument('-len', '-l', help="Number of Posts you want to scrape", type=int, required=True)
    optional_parser = parser.add_argument_group("optional arguments")
    optional_parser.add_argument('-infinite', '-i',
                                 help="Scroll until the end of the page (1 = infinite) (Default is 0)", type=int,
                                 default=0)
    optional_parser.add_argument('-usage', '-u', help="What to do with the data: "
                                                      "Print on Screen (PS), "
                                                      "Write to Text File (WT) (Default is WT)", default="CSV")

    optional_parser.add_argument('-comments', '-c', help="Scrape ALL Comments of Posts (y/n) (Default is n). When "
                                                         "enabled for pages where there are a lot of comments it can "
                                                         "take a while", default="No")
    args = parser.parse_args()

    infinite = False
    if args.infinite == 1:
        infinite = True

    scrape_comment = False
    if args.comments == 'y':
        scrape_comment = True

    


    #Read in URLs and CSV files.

    #df = pd.read_csv("facebook-fact-check.csv")
    #df = pd.read_csv("Palmer_Articles.csv")
    df = pd.read_csv("ABC_Articles.csv")
    print(df.head())

    #Facebook
    '''
    postID = df.iloc[:,1]
    Post_Links = df.iloc[:,4]
    Truth_Rating = df.iloc[:,7]
    '''
    #Breitbart
    Post_Links = df.iloc[:,0]
    Truth_Rating = df.iloc[:,1]
    postID = "11111111"


    print(Post_Links.head())
    print(Truth_Rating.head())

    with open('facebook_credentials.txt') as file:
        EMAIL = file.readline().split('"')[1]
        PASSWORD = file.readline().split('"')[1]
    print(EMAIL)
    print(PASSWORD)
    

    print('WORKINGGGG: ', args.page, "   ---   ",args.len)


    #sys.exit()
    postBigDict = scrape_functions.extract(page=Post_Links, truth_rating = Truth_Rating, numOfPost=args.len, EMAIL = EMAIL, PASSWORD = PASSWORD, infinite_scroll=infinite, scrape_comment=scrape_comment)

    #TODO: rewrite parser
    if args.usage == "WT":
        with open('output.txt', 'w') as file:
            for post in postBigDict:
                file.write(json.dumps(post))  # use json load to recover

    elif args.usage == "CSV":
        with open('data.csv', 'w',) as csvfile:
           writer = csv.writer(csvfile)
           #writer.writerow(['Post', 'Link', 'Image', 'Comments', 'Reaction'])
           writer.writerow(['Post', 'Link', 'Image', 'Comments', 'Shares'])

           for post in postBigDict:
               print(post['Post'])
              #writer.writerow([post['Post'], post['Link'],post['Image'], post['Comments'], post['Shares']])
              #writer.writerow([post['Post'], post['Link'],post['Image'], post['Comments'], post['Reaction']])

    else:
        for post in postBigDict:
            print(post)

    print("Finished")




    





    



'''
X1=df.iloc[:,0]
X2=df.iloc[:,1]
X=np.column_stack((X1,X2))
y=df.iloc[:,2]'''