#Contains functions
import argparse
import time
import json
import csv
import sys
from random import randint

from selenium import webdriver
#from selenium.webdriver.common.keys import Keys

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup as bs


def _extract_post_text(item):
    print("EXTRACTING POSTS")
    actualPosts = item.find_all(attrs={"data-testid": "post_message"})
    text = ""
    if actualPosts:
        for posts in actualPosts:
            paragraphs = posts.find_all('p')
            text = ""
            for index in range(0, len(paragraphs)):
                text += paragraphs[index].text
    return text

def _extract_comments(item):
    print("EXTRACTING Comments")
    #Facebook
    #AllComments = item.find_all(class_ = "kvgmc6g5 cxmmr5t8 oygrvhab hcukyx3x c1et5uql")
    #Breitbart
    AllComments = item.find_all(class_ = "post-message")
    
    text = ""
    texts = []
    if AllComments:
        for posts in AllComments:
            text = ""
            text = posts.text
            print(text)
            texts.append(text)
            
    return texts

    

def _extract_html(bs_data, page, truth_rating, postBigDict):
    print("EXTRACTING HTML")
    #sys.exit()
    #Add to check
    #Add to check
    #with open('.bs_test.html', "w", encoding="utf-8") as file:
    #    file.write(str(bs_data))


    with open('./bs.html',"w", encoding="utf-8") as file:
        file.write(str(bs_data.prettify()))

    print("Have made BS file")
    # Facebook
    # k = bs_data.find_all(class_="rq0escxv l9j0dhe7 du4w35lb hybvsw6c io0zqebd m5lcvass fbipl8qg nwvqtn77 k4urcfbm ni8dbmo4 stjgntxs sbcfpzgs")
    #Breitbart
    k = bs_data.find_all(class_="post-list")
    print(len(k))

    print("FOUND DATA") 
    for item in k:
        postDict = dict()
        print("Loop ENTERED") 
        #postDict['Post'] = _extract_post_text(item)
        #postDict['Link'] = _extract_link(item)
        #postDict['PostId'] = _extract_post_id(item)
        #postDict['Image'] = _extract_image(item)
        #postDict['Shares'] = _extract_shares(item)
        #postDict['postID'] = ID
        postDict['PostURL'] = page
        postDict['TruthRating'] = truth_rating
        postDict['Comments'] = _extract_comments(item)
        # postDict['Reaction'] = _extract_reaction(item)

        #Add to check
        print('Just AFTER READING POST MAYBE')
        postBigDict.append(postDict)
        
        
        #Cheap way of checking only taking first ITEM (DESIRED ITEM)
        break

    return postBigDict

def _login(browser, email, password):
    time.sleep(5)
    browser.get("http://facebook.com")


    browser.maximize_window()
    browser.implicitly_wait(10)
    time.sleep(5)

    browser.find_element(By.XPATH, '//button[text()="Allow All Cookies"]').click()
    
    wait = WebDriverWait(browser, 30)
    email_field = wait.until(EC.visibility_of_element_located((By.NAME, 'email')))
    email_field.send_keys(email)
    pass_field = wait.until(EC.visibility_of_element_located((By.NAME, 'pass')))
    pass_field.send_keys(password)
    #browser.find_element(By.ID,"email").send_keys(email)
    #browser.find_element(By.ID, "pass").send_keys(password)
    browser.find_element(By.XPATH, '//button[text()="Log In"]').click()
    time.sleep(5)
    #browser.implicitly_wait(1000)
    print("LOGGED IN!!!")
    #sys.exit()

def _cookie_request(browser):
    time.sleep(5)
    browser.get("https://www.palmerreport.com/")


    browser.maximize_window()
    browser.implicitly_wait(10)
    time.sleep(5)
    browser.find_element(By.XPATH, '/html/body/div[1]/div/div/div/div[2]/div/button[2]').click()


def extract(page, truth_rating, numOfPost,EMAIL , PASSWORD,infinite_scroll=False, scrape_comment=False ):
    option = Options()
    option.add_argument("--disable-infobars")
    option.add_argument("start-maximized")
    option.add_argument("--disable-extensions")

    # Pass the argument 1 to allow and 2 to block
    option.add_experimental_option("prefs", {
        "profile.default_content_setting_values.notifications": 1
    })
    #option.add_experimental_option('excludeSwitches', ['enable-logging'])

    # chromedriver should be in the same folder as file
    browser = webdriver.Chrome(chrome_options=option)
    

    #Facebook
    #_login(browser, EMAIL, PASSWORD)
    #sys.exit()

    #Palmer Report
    #_cookie_request(browser)




    postBigDict = list()

    for i in range(len(page)):

        #time.sleep(1)
        browser.get(page[i])
        time.sleep(3)

        #post = browser.find_element(By.CLASS_NAME,'kvgmc6g5 cxmmr5t8 oygrvhab hcukyx3x c1et5uql ii04i59q').text




        '''
        #-------------------------------BREITBART-----------------------
        #Comment this out for facebook comment sections
        #Locating element by link text and store in variable "Element"        		
        comment_element = browser.find_element(By.ID, "disqus_thread")


        # Scrolling down the page till the element is found		
        browser.execute_script("arguments[0].scrollIntoView();", comment_element)
        time.sleep(5)


        try:
            comment_element = browser.find_element(By.ID, "disqus_thread")
            comment_element = comment_element.find_element(By.XPATH, "/html/body/div[2]/div/div/section/section/div[1]/iframe[1]")
            #print(comment_element)
            Comment_Url_src = comment_element.get_attribute("src")
        except NoSuchElementException:
            with open('./postBigDict.json','w', encoding='utf-8') as file:
                file.write(json.dumps(postBigDict, ensure_ascii=False).encode('utf-8').decode())
            time.sleep(5)
            browser.close()

            return postBigDict_Final

        #time.sleep(2)
        browser.get(Comment_Url_src)
        time.sleep(3)
        #________________________END of BreitBart________________________________
        '''




        '''
        #--------------------Palmer Report----------------------------------------
         #Comment this out for facebook comment sections
        #Locating element by link text and store in variable "Element"        		
        comment_element = browser.find_element(By.ID, "disqus_thread")


        # Scrolling down the page till the element is found		
        browser.execute_script("arguments[0].scrollIntoView();", comment_element)
        time.sleep(5)


        try:
            comment_element = browser.find_element(By.ID, "disqus_thread")
            #/html/body/div[3]/div/div/div/div[1]/div/iframe[1]
            comment_element = comment_element.find_element(By.XPATH, "/html/body/div[3]/div/div/div/div[1]/*/iframe[1]")
            #print(comment_element)
            Comment_Url_src = comment_element.get_attribute("src")
        except NoSuchElementException:
            print("EXCEPTION FAILED TO FIND COMMENTS")
            with open('./postBigDict.json','w', encoding='utf-8') as file:
                file.write(json.dumps(postBigDict, ensure_ascii=False).encode('utf-8').decode())
            time.sleep(5)
            browser.close()

            return postBigDict_Final

        #time.sleep(2)
        browser.get(Comment_Url_src)
        time.sleep(3)

        #_________________________End of Palmer Report____________________________
        '''


        #----------------------------ABC News--------------------------------------



        #Commens must first be accessed by clicking comment button
        # 
        #This removes need to scroll        		

        


        try:

            #browser.find_element(By.XPATH, '//*[@id="fitt-analytics"]/div/main/div[2]/section/div[1]/div/div/div[2]/div/section/article/div[2]/div[1]/div/button').click()
            browser.find_element(By.XPATH, '//div[@class="CommentButton"]/button').click()
            
            time.sleep(5)
            #Button Button--default Button--icon CommentButton__Button

        except NoSuchElementException:
            print("EXCEPTION FAILED TO FIND Button for COMMENTS")
            with open('./postBigDict.json','w', encoding='utf-8') as file:
                file.write(json.dumps(postBigDict, ensure_ascii=False).encode('utf-8').decode())
            time.sleep(5)
            browser.close()

        try:

            #comment_element = browser.find_element(By.ID, "disqus_thread")
            #/html/body/div[3]/div/div/div/div[1]/div/iframe[1]
            #comment_element = comment_element.find_element(By.XPATH, "/html/body/div[2]/div/div/section/div/div/div[3]/iframe[1]")
            #print(comment_element)
            comment_element = browser.find_element(By.XPATH, '//div[@id="disqus_thread"]/iframe[1]')
            Comment_Url_src = comment_element.get_attribute("src")
        except NoSuchElementException:
            print("EXCEPTION FAILED TO FIND COMMENTS")
            with open('./postBigDict.json','w', encoding='utf-8') as file:
                file.write(json.dumps(postBigDict, ensure_ascii=False).encode('utf-8').decode())
            time.sleep(5)
            browser.close()

            return postBigDict_Final

        #time.sleep(2)
        browser.get(Comment_Url_src)
        time.sleep(3)




        #______________________________End of ABC news______________________________________________





    
        # Now that the page is fully scrolled, grab the source code.
        source_data = browser.page_source
        #with open('.bs_test_other.html', "w", encoding="utf-8") as file:
        #    file.write(str(source_data))

        # Throw your source into BeautifulSoup and start parsing!
        bs_data = bs(source_data, 'html.parser')
        #browser.close()
        postBigDict_Final = _extract_html(bs_data, page[i], truth_rating[i], postBigDict )
        time.sleep(randint(1,3))

        if(i > 230):
            break


    with open('./postBigDict.json','w', encoding='utf-8') as file:
            file.write(json.dumps(postBigDict, ensure_ascii=False).encode('utf-8').decode())
    time.sleep(100)
    browser.close()

    return postBigDict_Final