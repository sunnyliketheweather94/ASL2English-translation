import requests
from bs4 import BeautifulSoup as bs 
from pytube import YouTube
import sys



def get_video_info(url):
    # download HTML code
    content = requests.get(url)
    # create beautiful soup object to parse HTML
    soup = bs(content.content, "html.parser")
    # initialize the result
    result = {}
    # video title
    result['title'] = soup.find("span", attrs={"class": "watch-title"}).text.strip()
    # channel details
    channel_tag = soup.find("div", attrs={"class": "yt-user-info"}).find("a")
    # channel name
    channel_name = channel_tag.text
    result['name'] = channel_name

    source = YouTube(url)
    en_caption = source.captions.get_by_language_code('en')
    en_caption_convert_to_srt =(en_caption.generate_srt_captions())

    print(en_caption_convert_to_srt)

    #save the caption to a file named Output.txt
    text_file = open(result['title'] + ".txt", "w")
    text_file.write(en_caption_convert_to_srt)
    text_file.close()

    result['captions'] = en_caption_convert_to_srt

    # return the result
    return result


if __name__ == "__main__":
    # get the data
    data = get_video_info(sys.argv[1])
    # print in nice format
    print("Title: {}".format(data['title']))
    print("\nChannel Name: {}".format(data['name']))
    print("Captions:\n{}".format(data['captions']))