import os
import time

from pytube import YouTube
from moviepy.editor import VideoFileClip

def ytaudio(yt):
    rename_old = f"{yt.title}.mp4"

    invalid_chars = ['?', '/', '\\', ':', '*', '"', '<', '>', '|']
    for char in invalid_chars:
        rename_old = rename_old.replace(char, '')

    yt.streams.get_highest_resolution().download(filename=rename_old)
    video = VideoFileClip(rename_old)
    audio = video.audio
    audio.write_audiofile(f"{yt.title}.mp3")

def download():
    url = input("Ссылка на видео: ")
    yt = YouTube(url)

    if yt:
        print('Путь сохранения: ', f"{yt.streams.get_highest_resolution().download()}")
        ytaudio(yt)
        print('Видео загружено!')
    else:
        print('Видео не найдено')

if __name__ == '__main__':
    download()