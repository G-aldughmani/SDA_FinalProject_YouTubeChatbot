import os
import time
import random
from pathlib import Path
from typing import Tuple
import yt_dlp


class YouTubeDownloader:
    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
            'Mozilla/5.0 (X11; Linux x86_64)',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X)'
        ]
        self.referers = [
            'https://www.google.com/',
            'https://www.facebook.com/',
            'https://www.reddit.com/',
            'https://www.twitter.com/'
        ]

    def get_ydl_options(self):
        return {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '5',
            }],
            'outtmpl': 'audio_downloads/%(title)s.%(ext)s',
            'quiet': False,
            'no_warnings': False,
            'retries': 10,
            'fragment_retries': 10,
            'extractor_retries': 3,
            'http_headers': {
                'User-Agent': random.choice(self.user_agents),
                'Referer': random.choice(self.referers),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
            },
            'socket_timeout': 30,
            'extract_flat': True,
            'force_generic_extractor': True,
            'noplaylist': True,
            'verbose': True
        }

    def download_with_ytdlp(self, video_url: str) -> Tuple[str, str]:
        """Primary download method with yt-dlp"""
        try:
            with yt_dlp.YoutubeDL(self.get_ydl_options()) as ydl:
                info = ydl.extract_info(video_url, download=True)
                file_path = ydl.prepare_filename(info).replace('.webm', '.wav').replace('.m4a', '.wav')
                return file_path, info.get('title', 'untitled')
        except Exception as e:
            raise Exception(f"yt-dlp failed: {str(e)}")

    def download_with_pytube(self, video_url: str) -> Tuple[str, str]:
        """Fallback method using pytube"""
        try:
            from pytube import YouTube
            yt = YouTube(video_url, use_oauth=True, allow_oauth_cache=True)
            stream = yt.streams.filter(only_audio=True).first()
            output_path = stream.download(output_path='audio_downloads')
            return output_path, yt.title
        except Exception as e:
            raise Exception(f"pytube failed: {str(e)}")


def download_youtube_audio(video_url: str) -> Tuple[str, str]:
    """Main download function with multiple fallbacks"""
    downloader = YouTubeDownloader()
    methods = [
        downloader.download_with_ytdlp,
        downloader.download_with_pytube
    ]
    for attempt in range(3):
        for method in methods:
            try:
                return method(video_url)
            except Exception as e:
                print(f"Attempt {attempt + 1} with {method.__name__} failed: {str(e)}")
                time.sleep(random.randint(2, 5))
    raise Exception("All download methods failed after 3 attempts each")
