from IPython.display import Image, Video
from typing import  Union

from ml_dl_experiments import settings

core_path = settings.SOURCE_PATH

def show_image_from_source(
        path: str = "ml_dl/CNN/cat.webp", 
        is_video: bool=False,
        width: int = 300,
        height: int = 300) -> Union[Image, Video]:
    if not is_video:
        return Image(
            data=core_path+path, 
            width=width, 
            height=height)
    else:
        return Video(
            data=core_path+path,
            width=width,
            height=height,
            embed=True,
            html_attributes="muted loop autoplay" )
