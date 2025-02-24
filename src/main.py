from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from dependencies import img_service

app = FastAPI(title='Show Image Service')


@app.get('/image')
def show_image(image_service: img_service, category: List[str] = Query(None, max_length=10)):
    try:
        img_url = image_service.get_image(categories=category)
        return HTMLResponse(content=f'<img>{img_url}</img>', status_code=200)
    except Exception:
        raise HTTPException(detail='Unknown error', status_code=400)
