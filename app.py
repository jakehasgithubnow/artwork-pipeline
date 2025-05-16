import os
import re
import uuid
import json
import asyncio
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, ConfigDict

import cloudinary
from cloudinary.uploader import upload as cld_upload

from dotenv import load_dotenv
# Load local env first, then fallback to .env
load_dotenv(dotenv_path=".env.local", override=True)
load_dotenv(dotenv_path=".env")

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Config & constants ------------------------------------------------------
# ---------------------------------------------------------------------------

NOCODB_BASE_URL      = os.getenv("NOCODB_BASE_URL", "").rstrip("/")
NOCODB_API_TOKEN     = os.getenv("NOCODB_API_TOKEN")
NOCODB_PHOTO_TABLE   = os.getenv("NOCODB_PHOTO_TABLE")
NOCODB_ARTWORKS_TABLE = os.getenv("NOCODB_ARTWORKS_TABLE")
NOCODB_CATCHMENTS_LINK = os.getenv("NOCODB_CATCHMENTS_LINK")
NOCODB_LOCATIONS_LINK  = os.getenv("NOCODB_LOCATIONS_LINK")
NOCODB_PHOTO_LINK      = os.getenv("NOCODB_PHOTO_LINK")

PIAPI_KEY           = os.getenv("PIAPI_KEY")
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")

cloudinary.config(
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key    = os.getenv("CLOUDINARY_API_KEY"),
    api_secret = os.getenv("CLOUDINARY_API_SECRET"),
    secure     = True
)
CLOUD_PRESET = os.getenv("CLOUDINARY_UPLOAD_PRESET", "artwork")

REGEX_STORAGE_URL = re.compile(r"https:\/\/(?:storage\.)?theapi\.app\/[^\s\"]+\.png")

HEADERS_NOCODB = {
    "xc-token": NOCODB_API_TOKEN or "",
    "Content-Type": "application/json"
}
HEADERS_PIAPI = {
    "Authorization": f"Bearer {PIAPI_KEY}",
    "Content-Type": "application/json"
}

# ---------------------------------------------------------------------------
# FastAPI setup -----------------------------------------------------------
# ---------------------------------------------------------------------------

api = FastAPI(
    title="Artwork Pipeline",
    docs_url="/",
    redoc_url=None
)

# ---------------------------------------------------------------------------
# Models for incoming webhook ---------------------------------------------
# ---------------------------------------------------------------------------

class PhotoRow(BaseModel):
    Id: int
    url: str
    description: Optional[str] = None

    # Allow arbitrary extra fields (e.g. Catchments_id, locations_id, etc.)
    model_config = ConfigDict(extra="allow")

class WebhookData(BaseModel):
    rows: List[PhotoRow]
    previous_rows: Optional[List[Any]] = None

class WebhookPayload(BaseModel):
    id: str
    type: str
    data: WebhookData

# ---------------------------------------------------------------------------
# Utility functions -------------------------------------------------------
# ---------------------------------------------------------------------------

async def httpx_json(method: str, url: str, **kwargs) -> httpx.Response:
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        response = await client.request(method, url, **kwargs)
        response.raise_for_status()
        return response

async def mark_photo_not_ready(photo_id: int) -> None:
    logging.info(f"Marking photo {photo_id} as not ready")
    url = f"{NOCODB_BASE_URL}/api/v2/tables/{NOCODB_PHOTO_TABLE}/records/{photo_id}"
    body = {"ready to paint": "false"}
    try:
        await httpx_json("PATCH", url, headers=HEADERS_NOCODB, json=body)
        logging.info(f"Photo {photo_id} marked as not ready in NocoDB")
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            logging.warning(f"Photo {photo_id} not found in NocoDB (404); skipping mark as not ready")
        else:
            raise

async def cloudinary_upload(url: str) -> str:
    logging.info(f"Uploading URL to Cloudinary: {url}")
    last_exc = None
    for attempt in range(1, 4):
        try:
            logging.info(f"Upload attempt {attempt} for URL: {url}")
            res = cld_upload(
                url,
                upload_preset=CLOUD_PRESET,
                unsigned=True
            )
            logging.info(f"Uploaded to Cloudinary: {res['secure_url']}")
            return res["secure_url"]
        except Exception as e:
            logging.warning(f"Upload attempt {attempt} failed: {e}")
            last_exc = e
            await asyncio.sleep(1)
    logging.error(f"Failed to upload URL after 3 attempts: {url}")
    raise last_exc

async def generate_painting(photo_url: str) -> str:
    logging.info(f"Generating painting for photo URL: {photo_url}")
    payload = {
        "model": "gpt-4o-image",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": photo_url}},
                    {"type": "image_url", "image_url": {"url": "https://i.postimg.cc/ZYGSGGdd/image.png"}},
                    {"type": "text", "text": (
                        "Paint this place in the style of David Bomberg. "
                        "Simplify the composition, flatten shapes, avoid clutter. "
                        "Use colours from the photo and the reference. "
                        "Ignore clouds; paint sky flat blue or grey. "
                        "Output a 1024×1536 or 1536×1024 image."
                    )},
                ],
            }
        ],
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", "https://api.piapi.ai/v1/chat/completions",
                                 headers=HEADERS_PIAPI, json=payload) as r:
            r.raise_for_status()
            full_text = ""
            async for chunk in r.aiter_text():
                full_text += chunk
    logging.info(f"Received streamed response from PiAPI, concatenated length: {len(full_text)}")
    m = REGEX_STORAGE_URL.search(full_text)
    if not m:
        raise RuntimeError("No PNG URL in PiAPI response")
    logging.info(f"PiAPI returned painting URL: {m.group(0)}")
    return m.group(0)

async def analyse_painting(png_url: str, desc: str, location_name: str) -> Dict[str, Any]:
    logging.info(f"Analysing painting at {png_url}")
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    prompt = (
        "Does this image look like a handmade painting? respond ‘true’ or ‘false’\n\n"
        "Is the composition attractive? respond ‘true’ or ‘false’\n\n"
        "Does it look like an oil/acrylic painting on canvas, a watercolour painting on paper, or other? "
        "respond ‘oil/acrylic’ or ‘watercolour’ or ‘other’\n\n"
        "Is the format landscape (horizontal) or portrait (vertical)? respond ‘horizontal’ or ‘vertical’\n\n"
        "What is the main colour? respond with a single color\n\n"
        "What are the secondary colours? respond with a comma separated list of colours\n\n"
        f"Write a short description of the piece. The scene is {desc}. "
        "Your description should feel contemporary and compelling. "
        f"It must be written in the language spoken in {location_name}.\n\n"
        "Write a simple title for the piece, highlighting the location. "
        f"It must be written in the language spoken in {location_name}.\n\n"
        "Respond in json format exactly like this:\n"
        "{\n"
        '  "handmade_painting": true,\n'
        '  "attractive_composition": true,\n'
        '  "painting_style": "other",\n'
        '  "format": "horizontal",\n'
        '  "main_colour": "pink",\n'
        '  "secondary_colours": "blue,yellow",\n'
        '  "description": "…",\n'
        '  "title": "Sunset over Berlin"\n'
        "}"
    )
    chat = [{"role":"user","content":prompt,"image_url":png_url,"image_detail":"auto","image_input_type":"url"}]
    completion = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=chat,
        temperature=1,
        top_p=1,
        response_format={"type":"json_object"},
    )
    logging.info(f"Analysis result: {completion.choices[0].message.content}")
    return json.loads(completion.choices[0].message.content)

async def create_artwork_record(meta: Dict[str, Any], cloud_url: str, uuid_str: str, photo_id: int) -> int:
    logging.info(f"Creating artwork record for photo {photo_id} with UUID {uuid_str}")
    body = {
        "uuid": uuid_str,
        "url": cloud_url,
        "title": meta["title"],
        "description": meta["description"],
        "main colour": meta["main_colour"],
        "other colours": meta["secondary_colours"],
        "painting style": meta["painting_style"],
        "format": meta["format"],
        "locations photos": [str(photo_id)]
    }
    url = f"{NOCODB_BASE_URL}/api/v2/tables/{NOCODB_ARTWORKS_TABLE}/records"
    res = await httpx_json("POST", url, headers=HEADERS_NOCODB, json=body)
    logging.info(f"Created artwork record with ID: {res.json()['Id']}")
    return res.json()["Id"]

async def link_artwork(link_path: str, foreign_id: int, artwork_id: int) -> None:
    logging.info(f"Linking artwork {artwork_id} to record {foreign_id} via {link_path}")
    url = f"{NOCODB_BASE_URL}/api/v2/tables/{link_path}/records/{foreign_id}"
    await httpx_json("POST", url, headers=HEADERS_NOCODB, json=[artwork_id])

# ---------------------------------------------------------------------------
# Main pipeline ------------------------------------------------------------
# ---------------------------------------------------------------------------

async def pipeline(photo: PhotoRow) -> None:
    logging.info(f"Starting pipeline for PhotoRow ID={photo.Id}")
    try:
        await mark_photo_not_ready(photo.Id)
        src_url = await cloudinary_upload(photo.url)
        painted_png = await generate_painting(src_url)
        final_cld = await cloudinary_upload(painted_png)
        meta = await analyse_painting(painted_png, photo.description or "", photo.description or "Berlin")
        art_uuid = str(uuid.uuid4())
        artwork_id = await create_artwork_record(meta, final_cld, art_uuid, photo.Id)
        catch_id = getattr(photo, "Catchments_id", None)
        loc_id   = getattr(photo, "locations_id", None)
        if catch_id:
            await link_artwork(NOCODB_CATCHMENTS_LINK, catch_id, artwork_id)
        if loc_id:
            await link_artwork(NOCODB_LOCATIONS_LINK, loc_id, artwork_id)
        await link_artwork(NOCODB_PHOTO_LINK, photo.Id, artwork_id)
    except Exception as ex:
        logging.error(f"Error in pipeline for photo {photo.Id}: {ex}", exc_info=True)
        for attempt in range(2):
            await asyncio.sleep(3)
            try:
                await pipeline(photo)
                return
            except Exception:
                logging.warning(f"Retrying pipeline for photo {photo.Id}, attempt {attempt+1}")
        raise

# ---------------------------------------------------------------------------
# Webhook endpoint ---------------------------------------------------------
# ---------------------------------------------------------------------------

@api.post("/webhook")
async def receive_webhook(payload: WebhookPayload, background_tasks: BackgroundTasks):
    if not payload.data.rows:
        raise HTTPException(status_code=400, detail="No rows in payload")
    row = payload.data.rows[0]
    background_tasks.add_task(pipeline, row)
    return {"status": "queued"}