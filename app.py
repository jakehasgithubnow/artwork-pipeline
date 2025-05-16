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
from cloudinary.uploader import upload as cld_upload, destroy as cld_destroy

from dotenv import load_dotenv
# Load local env first, then fallback to .env
load_dotenv(dotenv_path=".env.local", override=True)
load_dotenv(dotenv_path=".env")

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Concurrency and rate limiting
_piapi_semaphore = asyncio.Semaphore(4)
_nocodb_lock = asyncio.Lock()
_nocodb_last_call: float = 0.0

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
    # Throttle NocoDB to max 1 request/sec
    if url.startswith(NOCODB_BASE_URL):
        global _nocodb_last_call
        async with _nocodb_lock:
            now = asyncio.get_event_loop().time()
            wait = 1.0 - (now - _nocodb_last_call)
            if wait > 0:
                logging.info(f"Throttling NocoDB call for {wait:.2f}s")
                await asyncio.sleep(wait)
            _nocodb_last_call = asyncio.get_event_loop().time()
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

async def cloudinary_upload(url: str, preset: Optional[str] = None) -> Optional[Dict[str, str]]:
    logging.info(f"Uploading URL to Cloudinary: {url}")
    last_exc = None
    for attempt in range(1, 4):
        try:
            logging.info(f"Upload attempt {attempt} for URL: {url}")
            if preset:
                res = cld_upload(url, upload_preset=preset)
            else:
                res = cld_upload(url)
            logging.info(f"Uploaded to Cloudinary: {res['secure_url']}")
            return {"secure_url": res["secure_url"], "public_id": res["public_id"]}
        except Exception as e:
            logging.warning(f"Upload attempt {attempt} failed: {e}")
            last_exc = e
            await asyncio.sleep(1)
    logging.error(f"Failed to upload URL after 3 attempts: {url}")
    return None

async def generate_painting(photo_url: str) -> str:
    async with _piapi_semaphore:
        logging.info("Acquired PiAPI semaphore for image generation")
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
                            "Paint this like Matisse. Your output must be recognisable as {location_name} but you must change the composition as much as necessary for it to feel like a Matisse masterpiece- feel free to remove people, cars and all text. Ignore clouds, render the sky a flat blue or grey. Output a 16:9 rectangle (either orientation)."
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
        f"Write a short description of the piece. The scene is {desc}.  "
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
    completion = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt,
                "image_url": png_url
            }
        ],
        temperature=1,
        top_p=1,
        response_format={"type":"json_object"},
    )
    logging.info(f"Analysis result: {completion.choices[0].message.content}")
    return json.loads(completion.choices[0].message.content)

async def create_artwork_record(meta: Dict[str, Any], cloud_url: str, uuid_str: str, photo_id: int, catch_id: int, loc_id: int) -> int:
    logging.info(f"Creating artwork record for photo {photo_id} with UUID {uuid_str}")

    # Compute web-optimized version by injecting the transformation after "/upload/"
    webimage_url = cloud_url.replace("/upload/", "/upload/t_web_image/")

    body = {
        "uuid": uuid_str,
        "url": cloud_url,
        "webimage": webimage_url,
        "title": meta["title"],
        "description": meta["description"],
        "main colour": meta["main_colour"],
        "other colours": meta["secondary_colours"],
        "painting style": meta["painting_style"],
        "format": meta["format"],
        "locations photos": [photo_id],
        "Catchment":          [catch_id],
        "location":           [loc_id],
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
        # Immediately mark photo as not ready to prevent duplicate webhooks
        await mark_photo_not_ready(photo.Id)
        src = await cloudinary_upload(photo.url, preset="basic_upload")
        if src is None:
            logging.error(f"Cloudinary upload failed for photo {photo.Id}; marking as failed and skipping.")
            return
        src_url = src["secure_url"]
        src_public_id = src["public_id"]
        painted_png = await generate_painting(src_url)
        final_cld = await cloudinary_upload(painted_png, preset=CLOUD_PRESET)
        if final_cld is None:
            logging.error(f"Cloudinary upload of painting failed for photo {photo.Id}; marking as failed and skipping.")
            return
        meta = await analyse_painting(painted_png, photo.description or "", photo.description or "Berlin")
        # Determine Catchment ID (object or _id)
        catch_id = None
        if hasattr(photo, "Catchments_id"):
            catch_id = getattr(photo, "Catchments_id")
        elif hasattr(photo, "Catchments") and isinstance(photo.Catchments, dict):
            catch_id = photo.Catchments.get("Id") or photo.Catchments.get("id")

        # Determine Location ID (object or _id)
        loc_id = None
        if hasattr(photo, "locations_id"):
            loc_id = getattr(photo, "locations_id")
        elif hasattr(photo, "locations") and isinstance(photo.locations, dict):
            loc_id = photo.locations.get("Id") or photo.locations.get("id")

        art_uuid = str(uuid.uuid4())
        artwork_id = await create_artwork_record(meta, final_cld["secure_url"], art_uuid, photo.Id, catch_id, loc_id)

        logging.info(f"Deleting source image from Cloudinary: {src_public_id}")
        cld_destroy(src_public_id)
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
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()
    rows = payload.get("data", {}).get("rows", [])
    prev_rows = payload.get("data", {}).get("previous_rows", [])
    if not rows:
        raise HTTPException(status_code=400, detail="No rows in payload")

    row = rows[0]
    curr_ready = str(row.get("ready to paint")).lower() == "true"
    prev_ready = False
    if prev_rows:
        prev_ready = str(prev_rows[0].get("ready to paint")).lower() == "true"

    # Trigger only on transition false→true
    if not curr_ready or prev_ready:
        logging.info(f"Skip photo {row.get('Id')} – ready_to_paint transition not false→true")
        return {"status": "skipped"}

    photo = PhotoRow.parse_obj(row)
    background_tasks.add_task(pipeline, photo)
    return {"status": "queued"}
