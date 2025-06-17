import os
import re
import uuid
import json
import asyncio
import io
import tempfile
from typing import Any, Dict, List, Optional, Union

import httpx
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, ConfigDict
from PIL import Image

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
NOCODB_STYLES_TABLE  = os.getenv("NOCODB_STYLES_TABLE") # Added for styles table ID
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
    country: Optional[str] = None
    town: Optional[str] = None
    location_name: Optional[str] = None

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

async def download_and_convert_to_webp(url: str) -> bytes:
    """Downloads an image from a URL and converts it to WebP format in memory."""
    logging.info(f"Downloading and converting to WebP: {url}")
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        img_bytes = response.content

    try:
        img = Image.open(io.BytesIO(img_bytes))
        # Convert to RGB if not already, as WebP doesn't support all modes (e.g., RGBA for transparency)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        
        webp_buffer = io.BytesIO()
        img.save(webp_buffer, format="webp", quality=85)
        webp_bytes = webp_buffer.getvalue()
        logging.info(f"Successfully converted image from {url} to WebP ({len(webp_bytes)} bytes)")
        return webp_bytes
    except Exception as e:
        logging.error(f"Failed to convert image from {url} to WebP: {e}", exc_info=True)
        raise

async def cloudinary_upload(
    file_source: Union[str, bytes], 
    preset: Optional[str] = None, 
    transformations: Optional[Dict[str, Any]] = None,
    public_id: Optional[str] = None
) -> Optional[Dict[str, str]]:
    """
    Uploads a file to Cloudinary from a URL or bytes, with optional transformations.
    If file_source is bytes, it's saved to a temporary file before upload.
    """
    log_msg = f"Uploading to Cloudinary from source type: {type(file_source)}"
    if transformations:
        log_msg += f" with transformations: {transformations}"
    logging.info(log_msg)

    last_exc = None
    for attempt in range(1, 4):
        try:
            logging.info(f"Upload attempt {attempt}")
            options = {}
            if preset:
                options["upload_preset"] = preset
            if transformations:
                options["transformation"] = transformations
            if public_id:
                options["public_id"] = public_id

            if isinstance(file_source, bytes):
                # Save bytes to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".webp") as temp_file:
                    temp_file.write(file_source)
                    temp_file_path = temp_file.name
                
                logging.info(f"Uploading temporary file: {temp_file_path}")
                res = cld_upload(temp_file_path, **options)
                os.unlink(temp_file_path) # Clean up the temporary file
            else: # Assume it's a URL string
                logging.info(f"Uploading URL: {file_source}")
                res = cld_upload(file_source, **options)
            
            logging.info(f"Uploaded to Cloudinary: {res['secure_url']}")
            return {"secure_url": res["secure_url"], "public_id": res["public_id"]}
        except Exception as e:
            logging.warning(f"Upload attempt {attempt} failed: {e}")
            last_exc = e
            await asyncio.sleep(1)
    logging.error(f"Failed to upload source after 3 attempts", exc_info=True)
    return None

async def generate_painting(photo_url: str, location_name: str) -> str:
    async with _piapi_semaphore:
        logging.info("Acquired PiAPI semaphore for image generation")
        logging.info(f"Generating painting for photo URL: {photo_url}")
        
        prompt_text = (
            f"Paint this like Matisse. Your output must be recognisable as {location_name} "
            "but you must change the composition as much as necessary for it to feel like a Matisse masterpiece- "
            "feel free to remove people, cars and all text. Ignore clouds, render the sky a flat blue or grey. "
            "Output a 16:9 rectangle (either orientation)."
        )

        payload = {
            "model": "gpt-4o-image",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": photo_url}},
                        {"type": "image_url", "image_url": {"url": "https://i.postimg.cc/ZYGSGGdd/image.png"}},
                        {"type": "text", "text": prompt_text},
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

async def analyse_painting(png_url: str, desc: str, location_name: str, city: Optional[str] = None, country: Optional[str] = None) -> Dict[str, Any]:
    logging.info(f"Analysing painting at {png_url}")
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    location_context = ""
    if city and country:
        location_context = f" in {city}, {country}"
    elif city:
        location_context = f" in {city}"
    elif country:
        location_context = f" in {country}"

    prompt = (
        "Does this image look like a handmade painting? Respond ‘true’ or ‘false’.\n\n"
        "Is the composition attractive? Respond ‘true’ or ‘false’.\n\n"
        "Does it look like an oil or acrylic painting on canvas, a watercolor painting on paper, or other? "
        "Respond ‘oil/acrylic’, ‘watercolour’, or ‘other’.\n\n"
        "Is the format landscape (horizontal) or portrait (vertical)? Respond ‘horizontal’ or ‘vertical’.\n\n"
        "What is the main colour? Respond with a single colour.\n\n"
        "What are the secondary colours? Respond with a comma-separated list of colours.\n\n"
        "Now, imagine you’re an art/lifestyle blogger. You're cool, subtly poetic, not pretentious. "
        "Write a short description of the painting."
        f"The scene is {{desc}}{location_context}. Keep it inviting and artfully unforced. Write in German. Write as if both you and the reader are local to the area.\n\n"
        "Next, craft a simple, evocative title that highlights the location—think of a short travel-meets-art headline. "
        "Keep it genuine, lightly poetic, again in the language spoken in German.\n\n"
        "Respond in JSON exactly like this:\n"
        "{\n"
        "  \"handmade_painting\": true,\n"
        "  \"attractive_composition\": true,\n"
        "  \"painting_style\": \"other\",\n"
        "  \"format\": \"horizontal\",\n"
        "  \"main_colour\": \"pink\",\n"
        "  \"secondary_colours\": \"blue,yellow\",\n"
        "  \"description\": \"…\",\n"
        "  \"title\": \"Sunset over Berlin\"\n"
        "}"
    )
    completion = await client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text",      "text": prompt},
                    {"type": "image_url", "image_url": {"url": png_url}}
                ]
            }
        ],
        temperature=1,
        top_p=1,
        response_format={"type": "json_object"},
    )
    logging.info(f"Analysis result: {completion.choices[0].message.content}")
    return json.loads(completion.choices[0].message.content)

async def create_artwork_record(meta: Dict[str, Any], cloud_url: str, uuid_str: str, photo_id: int, catch_id: Optional[int], loc_id: Optional[int], style_id: Optional[int] = None) -> int:
    logging.info(f"Creating artwork record for photo {photo_id} with UUID {uuid_str}")

    # Compute web-optimized version by injecting the transformation after "/upload/"
    webimage_url = cloud_url.replace("/upload/", "/upload/t_web_image/")
    # Remove file extension
    webimage_url = webimage_url.rsplit('.', 1)[0]

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
        "Catchment":          [catch_id] if catch_id is not None else [],
        "location":           [loc_id] if loc_id is not None else [],
    }
    if style_id is not None:
        body["styles"] = [style_id] # Use the field name for the 'style' link

    url = f"{NOCODB_BASE_URL}/api/v2/tables/{NOCODB_ARTWORKS_TABLE}/records"
    res = await httpx_json("POST", url, headers=HEADERS_NOCODB, json=body)
    logging.info(f"Created artwork record with ID: {res.json()['Id']}")
    return res.json()["Id"]

# ---------------------------------------------------------------------------
# Style Processing Functions ----------------------------------------------
# ---------------------------------------------------------------------------

async def get_ready_styles() -> List[Dict[str, Any]]:
    """Queries the styles table for rows where 'Ready' is 'yes'."""
    logging.info("Querying styles table for ready styles")
    if not NOCODB_STYLES_TABLE:
        logging.error("NOCODB_STYLES_TABLE environment variable is not set.")
        return []
    url = f"{NOCODB_BASE_URL}/api/v2/tables/{NOCODB_STYLES_TABLE}/records"
    params = {
        "where": "(Ready,eq,yes)",
        "limit": 0 # Get all records
    }
    try:
        response = await httpx_json("GET", url, headers=HEADERS_NOCODB, params=params)
        data = response.json()
        logging.info(f"Found {data.get('totalRows', 0)} ready styles")
        return data.get("list", [])
    except httpx.HTTPStatusError as exc:
        logging.error(f"Error querying styles table: {exc}", exc_info=True)
        return []

async def generate_painting_with_style(base_photo_url: str, style_prompt: str, location_name: str, style_image_url: Optional[str] = None) -> str:
    """Generates a painting using a base photo, a text prompt, and an optional style image."""
    async with _piapi_semaphore:
        logging.info("Acquired PiAPI semaphore for style image generation")
        
        # Replace {location_name} placeholder in the prompt
        final_prompt = style_prompt.format(location_name=location_name)

        log_message = f"Generating painting for base photo URL: {base_photo_url} with style prompt: {final_prompt}"
        if style_image_url:
            log_message += f" and style image: {style_image_url}"
        logging.info(log_message)

        content_blocks: List[Dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": base_photo_url}},
        ]
        if style_image_url:
            content_blocks.append({"type": "image_url", "image_url": {"url": style_image_url}})
        content_blocks.append({"type": "text", "text": final_prompt})

        payload = {
            "model": "gpt-4o-image",
            "messages": [
                {
                    "role": "user",
                    "content": content_blocks,
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

async def mark_style_not_ready(style_id: int) -> None:
    """Marks a style row in the styles table as not ready."""
    logging.info(f"Marking style row {style_id} as not ready")
    if not NOCODB_STYLES_TABLE:
        logging.error("NOCODB_STYLES_TABLE environment variable is not set. Cannot mark style as not ready.")
        # Optionally, raise an error or handle appropriately
        raise ValueError("NOCODB_STYLES_TABLE is not set, cannot proceed with marking style not ready.")
    url = f"{NOCODB_BASE_URL}/api/v2/tables/{NOCODB_STYLES_TABLE}/records" # Corrected endpoint
    body = [{"Id": style_id, "Ready": "no"}] # Corrected request body format
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            await httpx_json("PATCH", url, headers=HEADERS_NOCODB, json=body)
            logging.info(f"Style row {style_id} marked as not ready on attempt {attempt}")
            break # Success, exit retry loop
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404 and attempt < max_retries:
                logging.warning(f"Attempt {attempt} failed to mark style {style_id} as not ready (404 Not Found). Retrying in {attempt} second(s)...")
                await asyncio.sleep(attempt) # Exponential backoff (simple)
            else:
                logging.error(f"Failed to mark style row {style_id} as not ready after {attempt} attempts: {exc}", exc_info=True)
                raise # Re-raise the exception if it's not a 404 or max retries reached
        except Exception as e:
            logging.error(f"An unexpected error occurred on attempt {attempt} while marking style {style_id} as not ready: {e}", exc_info=True)
            if attempt < max_retries:
                logging.warning(f"Retrying in {attempt} second(s)...")
                await asyncio.sleep(attempt)
            else:
                raise # Re-raise the exception after max retries

async def process_styles_for_photo(cloudinary_photo_url: str, photo_id: int, catch_id: Optional[int], loc_id: Optional[int], location_name: str, city: Optional[str] = None, country: Optional[str] = None) -> None:
    """Processes ready styles from the styles table for a given photo."""
    logging.info(f"Starting style processing for photo {photo_id}")
    ready_styles = await get_ready_styles()

    for style_row in ready_styles:
        try:
            style_id = style_row.get("Id")
            style_prompt = style_row.get("Prompt")
            style_image_url = style_row.get("image") # Assuming 'image' is the column name

            if not style_id or not style_prompt:
                logging.warning(f"Skipping style row with missing ID or Prompt: {style_row}")
                continue

            logging.info(f"Processing style {style_id} for photo {photo_id}")
            logging.info(f"Using style_id {style_id} for artwork record creation.")

            # Generate painting using cloudinary_photo_url, style_prompt, and style_image_url (if available)
            painted_png = await generate_painting_with_style(cloudinary_photo_url, style_prompt, location_name, style_image_url)

            # Download and convert the generated PNG to WebP locally
            webp_bytes = await download_and_convert_to_webp(painted_png)

            # Generate a consistent public_id for the style-generated artwork
            # This helps in overwriting existing images if the process is re-run
            art_uuid = str(uuid.uuid4()) # Generate a UUID for the artwork
            # Use a combination of photo_id, style_id, and a unique identifier for the public_id
            # This ensures uniqueness and allows for potential overwriting if the same style is re-processed for the same photo
            public_id_for_style_artwork = f"artwork_photo_{photo_id}_style_{style_id}_{art_uuid}"

            # Upload the locally converted WebP bytes to Cloudinary, using the generated public_id
            final_cld = await cloudinary_upload(
                webp_bytes, 
                preset=CLOUD_PRESET,
                public_id=public_id_for_style_artwork
            )
            if final_cld is None:
                logging.error(f"Cloudinary upload of style painting failed for style {style_id}; skipping.")
                continue

            # Analyse painting using the original PNG URL, as OpenAI API might not support AVIF directly
            style_title = style_row.get("Title", "Artwork")
            meta = await analyse_painting(painted_png, f"Artwork in {style_title} style", style_title, city, country)

            # Create artwork record, linking to the original photo details from the webhook and the style
            # Use the same art_uuid generated earlier for the public_id
            try:
                artwork_id = await create_artwork_record(meta, final_cld["secure_url"], art_uuid, photo_id, catch_id, loc_id, style_id)
                logging.info(f"Successfully created artwork record {artwork_id} for style {style_id}, photo {photo_id}")
            except Exception as create_ex:
                logging.error(f"Error creating artwork record for style {style_id}, photo {photo_id}: {create_ex}", exc_info=True)
                # Continue to the next style row if artwork record creation fails
                continue


            # Add a small delay to allow NocoDB to process the artwork linking
            await asyncio.sleep(2)

            logging.info(f"Successfully processed style {style_id} for photo {photo_id}")

        except Exception as ex:
            logging.error(f"Error processing style row {style_row.get('Id')}: {ex}", exc_info=True)
            # Continue to the next style row even if one fails

    logging.info(f"Finished style processing for photo {photo_id}")


# ---------------------------------------------------------------------------
# Main pipeline ------------------------------------------------------------
# ---------------------------------------------------------------------------

async def pipeline(photo: PhotoRow) -> None:
    logging.info(f"Starting pipeline for PhotoRow ID={photo.Id}")
    try:
        # Immediately mark photo as not ready to prevent duplicate webhooks
        await mark_photo_not_ready(photo.Id)
        # Use the source URL directly for painting generation
        painted_png_url = await generate_painting(photo.url, photo.location_name or "the location")
        
        # Download and convert the generated PNG to WebP locally
        webp_bytes = await download_and_convert_to_webp(painted_png_url)
        
        # Upload the locally converted WebP bytes to Cloudinary
        final_cld = await cloudinary_upload(
            webp_bytes,
            preset=CLOUD_PRESET
        )
        if final_cld is None:
            logging.error(f"Cloudinary upload of AVIF painting failed for photo {photo.Id}; marking as failed and skipping.")
            return
        
        # Analyse painting using the original PNG URL, as OpenAI API might not support AVIF directly
        meta = await analyse_painting(
            painted_png_url,
            photo.description or "",
            photo.description or "Berlin", # location_name
            photo.town, # city
            photo.country # country
        )
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

        # No need to delete source image from Cloudinary as it's not uploaded there
    except Exception as ex:
        logging.error(f"Error in pipeline for photo {photo.Id}: {ex}", exc_info=True)
        # No retry logic here, the exception is effectively handled by logging

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

    # Determine Catchment ID (object or _id) and Location ID (object or _id) from the webhook payload
    # Need to parse the row to access attributes like Catchments_id or Catchments
    try:
        # Manually add 'location_name' to the row dict before parsing
        if 'location name' in row:
            row['location_name'] = row['location name']

        photo_row_obj = PhotoRow.parse_obj(row)
        catch_id = None
        if hasattr(photo_row_obj, "Catchments_id"):
            catch_id = getattr(photo_row_obj, "Catchments_id")
        elif hasattr(photo_row_obj, "Catchments") and isinstance(photo_row_obj.Catchments, dict):
            catch_id = photo_row_obj.Catchments.get("Id") or photo_row_obj.Catchments.get("id")

        loc_id = None
        if hasattr(photo_row_obj, "locations_id"):
            loc_id = getattr(photo_row_obj, "locations_id")
        elif hasattr(photo_row_obj, "locations") and isinstance(photo_row_obj.locations, dict):
            loc_id = photo_row_obj.locations.get("Id") or photo_row_obj.locations.get("id")

        photo_id = photo_row_obj.Id
        photo_url = photo_row_obj.url
        location_name = photo_row_obj.location_name or "the location"

    except Exception as e:
        logging.error(f"Error parsing webhook payload or extracting photo details: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Invalid webhook payload structure")


    # Trigger original pipeline only on transition false→true
    if curr_ready and not prev_ready:
        # logging.info(f"Triggering original pipeline for photo {photo_id} due to false→true transition")
        # # The original pipeline function expects a PhotoRow object
        # background_tasks.add_task(pipeline, photo_row_obj)
        pass # Old pipeline commented out
    else:
         logging.info(f"Skip original pipeline for photo {photo_id} – ready_to_paint transition not false→true")


    # Always trigger style processing, regardless of the incoming photo's ready status transition
    # Pass the photo details from the webhook to the style processing task
    if photo_id and photo_url:
        logging.info(f"Adding style processing task for photo {photo_id} with source URL: {photo_url}")
        background_tasks.add_task(process_styles_for_photo, photo_url, photo_id, catch_id, loc_id, location_name, photo_row_obj.town, photo_row_obj.country)
    else:
         logging.warning("Skipping style processing task due to missing photo ID or URL in webhook payload")


    return {"status": "queued"} # Status is queued as tasks are added to background
