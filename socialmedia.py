import os
import re
import sys
import json
import asyncio
import urllib.parse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# Safe Windows UTF-8 stdout/stderr reconfiguration to prevent charmap UnicodeEncodeErrors
if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Attempt dotenv import safely
try:
    from dotenv import load_dotenv, find_dotenv
except ImportError:
    load_dotenv = None
    find_dotenv = None

from openai import AsyncOpenAI, OpenAI
from youtube_transcript_api import YouTubeTranscriptApi

# Attempt agents import from openai-agents
try:
    from agents import (
        Agent,
        Runner,
        ItemHelpers,
        function_tool,
        set_default_openai_client,
        set_default_openai_api,
        set_tracing_disabled,
    )
    # Disable OpenAI platform tracing to prevent 401 unauthorized errors with Gemini keys
    try:
        set_tracing_disabled(True)
    except Exception:
        pass
    HAS_AGENTS = True
except ImportError:
    HAS_AGENTS = False
    Agent = None
    Runner = None
    ItemHelpers = None
    function_tool = lambda f: f
    set_default_openai_client = lambda c: None
    set_default_openai_api = lambda a: None

# -----------------------------------
# Load Environment Variables
# -----------------------------------
if load_dotenv:
    _current_dir = Path(__file__).parent.resolve()
    for _env_candidate in [
        _current_dir / ".env",
        _current_dir.parent / ".env",
        Path.cwd() / ".env",
    ]:
        if _env_candidate.is_file():
            load_dotenv(_env_candidate, override=False)
    if find_dotenv:
        _found = find_dotenv()
        if _found:
            load_dotenv(_found, override=False)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL = "gemini-3.8-flash"
FALLBACK_MODELS = [DEFAULT_MODEL, "gemini-flash-latest"]

gemini_async_client: Optional[AsyncOpenAI] = None
gemini_client: Optional[OpenAI] = None


def configure_clients(api_key: Optional[str] = None) -> bool:
    """
    Configures OpenAI / Gemini clients with the provided or environment API key.
    """
    global GEMINI_API_KEY, gemini_async_client, gemini_client
    key = (api_key or os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY or "").strip()
    if not key:
        return False
    GEMINI_API_KEY = key
    gemini_async_client = AsyncOpenAI(
        api_key=GEMINI_API_KEY,
        base_url=GEMINI_BASE_URL,
    )
    if HAS_AGENTS:
        try:
            set_default_openai_client(gemini_async_client)
            set_default_openai_api("chat_completions")
            set_tracing_disabled(True)
        except Exception:
            pass

    gemini_client = OpenAI(
        api_key=GEMINI_API_KEY,
        base_url=GEMINI_BASE_URL,
    )
    return True


# Auto-configure if GEMINI_API_KEY is found in environment
if GEMINI_API_KEY:
    configure_clients(GEMINI_API_KEY)


# -----------------------------------
# Data Structures
# -----------------------------------
@dataclass
class Post:
    platform: str
    content: str


@dataclass
class GenerationResult:
    final_output: str
    posts: List[Dict[str, str]] = field(default_factory=list)
    transcript: str = ""
    raw_output: str = ""

    def __str__(self) -> str:
        return self.final_output


# -----------------------------------
# YouTube Transcript Helpers
# -----------------------------------
def extract_video_id(url_or_id: str) -> str:
    """
    Extract YouTube video ID from various URL formats or raw ID string.
    Supports:
      - Raw ID: 'dQw4w9WgXcQ'
      - Standard: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
      - Short: 'https://youtu.be/dQw4w9WgXcQ'
      - Shorts: 'https://www.youtube.com/shorts/dQw4w9WgXcQ'
      - Embed: 'https://www.youtube.com/embed/dQw4w9WgXcQ'
      - Live: 'https://www.youtube.com/live/dQw4w9WgXcQ'
      - Mobile: 'https://m.youtube.com/watch?v=dQw4w9WgXcQ'
    """
    if not url_or_id:
        return ""
    text = url_or_id.strip()

    # Standard watch URL or mobile watch URL
    if "youtube.com/watch" in text or "m.youtube.com/watch" in text:
        try:
            parsed = urllib.parse.urlparse(text)
            params = urllib.parse.parse_qs(parsed.query)
            if "v" in params and params["v"]:
                return params["v"][0]
        except Exception:
            pass

    # youtu.be / shorts / embed / live URLs
    patterns = [
        r"youtu\.be\/([a-zA-Z0-9_-]{11})",
        r"youtube\.com\/shorts\/([a-zA-Z0-9_-]{11})",
        r"youtube\.com\/embed\/([a-zA-Z0-9_-]{11})",
        r"youtube\.com\/live\/([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    # Check if raw string is an 11-char YouTube ID
    clean_id = text.split("?")[0].split("&")[0].split("/")[0].strip()
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", clean_id):
        return clean_id

    # Fallback heuristic
    if "/" in text:
        clean_id = text.split("/")[-1].split("?")[0].split("&")[0].strip()
        if re.fullmatch(r"[a-zA-Z0-9_-]{11}", clean_id):
            return clean_id

    return text


def get_transcript(video_id_or_url: str) -> str:
    """
    Fetch transcript from YouTube video with multiple fallbacks.
    Compatible with youtube_transcript_api 1.2+ as well as earlier versions.
    """
    video_id = extract_video_id(video_id_or_url)
    if not video_id:
        raise ValueError("Please provide a valid YouTube video ID or URL.")

    transcript_list = None
    ytt = YouTubeTranscriptApi()

    # Strategy 1: Direct fetch via modern instance API
    try:
        if hasattr(ytt, "fetch"):
            transcript_list = ytt.fetch(video_id)
        elif hasattr(YouTubeTranscriptApi, "get_transcript"):
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e_direct:
        # Strategy 2: Listing transcripts to find manual, auto, or translated
        try:
            if hasattr(ytt, "list"):
                transcript_manifest = ytt.list(video_id)
            elif hasattr(YouTubeTranscriptApi, "list_transcripts"):
                transcript_manifest = YouTubeTranscriptApi.list_transcripts(video_id)
            else:
                raise e_direct

            chosen_transcript = None
            try:
                chosen_transcript = transcript_manifest.find_transcript(["en", "en-US", "en-GB"])
            except Exception:
                pass

            if not chosen_transcript:
                try:
                    chosen_transcript = transcript_manifest.find_generated_transcript(["en", "en-US", "en-GB"])
                except Exception:
                    pass

            if not chosen_transcript:
                for t in transcript_manifest:
                    chosen_transcript = t
                    break

            if chosen_transcript:
                if chosen_transcript.language_code not in ["en", "en-US", "en-GB"] and chosen_transcript.is_translatable:
                    try:
                        chosen_transcript = chosen_transcript.translate("en")
                    except Exception:
                        pass
                transcript_list = chosen_transcript.fetch()
            else:
                raise Exception("No transcripts found for this video.")
        except Exception as e_list:
            raise Exception(
                f"Could not retrieve transcript for video '{video_id}'. "
                f"Please verify the video has closed captions / subtitles enabled. Details: {e_list}"
            )

    if not transcript_list:
        raise Exception(f"No transcript content could be retrieved for video '{video_id}'.")

    # Extract text from snippet items
    text_chunks = []
    for item in transcript_list:
        chunk = ""
        if hasattr(item, "text"):
            chunk = item.text
        elif isinstance(item, dict) and "text" in item:
            chunk = item["text"]
        elif isinstance(item, str):
            chunk = item
        
        # Strip bracket annotations like [♪♪♪] or [Music]
        cleaned_chunk = re.sub(r"\[.*?\]", "", chunk).strip()
        if cleaned_chunk:
            text_chunks.append(cleaned_chunk)

    full_transcript = " ".join(text_chunks).strip()
    if not full_transcript:
        raise Exception("The fetched transcript is empty. The video may contain only non-verbal audio.")

    return full_transcript


# -----------------------------------
# Content Generation Tool Function
# -----------------------------------
@function_tool
def generate_platform_content(
    video_transcript: str,
    social_media_platform: str,
) -> str:
    """
    Tool function to generate platform-specific social media post from video transcript.
    """
    global gemini_client
    if not gemini_client:
        if not configure_clients():
            raise ValueError("GEMINI_API_KEY is not configured.")

    response = gemini_client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {
                "role": "user",
                "content": f"""
Create an engaging {social_media_platform} post from this transcript.

Transcript:
{video_transcript[:12000]}

Requirements:
- Platform specific style and tone
- Engaging hook
- Easy to read with clean formatting
- Relevant hashtags and emojis
"""
            }
        ],
        max_tokens=1200,
    )
    return response.choices[0].message.content or ""


# -----------------------------------
# Agent Definition
# -----------------------------------
content_writer_agent = None
if HAS_AGENTS and Agent is not None:
    content_writer_agent = Agent(
        name="Content Writer Agent",
        instructions="""
You are a world-class social media copywriter and content strategist.
Given a YouTube video transcript and target platforms (such as LinkedIn, Instagram, Twitter/X, Threads):
1. Analyze the core message, unique insights, quotes, and takeaways from the transcript.
2. Produce tailored, platform-native posts for each requested platform:
   - LinkedIn: Professional yet conversational, strong hook, bullet points, insightful value, call to action, 3-5 relevant hashtags.
   - Instagram: Visual caption style, engaging first sentence, relatable storytelling, readable spacing, clear CTA, 10-15 hashtags.
   - Twitter / X: High-engagement thread or punchy single tweet format, maximum value, crisp hooks, concise actionable takeaways.
   - Threads / Facebook: Community discussion style with an engaging closing question.
3. You MUST return your response strictly as a valid JSON object matching this schema:
{
  "response": [
    {
      "platform": "Platform Name (e.g. LinkedIn, Instagram, Twitter/X)",
      "content": "Full formatted post text with emojis and hashtags"
    }
  ]
}
Do not wrap your output in conversational filler outside the JSON.
""",
        model=DEFAULT_MODEL,
        tools=[generate_platform_content],
    )


# -----------------------------------
# Response Parser
# -----------------------------------
def parse_agent_response(output_str: str) -> List[Dict[str, str]]:
    """
    Safely parses JSON response from agent or LLM, stripping markdown code blocks.
    """
    text = (output_str or "").strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            if "response" in data and isinstance(data["response"], list):
                return data["response"]
            elif "posts" in data and isinstance(data["posts"], list):
                return data["posts"]
        elif isinstance(data, list):
            return data
    except Exception:
        pass

    json_match = re.search(r"\{[\s\S]*\"response\"[\s\S]*\}", text)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            if "response" in data and isinstance(data["response"], list):
                return data["response"]
        except Exception:
            pass

    posts = []
    platform_blocks = re.split(r"(?:^|\n)#{1,3}\s*([\w\s/]+)(?:\n|$)", text)
    if len(platform_blocks) > 1:
        for i in range(1, len(platform_blocks), 2):
            plat_name = platform_blocks[i].strip()
            content = platform_blocks[i+1].strip() if i+1 < len(platform_blocks) else ""
            if plat_name and content:
                posts.append({"platform": plat_name, "content": content})
    
    if not posts and text:
        posts.append({"platform": "Generated Post", "content": text})

    return posts


# -----------------------------------
# Primary High-Level Generation Function
# -----------------------------------
async def generate_content(
    video_id: str,
    query: str = "",
    platforms: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    tone: str = "Engaging & Viral",
    **kwargs
) -> GenerationResult:
    """
    High-level generator function.
    Accepts video ID or URL, fetches transcript, runs generation, and returns a GenerationResult.
    """
    if api_key:
        configure_clients(api_key)
    elif not GEMINI_API_KEY or not gemini_async_client or not gemini_client:
        if not configure_clients():
            raise ValueError("GEMINI_API_KEY is not set. Please provide it in the sidebar or in your .env file.")

    target_platforms = platforms or ["LinkedIn", "Instagram", "Twitter/X"]
    platforms_str = ", ".join(target_platforms)

    transcript = get_transcript(video_id)
    clipped_transcript = transcript[:25000]

    prompt = f"""
Target Social Media Platforms: {platforms_str}
Desired Tone / Style: {tone}
{f'Custom User Instructions: {query}' if query.strip() else ''}

YouTube Video Transcript:
{clipped_transcript}

Please generate platform-tailored posts for each requested platform ({platforms_str}).
Return strictly a JSON object with this format:
{{
  "response": [
    {{
      "platform": "Platform Name",
      "content": "Full formatted post content with hooks, insights, emojis, and hashtags"
    }}
  ]
}}
"""

    raw_output = ""
    if HAS_AGENTS and content_writer_agent and Runner is not None:
        try:
            result = await asyncio.wait_for(Runner.run(content_writer_agent, prompt), timeout=12.0)
            if hasattr(result, "final_output") and result.final_output:
                raw_output = result.final_output
            elif hasattr(result, "new_items") and ItemHelpers:
                raw_output = ItemHelpers.text_message_outputs(result.new_items)
        except Exception:
            raw_output = ""

    if not raw_output:
        if not gemini_async_client:
            configure_clients()
        
        last_error = None
        for model_name in FALLBACK_MODELS:
            try:
                response = await gemini_async_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert social media strategist. "
                                "Given a video transcript, you write high-converting, engaging, platform-native posts. "
                                "Always return valid JSON with a 'response' array."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=3000,
                )
                raw_output = response.choices[0].message.content or ""
                if raw_output:
                    break
            except Exception as exc:
                last_error = exc
                continue

        if not raw_output and last_error:
            raise last_error

    posts = parse_agent_response(raw_output)

    formatted_chunks = []
    for post in posts:
        plat = post.get("platform", "Social Media")
        cnt = post.get("content", "").strip()
        formatted_chunks.append(f"### 📱 {plat}\n\n{cnt}\n")

    final_markdown = "\n---\n\n".join(formatted_chunks) if formatted_chunks else raw_output

    return GenerationResult(
        final_output=final_markdown,
        posts=posts,
        transcript=transcript,
        raw_output=raw_output,
    )


# -----------------------------------
# CLI Entrypoint for Testing
# -----------------------------------
async def _cli_test():
    test_id = "dQw4w9WgXcQ"
    print(f"Testing transcript fetching for: {test_id}...")
    try:
        t = get_transcript(test_id)
        print(f"Success! Transcript length: {len(t)} characters.")
    except Exception as e:
        print(f"Transcript error: {e}")

    if GEMINI_API_KEY:
        print("Testing content generation...")
        try:
            res = await generate_content(test_id, platforms=["Twitter/X"])
            print(f"Generation Success! Output length: {len(res.final_output)} chars.")
        except Exception as e:
            print(f"Generation error: {e}")


if __name__ == "__main__":
    asyncio.run(_cli_test())