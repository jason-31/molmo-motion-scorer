#!/usr/bin/env python3
"""
MolmoPoint-Vid-4B integration for detecting manipulated objects in robot videos.

Uses VIDEO input with grounding token output for precise video pointing.
Follows the official usage from https://huggingface.co/allenai/MolmoPoint-Vid-4B:
    processor.apply_chat_template(..., return_pointing_metadata=True)
    model.generate(..., logits_processor=model.build_logit_processor_from_inputs(inputs))
    model.extract_video_points(text, metadata) → [[object_id, image_num, x, y], ...]

Requires: transformers==4.57.1, decord2

Usage (standalone test):
    python molmo2_pointing.py --video_path /path/to/video.mp4 --instruction "Put the blue block in the green bowl"

Usage (as module):
    from molmo2_pointing import load_molmo2, detect_manipulated_objects
    model, processor = load_molmo2()
    points = detect_manipulated_objects(model, processor, video_path="/path/to/video.mp4",
                                        instruction="Put the blue block in the green bowl")
"""

import argparse
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers.video_utils import VideoMetadata

# Default fallback prompt (used only when no instruction is provided)
DEFAULT_PROMPT = "Point to all objects in contact and moved by robot arm"

MODEL_ID = "allenai/MolmoPoint-Vid-4B"

# Action verbs recognized from DROID language instructions, multi-word first
_ACTION_VERBS = [
    "pick up", "put down", "push down", "pull out", "pull up",
    "put", "place", "move", "fold", "pick", "push", "pull",
    "grab", "grasp", "lift", "drop", "slide", "rotate", "turn",
    "flip", "open", "close", "press", "squeeze", "stack",
    "insert", "remove", "take", "hand", "give", "bring",
]


def load_molmo2(
    model_id: str = MODEL_ID,
    device: Optional[str] = None,
    dtype=torch.bfloat16,
):
    """
    Load MolmoPoint-Vid-4B model and processor.

    Requires transformers==4.57.1.

    Returns (model, processor) tuple.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    if device is None:
        device_map = "auto"
    else:
        device_map = device

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.eval()

    return model, processor


def extract_object_and_action(instruction: str) -> tuple[str, str]:
    """
    Extract the target object and action verb from a DROID language instruction.

    Examples:
      "Put the blue block in the green bowl" -> ("blue block", "put")
      "Fold the bottom right tip of the duvet" -> ("bottom right tip of the duvet", "fold")
      "Pick up the tile letter" -> ("tile letter", "pick up")
      "Move the longer upright white container from the table to the bag"
        -> ("longer upright white container", "move")
    """
    instruction = instruction.strip()
    lower = instruction.lower()

    action = ""
    rest = instruction
    for verb in _ACTION_VERBS:
        if lower.startswith(verb):
            action = verb
            rest = instruction[len(verb):].strip()
            break

    if not action:
        # Fallback: first word is the action
        parts = instruction.split(None, 1)
        action = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""

    # Remove leading article
    rest = re.sub(r'^(the|a|an)\s+', '', rest, flags=re.IGNORECASE)

    # Extract until a preposition indicates destination/location
    # "around" included so "Put rubber band around X" → "rubber band"
    prep_pattern = r'\b(in|inside|into|on|onto|to|from|toward|towards|over|under|through|out of|off of|off|down|up|around)\b'
    match = re.search(prep_pattern, rest, re.IGNORECASE)
    if match:
        obj = rest[:match.start()].strip()
    else:
        obj = rest.strip()

    obj = obj.strip().rstrip('.,;:')

    # If object contains " with ", extract the instrument (part after "with").
    # E.g. "table with the sponge" → "sponge", "computer mouse with tea towel" → "tea towel"
    # But skip if "with" is a descriptor like "plate with blue rim" (no article/noun pattern needed
    # here — we just always take the instrument for manipulation verbs).
    with_match = re.search(r'\bwith\b\s+(?:the\s+|a\s+|an\s+)?(.+)', obj, re.IGNORECASE)
    if with_match:
        obj = with_match.group(1).strip().rstrip('.,;:')

    return obj, action


def build_molmo2_prompt(instruction: str) -> str:
    """Build an instruction-aware Molmo2 prompt from a DROID language instruction.

    Format: "Point to the <object> the robot is about to <action>"

    If the instruction cannot be parsed, falls back to DEFAULT_PROMPT.
    """
    obj, action = extract_object_and_action(instruction)
    if obj and action:
        return f"Point to the {obj} the robot is about to {action}"
    elif obj:
        return f"Point to the {obj}"
    else:
        return DEFAULT_PROMPT


def detect_manipulated_objects(
    model,
    processor,
    video_path: str,
    prompt: Optional[str] = None,
    instruction: Optional[str] = None,
) -> list[tuple[int, float, float]]:
    """
    Detect objects being manipulated in a video using MolmoPoint-Vid-4B.

    Follows https://huggingface.co/allenai/MolmoPoint-Vid-4B:
      - processor.apply_chat_template with return_pointing_metadata=True
      - model.build_logit_processor_from_inputs for constrained generation
      - model.extract_video_points to decode grounding tokens into coordinates

    Args:
        model: Loaded MolmoPoint-Vid-4B model
        processor: Loaded MolmoPoint-Vid-4B processor
        video_path: Path to the video file
        prompt: Explicit text prompt. If None, derived from instruction.
        instruction: Language instruction used to build prompt if prompt is not provided.

    Returns:
        List of (frame_idx, x_pixel, y_pixel) tuples. frame_idx is in the original video's
        frame index space.
    """
    # Build prompt
    if prompt is None:
        if instruction is not None:
            prompt = build_molmo2_prompt(instruction)
        else:
            prompt = DEFAULT_PROMPT

    print(f"  [Molmo2] Prompt: {prompt!r}")
    print(f"  [Molmo2] Video: {video_path}")

    # Get original video metadata for frame index mapping
    cap = cv2.VideoCapture(video_path)
    total_frames_cv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    image_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"  [Molmo2] Video: {total_frames_cv} frames @ {orig_fps:.1f}fps, {image_w}x{image_h}")

    # Build messages with video path
    messages = [
        {
            "role": "user",
            "content": [
                dict(type="text", text=prompt),
                dict(type="video", video=video_path),
            ],
        }
    ]

    # Reset decord bridge to native — SAM3 sets it to "torch" globally,
    # which makes vr.get_batch() return torch.Tensor (no .asnumpy()).
    # transformers' video_utils.py expects the native decord NDArray.
    try:
        import decord
        decord.bridge.set_bridge("native")
    except Exception:
        pass

    # apply_chat_template with return_pointing_metadata=True
    # Returns inputs dict containing both model inputs and pointing metadata
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
        return_pointing_metadata=True,
    )

    # Extract pointing metadata before moving tensors to GPU.
    # processor.apply_chat_template with return_pointing_metadata=True returns:
    #   - model inputs: input_ids, attention_mask, token_type_ids, pixel_values_videos,
    #                   video_token_pooling, video_grids
    #   - metadata dict: {token_pooling, subpatch_mapping, timestamps, video_size}
    metadata = inputs.pop("metadata")
    # Convert metadata tensors to numpy — extract_video_points uses np.argwhere
    # internally, which fails on PyTorch tensors ('Tensor' has no 'asnumpy')
    for key in ("token_pooling", "subpatch_mapping", "timestamps"):
        if key in metadata and isinstance(metadata[key], torch.Tensor):
            metadata[key] = metadata[key].cpu().numpy()

    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        generated_ids = model.generate(
            **inputs,
            logits_processor=model.build_logit_processor_from_inputs(inputs),
            max_new_tokens=512,
            do_sample=False,
        )

    generated_tokens = generated_ids[:, inputs['input_ids'].size(1):]
    generated_text = processor.post_process_image_text_to_text(
        generated_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0]

    print(f"  [Molmo2] Raw output: {generated_text[:200]!r}")

    # Decode grounding tokens into points using model method
    # Returns array of [object_id, image_num, x, y]
    points = model.extract_video_points(
        generated_text,
        metadata["token_pooling"],
        metadata["subpatch_mapping"],
        metadata["timestamps"],
        metadata["video_size"],
    )

    if points is None or len(points) == 0:
        print(f"  [Molmo2] No points detected")
        return []

    print(f"  [Molmo2] Raw points ({len(points)}):\n{points}")

    # Convert [object_id, image_num, x, y] → (frame_idx, x_pixel, y_pixel)
    # image_num is a (possibly fractional) index into the processor's sampled frames.
    # timestamps[i] gives the time in seconds of the i-th sampled frame.
    # We use timestamps to map back to the original video's frame index space.
    timestamps = metadata["timestamps"]
    results = []
    for pt in points:
        obj_id, image_num, x, y = pt[0], pt[1], pt[2], pt[3]
        # Map image_num → timestamp → original frame index
        img_idx = int(round(image_num))
        img_idx = max(0, min(img_idx, len(timestamps) - 1))
        time_sec = timestamps[img_idx]
        frame_idx = min(int(time_sec * orig_fps), total_frames_cv - 1)
        results.append((frame_idx, float(x), float(y)))

    # Deduplicate: keep unique (frame_idx, x, y) tuples
    seen = set()
    unique_results = []
    for f, x, y in results:
        key = (f, round(x, 1), round(y, 1))
        if key not in seen:
            seen.add(key)
            unique_results.append((f, x, y))
    results = unique_results

    print(f"  [Molmo2] Detected {len(results)} point(s): "
          f"{[(int(f), int(x), int(y)) for f, x, y in results]}")

    return results


def detect_manipulated_objects_multi_frame(
    model,
    processor,
    video_path: str,
    prompt: Optional[str] = None,
    instruction: Optional[str] = None,
    fps: float = 15.0,  # unused; fps extracted from video metadata internally
) -> list[tuple[int, float, float]]:
    """Thin wrapper around detect_manipulated_objects for API compatibility."""
    return detect_manipulated_objects(
        model, processor, video_path,
        prompt=prompt, instruction=instruction,
    )


# ─── Standalone test ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test Molmo2 object detection on a video")
    parser.add_argument("--video_path", required=True, help="Path to input video")
    parser.add_argument("--instruction", default=None, help="DROID language instruction (builds specific prompt)")
    parser.add_argument("--prompt", default=None, help="Explicit text prompt (overrides --instruction)")
    parser.add_argument("--fps", type=float, default=15.0, help="Video FPS (unused, extracted from video metadata)")
    parser.add_argument("--model_id", default=MODEL_ID, help="HuggingFace model ID")
    parser.add_argument("--debug_png", default=None, help="Save debug overlay PNG")
    args = parser.parse_args()

    # Load model
    print("Loading Molmo2 model...")
    model, processor = load_molmo2(args.model_id)
    print("Model loaded!")

    # Detect — pass video path directly (process_vision_info handles loading via decord)
    results = detect_manipulated_objects(
        model, processor, args.video_path,
        prompt=args.prompt, instruction=args.instruction,
    )

    if not results:
        print("No objects detected!")
        return 1

    print(f"\nDetected {len(results)} points:")
    for i, (fidx, x, y) in enumerate(results):
        print(f"  Point {i}: frame={fidx}, x={x:.1f}, y={y:.1f}")

    # Optional debug PNG — load frames via OpenCV only if needed for visualization
    if args.debug_png:
        import cv2
        cap = cv2.VideoCapture(args.video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        video_frames = np.array(frames)
        fidx = min(results[0][0], len(video_frames) - 1)
        frame = video_frames[fidx].copy()
        for _, x, y in results:
            cx, cy = int(x), int(y)
            cv2.line(frame, (cx - 15, cy), (cx + 15, cy), (255, 0, 0), 2)
            cv2.line(frame, (cx, cy - 15), (cx, cy + 15), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 10, (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
        cv2.imwrite(args.debug_png, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"Saved debug PNG: {args.debug_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# Qwen3-0.6B — object name extraction from captions
# ---------------------------------------------------------------------------

QWEN3_MODEL_ID = "Qwen/Qwen3-0.6B"


def load_qwen3():
    """Load Qwen3-0.6B for extracting object names from task captions.

    The model is already cached at:
      ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {QWEN3_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN3_MODEL_ID, dtype="auto", device_map="auto"
    )
    model.eval()
    print(f"  Qwen3-0.6B loaded on {next(model.parameters()).device}")
    return model, tokenizer


def extract_object_with_qwen3(model, 
                            tokenizer, 
                            caption: str, 
                            agent: str = "robot gripper") -> str:
    """Use Qwen3-0.6B to extract the object name from a task caption.

    agent: describes who is performing the action, e.g. "robot gripper" or "human hand".
    Returns the extracted object name string (thinking tags stripped).
    """
    # task_label = "Human task" if "human" in agent.lower() else "Robot task"
    # user_content = (
    #     f'{task_label}: "{caption}"\n'
    #     f'Extract the noun phrase naming the object(s) directly manipulated by the {agent}. '
    #     'Rules:\n'
    #     '- Return ONLY the noun phrase. No verbs, no articles (a/an/the), no extra words.\n'
    #     '- Use ONLY words from the task description. Do NOT invent new words.\n'
    #     '- Include adjectives and numerical words (two, three) from the text.\n'
    #     '- EXCLUDE destinations after "on/onto/into/in" and sources after "from".\n'
    #     '- EXCLUDE objects after "around" (e.g. "put elastic around bottle" → elastic).\n'
    #     '- EXCLUDE sub-parts referred to by "its/their" (e.g. "open laptop, wipe its screen" → laptop).\n'
    #     '- For pouring, return the container held (e.g. "pour water from jug" → jug).\n'
    #     '- For tools, return the tool (e.g. "wipe table with sponge" → sponge).\n'
    #     '- If multiple objects are ALL directly grasped, separate with " | ".\n'
    #     '- Do NOT convert verbs to adjectives (e.g. "stack lego blocks" → lego blocks).\n'
    #     'Examples:\n'
    #     '"pick up red plush toy from table" → red plush toy\n'
    #     '"put brown chess pieces on chess board" → brown chess pieces\n'
    #     '"stack lego blocks together" → lego blocks\n'
    #     '"put green plastic bead on red string" → green plastic bead\n'
    #     '"Place the container, strawberry toy and moose toy in the bag" → container | strawberry toy | moose toy\n'
    #     'Answer:'
    # )
    # excludes: grabbing tools, supporting surfaces, containers, pipes
    # includes: moving objcets, impact ojbects (objects being hit), smoke, liquid
    user_content = (
        f'Scene: "{caption}"\n'
        'This is a description of a physics scene. Name the thing(s) that PHYSICALLY MOVE, '
        'flow, or change shape during the event. Most nouns in the description are static '
        'scenery — ignore them. Name what moves, NOT what causes the motion.\n'
        'Rules:\n'
        '- Return ONLY the noun phrase. No verbs, no articles (a/an/the), no extra words.\n'
        '- Use ONLY words from the description. Do NOT invent new words.\n'
        '- Include adjectives and numerical words (two, three) from the text.\n'
        '- EXCLUDE static scenery: table, tabletop, surface, floor, wall, background, '
        'couch, shelf, ramp, whiteboard, mirror.\n'
        f'- EXCLUDE whatever holds or releases the moving object: {agent}, grabber tool, '
        'claw, gripper, cord, string, rope. Return what it lets go of.\n'
        '- EXCLUDE a handheld tool acting on something else (e.g. a knife slicing fruit '
        '→ the fruit, NOT the knife).\n'
        '- EXCLUDE the rotating platform / turntable / rotating base itself. '
        'Return whatever sits on it and moves with it.\n'
        '- EXCLUDE conduits that things come out of: pipe, tube, hose, nozzle, tap, '
        'chimney, dispenser. Return what comes out instead.\n'
        '- EXCLUDE a vessel ONLY when it holds liquid (glass, mug, bowl, dish, jar of '
        'liquid). Return the liquid, or the object entering it.\n'
        '- INCLUDE a box, basket, jar, crate, mug or cup when the vessel ITSELF is moved — '
        'lowered, released, dropped, crushed, lifted.\n'
        '- INCLUDE free-flowing substances: liquid, water, juice, smoke, steam, dust, sand.\n'
        '- INCLUDE objects that move because something else acts on them: struck, knocked '
        'over, pushed, dropped, poured on, crushed, torn, sliced.\n'
        '- For inflating or deforming, return the object that changes shape, NOT the air.\n'
        '- EXCLUDE shadows, reflections and mirror images — they are not objects.\n'
        '- If several separate things move, separate with " | ".\n'
        '- If nothing in the scene moves, return NONE.\n'
        'Examples:\n'
        '"A small white lampshade is on a light wood surface. A grey tennis ball rolls out '
        'of the black tube sitting on the table and rolls towards the right." '
        '→ grey tennis ball\n'
        '"A blue grabber tool holds a tennis ball above a pile of green kinetic sand on a '
        'wooden table. The grabber then releases the ball." → tennis ball\n'
        '"A clear acrylic box suspended from a cord hangs above a tennis ball positioned on '
        'a wooden table. The box is lowered to cover the ball." → clear acrylic box\n'
        '"A glass beverage dispenser filled with a bright red liquid is set up on a woven '
        'basket and is pouring the liquid into a clear glass on a wooden table." '
        '→ bright red liquid\n'
        '"Two rows of alternating black and white dominoes are set up on a wooden table. A '
        'wooden stick attached to a rotating platform rotates clockwise and knocks the first '
        'domino." → wooden stick | dominoes\n'
        '"A yellow mug is placed on a rotating turntable that rotates clockwise illuminated '
        'by a spotlight casting a shadow on the wall behind it." → yellow mug\n'
        '"Two pillows on a table and two grabber tools hanging above them from which a brown '
        'tennis ball and an orange block are suspended. The grabber tools let go of the ball '
        'and block." → brown tennis ball | orange block\n'
        '"A black balloon is attached to a fixed electric air pump hose on a table. Air is '
        'being pumped in the balloon." → black balloon\n'
        '"A stationary yellow rubber duck on a light brown wooden table against a plain '
        'white background." → NONE\n'
        'Answer:'
    )
    messages = [{"role": "user", "content": user_content}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(next(model.parameters()).device)
    output = None
    count = 0
    no_motion = False
    while not output:
        # Attempt 0 is greedy (reproducible). An empty result means the thinking chain
        # ate the whole 2048-token budget, and re-running greedily would reproduce it
        # verbatim — so each retry raises the temperature by 0.2 to break out of it.
        temperature = 0.2 * count
        sampling = (
            dict(do_sample=True, temperature=temperature) if temperature > 0
            else dict(do_sample=False)
        )
        with torch.no_grad():
            generated_ids = model.generate(**model_inputs, max_new_tokens=2048, **sampling)
        new_tokens = generated_ids[0][model_inputs.input_ids.shape[1]:]
        output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        # Strip closed thinking tags, then unclosed ones (if max_new_tokens cut it off)
        output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL).strip()
        output = re.sub(r'<think>.*', '', output, flags=re.DOTALL).strip()
        # Strip leading articles (a/an/the) and trailing punctuation
        output = re.sub(r'^(a|an|the)\s+', '', output, flags=re.IGNORECASE).strip()
        output = output.rstrip('.,;:')
        # "NONE" = the model decided nothing in the scene moves. That is a valid answer,
        # not a failed generation, so stop here instead of retrying. Empty string makes
        # grounding_worker skip the video (it can't ground an object that isn't there).
        if output.strip().upper() in ("NONE", "NULL", "N/A"):
            output = ""
            no_motion = True
            break
        count += 1
        if count > 5:
            break
    if no_motion:
        print(f"  [Qwen3] Caption: '{caption[:80]}' → NONE (no moving object)")
    else:
        print(f"  [Qwen3] Caption: '{caption[:80]}' → Object: '{output}'"
              + (f" (after {count} retries)" if count else ""))
    return output


# def extract_all_objects_with_qwen3(model, tokenizer, caption: str) -> str:
#     """Use Qwen3-0.6B to extract ALL objects (including destinations) from a task caption.

#     Unlike extract_object_with_qwen3() which only extracts moved objects,
#     this extracts every noun phrase with its adjectives.

#     Returns the extracted object names string (thinking tags stripped).
#     """
#     user_content = (
#         f'Sentence: "{caption}"\n'
#         'List ALL physical objects mentioned in this sentence. Include objects that are picked up, '
#         'placed on, placed into, wiped, poured from, or otherwise interacted with. '
#         'Use ONLY words from the sentence — do NOT add words not in the text. '
#         'Include adjectives and numerical words from the text. '
#         'Do NOT list any object more than once. Do NOT include verbs or actions. '
#         'Repeat shared nouns (e.g. "plum plush toy and banana plush toy", NOT "plum and banana"). '
#         'Separate with commas and "and" before the last item. '
#         'Examples: '
#         '"Grab the green blanket and put that on the table" → green blanket and table, '
#         '"Put the lid on the pot" → lid and pot, '
#         '"Pick up the red plush toy" → red plush toy, '
#         '"Move the plum and the banana plush toy from the oven tray to the counter then move the pineapple plush toy to the box" → plum plush toy, banana plush toy, pineapple plush toy, oven tray, counter and box, '
#         '"Put the elastic around the bottle" → elastic and bottle, '
#         '"Open the toaster oven" → toaster oven, '
#         '"Cover the computer mouse with the tea towel" → computer mouse and tea towel, '
#         '"Pick up the pen from the mug cup and put it on the table" → pen, mug cup and table, '
#         '"Put the fork into the mug" → fork and mug, '
#         '"Pick the two bottles on the table and put them in the bowl" → two bottles and bowl, '
#         '"Pour some contents of the jug into the bowl" → contents, jug and bowl, '
#         '"Wipe the table with the sponge" → table and sponge, '
#         '"Place the container, strawberry toy and moose toy in the bag" → container, strawberry toy, moose toy and bag, '
#         '"Move the colorless bottle" → colorless bottle. '
#         'Answer with the noun phrases only, no extra words.'
#     )
#     messages = [{"role": "user", "content": user_content}]
#     text = tokenizer.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
#     )
#     model_inputs = tokenizer([text], return_tensors="pt").to(next(model.parameters()).device)
#     with torch.no_grad():
#         generated_ids = model.generate(**model_inputs, max_new_tokens=2048, do_sample=False)
#     new_tokens = generated_ids[0][model_inputs.input_ids.shape[1]:]
#     output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
#     # Strip closed thinking tags, then unclosed ones (if max_new_tokens cut it off)
#     output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL).strip()
#     output = re.sub(r'<think>.*', '', output, flags=re.DOTALL).strip()
#     # Strip leading articles (a/an/the) and trailing punctuation
#     output = re.sub(r'^(a|an|the)\s+', '', output, flags=re.IGNORECASE).strip()
#     output = output.rstrip('.,;:')
#     # Strip trailing "and" artifact (e.g. "heat gun and counter, and" → "heat gun and counter")
#     output = re.sub(r',?\s+and\s*$', '', output).strip()
#     # Remove duplicate object phrases
#     parts = re.split(r',\s*', output)
#     seen = []
#     for p in parts:
#         p = p.strip()
#         if p and p.lower() not in [s.lower() for s in seen]:
#             seen.append(p)
#     output = ', '.join(seen)
#     # Fallback: if thinking consumed all tokens, use rule-based parser
#     if not output:
#         output, _ = extract_object_and_action(caption)
#         print(f"  [Qwen3-all] Caption: '{caption[:80]}' → Objects: '{output}' (fallback: rule-based)")
#     else:
#         print(f"  [Qwen3-all] Caption: '{caption[:80]}' → Objects: '{output}'")
#     return output


# ---------------------------------------------------------------------------
# Molmo2-8B — visual recaptioning for vague object descriptions
# ---------------------------------------------------------------------------

MOLMO2_8B_MODEL_ID = "allenai/Molmo2-8B"

# Generic nouns / pronouns that indicate the caption is too vague for useful Molmo2 pointing
_VAGUE_NOUNS = frozenset({
    # Generic nouns
    "object", "objects", "thing", "things", "item", "items",
    "content", "contents", "stuff", "material", "materials",
    "something", "anything",
    # Pronouns / demonstratives — appear in DROID instructions like "pick it up", "move this"
    "it", "this", "that", "one", "ones",
})


def is_vague_object(obj: str) -> bool:
    """Return True if any object phrase in the string is too generic for detection.

    Handles "|"-separated (per-object), comma-separated, and 'and'-separated lists
    by checking each part independently.

    Examples:
      "two objects"              → True
      "two objects, bowl, pot"   → True  (because "two objects" is vague)
      "contents, jug and bowl"   → True  (because "contents" is vague)
      "it"                       → True
      "this"                     → True
      "pick it up"               → True  (word "it" found)
      "fork"                     → False
      "red plush toy"            → False
      "fork and mug"             → False
      "fork | mug"               → False
    """
    parts = re.split(r'\s*\|\s*|,\s*|\s+and\s+', obj.lower())
    for part in parts:
        words = set(part.strip().split())
        if words & _VAGUE_NOUNS:
            return True
    return False


def load_molmo2_8b(device: str = "cuda"):
    """Load Molmo2-8B for visual recaptioning of vague object descriptions.

    Uses apply_chat_template API (image input, not video).
    Loads directly to `device` without device_map to avoid accelerate hook issues.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Loading {MOLMO2_8B_MODEL_ID}...")

    processor = AutoProcessor.from_pretrained(
        MOLMO2_8B_MODEL_ID,
        trust_remote_code=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MOLMO2_8B_MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = model.to(device)
    model.eval()
    print(f"  Molmo2-8B loaded on {device}")
    return model, processor


def recaption_object_with_molmo2_8b(
    model,
    processor,
    frame_rgb: np.ndarray,
    caption: str,
) -> str:
    """Use Molmo2-8B to rewrite a vague task caption with specific visual object names.

    The model looks at a video frame and rewrites the task description, replacing
    the vague object noun (e.g. "contents", "two objects") with the specific object(s)
    it sees in the frame (e.g. "white cylindrical marshmallows").

    Returns a full task sentence. Pass this to extract_object_with_qwen3() to get
    the final object noun phrase for Molmo2-VideoPoint prompting.

    Args:
        model: Loaded Molmo2-8B model
        processor: Loaded Molmo2-8B processor
        frame_rgb: (H, W, 3) uint8 RGB numpy array — representative video frame
        caption: The vague task caption

    Returns:
        Rewritten task description as a complete sentence with specific object names.
    """
    image = Image.fromarray(frame_rgb)
    prompt = (
        f'Original task: "{caption}"\n'
        'The object name in this task description is vague (e.g. "contents", "objects", "things"). '
        'Look at this robot camera frame to identify the specific object(s) present. '
        'Rewrite the task description replacing the vague object name with the specific '
        'visual description you see (include color, material, shape if visible). '
        'Keep the same action verb and sentence structure — only replace the vague noun. '
        'Output the rewritten task description as a complete sentence only, no extra words.'
    )
    messages = [
        {
            "role": "user",
            "content": [
                dict(type="image", image=image),
                dict(type="text", text=prompt),
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)

    generated_tokens = generated_ids[0, inputs["input_ids"].size(1):]
    result = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    result = result.rstrip('.,;:')
    print(f"  [Molmo2-8B] Recaptioned task: '{caption[:60]}' → '{result}'")
    return result


def recaption_object_with_molmo2_8b_video(
    model,
    processor,
    video_frames: np.ndarray,
    caption: str,
    fps: float = 15.0,
) -> str:
    """Use Molmo2-8B in VIDEO mode to rewrite a vague task caption with specific visual object names.

    Like recaption_object_with_molmo2_8b but feeds the full video instead of a single frame.
    The model observes the whole video to identify the specific object(s) being manipulated.

    Args:
        model: Loaded Molmo2-8B model
        processor: Loaded Molmo2-8B processor
        video_frames: (T, H, W, 3) uint8 RGB numpy array — full video
        caption: The vague task caption
        fps: Video frames per second

    Returns:
        Rewritten task description as a complete sentence with specific object names.
    """
    total_frames = len(video_frames)
    h, w = video_frames.shape[1], video_frames.shape[2]

    video_meta = VideoMetadata(
        total_num_frames=total_frames,
        fps=fps,
        width=w,
        height=h,
        duration=total_frames / fps,
    )

    prompt = (
        f'Original task: "{caption}"\n'
        'The object name in this task description is vague (e.g. "contents", "objects", "things"). '
        'Look at this robot camera video to identify the specific object(s) being manipulated. '
        'Rewrite the task description replacing the vague object name with the specific '
        'visual description you see (include color, material, shape if visible). '
        'Keep the same action verb and sentence structure — only replace the vague noun. '
        'Output the rewritten task description as a complete sentence only, no extra words.'
    )

    messages = [
        {
            "role": "user",
            "content": [
                dict(type="video", video="placeholder"),
                dict(type="text", text=prompt),
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        videos=[video_frames],
        text=text,
        return_tensors="pt",
        videos_kwargs={"video_metadata": video_meta, "return_metadata": True},
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)

    generated_tokens = generated_ids[0, inputs["input_ids"].size(1):]
    result = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    result = result.rstrip('.,;:')
    print(f"  [Molmo2-8B video] Recaptioned task: '{caption[:60]}' → '{result}'")
    return result
