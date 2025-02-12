#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SpongeQuant

Inspired by text-generation-webui and the original Colab AutoQuant notebook,
this script provides a web-based UI for quantizing a model downloaded from Hugging Face.
It now features dynamic default parameters for each quantization method along with
the ability to select exactly which methods you want to run.
It also can compute an imatrix file (using llama-imatrix) from a provided calibration dataset,
so that users don’t have to manually run a separate tool.
"""

import os
import subprocess

def is_windows_host():
    """Detects if the container is running on a Windows host or WSL."""
    try:
        output = subprocess.check_output("grep -qEi '(microsoft|wsl)' /proc/version && echo 'WSL' || echo 'Linux'", shell=True).decode().strip()
        return output == "WSL" or sys.platform == "win32"  # Also check sys.platform for native Windows
    except Exception:
        return False  # Assume Linux if detection fails
    
# Set the environment variable if running on Windows or WSL (same as --no-mmap).
# On Windows or WSL (idk), the mmap module seems to have limitations when mapping files larger than 1 GB. This is due to the underlying implementation and the way Python handles large file mappings on Windows systems.
# if is_windows_host():
#     os.environ['LLAMA_ARG_NO_MMAP'] = '1'
#     print("[INFO] Windows or WSL detected, setting LLAMA_ARG_NO_MMAP environment variable.")


import random
import sys  # Added to use sys.executable for cross-platform Python calls

# Use expandable segments to help reduce fragmentation issues.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import yaml
import subprocess
import gradio as gr
import shutil
import threading
from huggingface_hub import snapshot_download, HfApi, ModelCard, create_repo

# ---------------------------
# Global Default Parameters
# ---------------------------
DEFAULT_IMATRIX_FILE = os.path.join("gguf", "imatrix.dat")
DEFAULT_CALIBRATION_FILE = os.path.join("gguf", "calibration_datav3.txt")

DEFAULT_PARAMS = {
    "GGUF": "IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M, IQ3_XXS, IQ3_S, IQ3_M, IQ3_XS, IQ4_XS, IQ4_NL, Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K",
    "GPTQ": "4, 128, 0.1",
    "ExLlamaV2": "4.5",
    "AWQ": "4, 128, GEMM, True",
    "HQQ": "2, 128"
}

# ---------------------------
# Helper: Patch config file on disk
# ---------------------------
def patch_model_config(model_dir):
    """
    Modify the config.json file in model_dir to override the rope_scaling field,
    leaving only the two required keys: 'type' and 'factor'. Also remove extraneous keys
    that may confuse AWQ (e.g., 'low_freq_factor', 'high_freq_factor', 'original_max_position_embeddings', 'rope_type').
    """
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"[WARN] No config.json found in {model_dir}")
        return
    try:
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        if "rope_scaling" in config_dict and isinstance(config_dict["rope_scaling"], dict):
            factor = config_dict["rope_scaling"].get("factor", 1.0)
            config_dict["rope_scaling"] = {"type": "linear", "factor": factor}
        for key in ["low_freq_factor", "high_freq_factor", "original_max_position_embeddings", "rope_type"]:
            if key in config_dict:
                del config_dict[key]
        with open(config_path, "w") as f:
            json.dump(config_dict, f)
        print(f"[INFO] Patched config in {config_path}")
    except Exception as e:
        print(f"[ERROR] Failed to patch config file: {e}")

# ---------------------------
# Helper: Run a shell command and stream output
# ---------------------------
def run_command(command: str):
    """Run a shell command and yield its output line by line."""
    yield f"[DEBUG] Executing command: {command}\n"
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    while True:
        line = process.stdout.readline()
        if line:
            yield line
        elif process.poll() is not None:
            break
    remaining = process.stdout.read()
    if remaining:
        yield remaining
    if process.returncode not in (None, 0):
        yield f"[ERROR] Command returned non-zero exit code: {process.returncode}\n"

# ---------------------------
# Helper: Compute imatrix file
# ---------------------------
def compute_imatrix_file(model_file: str, calibration_file: str, imatrix_output: str,
                         process_output: bool, verbosity: int, no_ppl: bool,
                         chunk: int, output_frequency: int, save_frequency: int,
                         in_files: list, ngl: int):
    """
    Compute the importance matrix file using the llama-imatrix tool.
    All optional parameters for imatrix are provided.
    """
    yield f"[INFO] Computing imatrix file for model {model_file} using calibration data from {calibration_file}\n"
    cmd_parts = [
        "llama-imatrix",
        "-m", model_file,
        "-f", calibration_file,
        "-o", imatrix_output,
        "--chunk", str(chunk),
        "-ngl", str(ngl),
        "--output-frequency", str(output_frequency),
        "--save-frequency", str(save_frequency),
        "--verbosity", str(verbosity)
    ]

    if process_output:
        cmd_parts.append("--process-output")
    if no_ppl:
        cmd_parts.append("--no-ppl")
    for in_file in in_files:
        cmd_parts.extend(["--in-file", in_file])
    cmd = build_llama_cmd(*cmd_parts)
    yield f"[INFO] Running imatrix command:\n  {cmd}\n"
    for line in run_command(cmd):
        yield line

# ---------------------------
# Utility functions
# ---------------------------
def is_model_fully_downloaded(model_id: str, target_dir: str, hf_token: str) -> bool:
    """
    Checks if all expected files for a Hugging Face model exist in the local directory.
    """
    try:
        api = HfApi()
        model_files = api.list_repo_files(repo_id=model_id, token=hf_token)
        for file in model_files:
            file_path = os.path.join(target_dir, file)
            if not os.path.exists(file_path):
                print(f"[DEBUG] Missing file: {file_path}")
                return False
        return True
    except Exception as e:
        print(f"[ERROR] Error checking model files: {e}")
        return False

def download_model(model_id: str, hf_token: str):
    """Download model from Hugging Face only if not already fully downloaded."""
    model_name = model_id.split("/")[-1]
    target_dir = os.path.join("models", model_name)
    
    yield "=== Downloading Model ===\n"
    yield f"[INFO] Model ID: {model_id}\n"
    yield f"[INFO] Target directory: {target_dir}\n"
    
    if os.path.exists(target_dir):
        if is_model_fully_downloaded(model_id, target_dir, hf_token):
            yield f"[INFO] Model {model_id} is already fully downloaded at {target_dir}. Skipping download.\n"
            return
        else:
            yield "[WARN] Model directory exists but files appear incomplete. Re-downloading...\n"
    else:
        yield "[INFO] Model directory does not exist. Starting download...\n"

    try:
        yield f"[INFO] Downloading model {model_id}...\n"
        model_path = snapshot_download(
            repo_id=model_id,
            token=hf_token,
            local_dir=target_dir,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx"],
            resume_download=True
        )
        patch_model_config(target_dir)
        yield f"[INFO] Model downloaded and patched at: {model_path}\n"
    except Exception as e:
        yield f"[ERROR] Error downloading model: {e}\n"

def generate_custom_model_card(model_id, base_model_name, quant_method, username, save_folder, license="mit", datasets=None):
    """
    Generate a custom model card with a randomly selected image and audio file.
    """
    formatted_qtype = format_quant_type(quant_method)
    custom_metadata = {
        "quantized_by": "SpongeQuant",
        "base_model": model_id,
        "language": ["en"],
        "license": license,
        "tags": ["SpongeQuant", formatted_qtype],
    }
    if datasets and isinstance(datasets, list) and any(datasets):
        custom_metadata["datasets"] = datasets

    # Define the list of 122 images (001.png to 122.png) with associated captions.
    images = [
        {"file": "001.png", "caption": "1. Calibration circle"},
        {"file": "002.png", "caption": "2. Solar location map"},
        {"file": "003.png", "caption": "3. Mathematical definitions"},
        {"file": "004.png", "caption": "4. Physical unit definitions"},
        {"file": "005.png", "caption": "5. Solar system parameters"},
        {"file": "006.png", "caption": "6. Solar system parameters"},
        {"file": "007.png", "caption": "7. The Sun"},
        {"file": "008.png", "caption": "8. Solar spectrum"},
        {"file": "009.png", "caption": "9. Mercury"},
        {"file": "010.png", "caption": "10. Mars"},
        {"file": "011.png", "caption": "11. Jupiter"},
        {"file": "012.png", "caption": "12. Earth"},
        {"file": "013.png", "caption": "13. Egypt, Red Sea, Sinai Peninsula and the Nile"},
        {"file": "014.png", "caption": "14. Chemical definitions"},
        {"file": "015.png", "caption": "15. DNA Structure"},
        {"file": "016.png", "caption": "16. DNA Structure magnified, light hit"},
        {"file": "017.png", "caption": "17. Cells and cell division"},
        {"file": "018.png", "caption": "18. Anatomy 1 (Skeleton front)"},
        {"file": "019.png", "caption": "19. Anatomy 2 (Internal organs front)"},
        {"file": "020.png", "caption": "20. Anatomy 3 (Skeleton and muscles back)"},
        {"file": "021.png", "caption": "21. Anatomy 4 (Internal organs back)"},
        {"file": "022.png", "caption": "22. Anatomy 5 (Ribcage)"},
        {"file": "023.png", "caption": "23. Anatomy 6 (Muscles front)"},
        {"file": "024.png", "caption": "24. Anatomy 7 (Heart, lungs, kidneys and main blood vessels back)"},
        {"file": "025.png", "caption": "25. Anatomy 8 (Heart, lungs, kidneys and main blood vessels front)"},
        {"file": "026.png", "caption": "26. Human sex organs"},
        {"file": "027.png", "caption": "27. Diagram of conception"},
        {"file": "028.png", "caption": "28. Conception"},
        {"file": "029.png", "caption": "29. Fertilized ovum"},
        {"file": "030.png", "caption": "30. Fetus diagram"},
        {"file": "031.png", "caption": "31. Fetus"},
        {"file": "032.png", "caption": "32. Diagram of male and female"},
        {"file": "033.png", "caption": "33. Birth"},
        {"file": "034.png", "caption": "34. Nursing mother"},
        {"file": "035.png", "caption": "35. Father and daughter (Malaysia)"},
        {"file": "036.png", "caption": "36. Group of children"},
        {"file": "037.png", "caption": "37. Diagram of family ages"},
        {"file": "038.png", "caption": "38. Family portrait"},
        {"file": "039.png", "caption": "39. Diagram of continental drift"},
        {"file": "040.png", "caption": "40. Structure of the Earth"},
        {"file": "041.png", "caption": "41. Heron Island (Great Barrier Reef of Australia)"},
        {"file": "042.png", "caption": "42. Seashore"},
        {"file": "043.png", "caption": "43. Snake River and Grand Tetons"},
        {"file": "044.png", "caption": "44. Sand dunes"},
        {"file": "045.png", "caption": "45. Monument Valley"},
        {"file": "046.png", "caption": "46. Forest scene with mushrooms"},
        {"file": "047.png", "caption": "47. Leaf"},
        {"file": "048.png", "caption": "48. Autumn Fallen leaves"},
        {"file": "049.png", "caption": "49. Snowflakes over Sequoia"},
        {"file": "050.png", "caption": "50. Tree with daffodils"},
        {"file": "051.png", "caption": "51. Flying insect with flowers"},
        {"file": "052.png", "caption": "52. Diagram of vertebrate evolution"},
        {"file": "053.png", "caption": "53. Seashell (Xancidae)"},
        {"file": "054.png", "caption": "54. Dolphins"},
        {"file": "055.png", "caption": "55. School of fish"},
        {"file": "056.png", "caption": "56. Tree toad"},
        {"file": "057.png", "caption": "57. Crocodile"},
        {"file": "058.png", "caption": "58. Eagle"},
        {"file": "059.png", "caption": "59. Waterhole"},
        {"file": "060.png", "caption": "60. Jane Goodall and chimps"},
        {"file": "061.png", "caption": "61. Sketch of bushmen"},
        {"file": "062.png", "caption": "62. Bushmen hunters"},
        {"file": "063.png", "caption": "63. Man from Guatemala"},
        {"file": "064.png", "caption": "64. Dancer from Bali"},
        {"file": "065.png", "caption": "65. Andean girls"},
        {"file": "066.png", "caption": "66. Thailand master craftsman"},
        {"file": "067.png", "caption": "67. Elephant"},
        {"file": "068.png", "caption": "68. Old man with beard and glasses (Turkey)"},
        {"file": "069.png", "caption": "69. Old man with dog and flowers"},
        {"file": "070.png", "caption": "70. Mountain climber"},
        {"file": "071.png", "caption": "71. Gymnast"},
        {"file": "072.png", "caption": "72. Sprinters (Valeriy Borzov of the U.S.S.R. in lead)"},
        {"file": "073.png", "caption": "73. Schoolroom"},
        {"file": "074.png", "caption": "74. Children with globe"},
        {"file": "075.png", "caption": "75. Cotton harvest"},
        {"file": "076.png", "caption": "76. Grape picker"},
        {"file": "077.png", "caption": "77. Supermarket"},
        {"file": "078.png", "caption": "78. Underwater scene with diver and fish"},
        {"file": "079.png", "caption": "79. Fishing boat with nets"},
        {"file": "080.png", "caption": "80. Cooking fish"},
        {"file": "081.png", "caption": "81. Chinese dinner party"},
        {"file": "082.png", "caption": "82. Demonstration of licking, eating and drinking"},
        {"file": "083.png", "caption": "83. Great Wall of China"},
        {"file": "084.png", "caption": "84. House construction (African)"},
        {"file": "085.png", "caption": "85. Construction scene (Amish country)"},
        {"file": "086.png", "caption": "86. House (Africa)"},
        {"file": "087.png", "caption": "87. House (New England)"},
        {"file": "088.png", "caption": "88. Modern house (Cloudcroft, New Mexico)"},
        {"file": "089.png", "caption": "89. House interior with artist and fire"},
        {"file": "090.png", "caption": "90. Taj Mahal"},
        {"file": "091.png", "caption": "91. English city (Oxford)"},
        {"file": "092.png", "caption": "92. Boston"},
        {"file": "093.png", "caption": "93. UN Building Day"},
        {"file": "094.png", "caption": "94. UN Building Night"},
        {"file": "095.png", "caption": "95. Sydney Opera House"},
        {"file": "096.png", "caption": "96. Artisan with drill"},
        {"file": "097.png", "caption": "97. Factory interior"},
        {"file": "098.png", "caption": "98. Museum"},
        {"file": "099.png", "caption": "99. X-ray of hand"},
        {"file": "100.png", "caption": "100. Woman with microscope"},
        {"file": "101.png", "caption": "101. Street scene, Asia (Pakistan)"},
        {"file": "102.png", "caption": "102. Rush hour traffic, India"},
        {"file": "103.png", "caption": "103. Modern highway (Ithaca, NY)"},
        {"file": "104.png", "caption": "104. Golden Gate Bridge"},
        {"file": "105.png", "caption": "105. Train"},
        {"file": "106.png", "caption": "106. Airplane in flight"},
        {"file": "107.png", "caption": "107. Airport (Toronto)"},
        {"file": "108.png", "caption": "108. Antarctic Expedition"},
        {"file": "109.png", "caption": "109. Radio telescope (Westerbork, Netherlands)"},
        {"file": "110.png", "caption": "110. Radio telescope (Arecibo)"},
        {"file": "111.png", "caption": "111. Page of book (Newton, System of the World)"},
        {"file": "112.png", "caption": "112. Astronaut in space"},
        {"file": "113.png", "caption": "113. Titan Centaur launch"},
        {"file": "114.png", "caption": "114. Sunset with birds"},
        {"file": "115.png", "caption": "115. String Quartet (Quartetto Italiano)"},
        {"file": "116.png", "caption": "116. Violin with music score (Cavatina)"},
        {"file": "117.png", "caption": "117. Statement 1/2"},
        {"file": "118.png", "caption": "118. Statement 2/2"},
        {"file": "119.png", "caption": "119. Credits 1/4"},
        {"file": "120.png", "caption": "120. Credits 2/4"},
        {"file": "121.png", "caption": "121. Credits 3/4"},
        {"file": "122.png", "caption": "122. Credits 4/4"}
    ]
    
    # Define the list of 31 audio files (001.mp3 to 031.mp3) with associated captions.
    audios = [
        {"file": "001.mp3", "caption": "1. Greetings from the Secretary-General of the UN – Kurt Waldheim"},
        {"file": "002.mp3", "caption": "2. Greetings In 55 Languages"},
        {"file": "003.mp3", "caption": "3. United Nations Greetings / Whale Songs"},
        {"file": "004.mp3", "caption": "4. The Sounds Of Earth"},
        {"file": "005.mp3", "caption": "5. Brandenburg Concerto No. 2 in F Major, BWV 1047: I. Allegro – Munich Bach Orchestra / Karl Richter (Johann Sebastian Bach)"},
        {"file": "006.mp3", "caption": "6. Ketawang: Puspåwårnå (Kinds of Flowers) – Pura Paku Alaman Palace Orchestra / K.R.T. Wasitodipuro"},
        {"file": "007.mp3", "caption": "7. Cengunmé – Mahi musicians"},
        {"file": "008.mp3", "caption": "8. Alima Song – Mbuti of the Ituri Rainforest"},
        {"file": "009.mp3", "caption": "9. Barnumbirr & Moikoi Song – Tom Djawa, Mudpo and Waliparu"},
        {"file": "010.mp3", "caption": "10. El Cascabel – Antonio Maciel and Los Aguilillas with Mariachi México de Pepe Villa / Rafael Carrión (Lorenzo Barcelata)"},
        {"file": "011.mp3", "caption": "11. Johnny B. Goode – Chuck Berry"},
        {"file": "012.mp3", "caption": "12. Mariuamangɨ – Pranis Pandang and Kumbui of the Nyaura Clan"},
        {"file": "013.mp3", "caption": "13. Sokaku-Reibo (Depicting the Cranes in Their Nest) – Goro Yamaguchi"},
        {"file": "014.mp3", "caption": "14. Partita for Violin Solo No. 3 in E Major, BWV 1006: III. Gavotte en Rondeau – Arthur Grumiaux (Johann Sebastian Bach)"},
        {"file": "015.mp3", "caption": "15. The Magic Flute (Die Zauberflöte), K. 620, Act II: Hell’s Vengeance Boils in My Heart – Bavarian State Opera Orchestra and Chorus / Wolfgang Sawallisch (Wolfgang Amadeus Mozart)"},
        {"file": "016.mp3", "caption": "16. Chakrulo – Georgian State Merited Ensemble of Folk Song and Dance / Anzor Kavsadze"},
        {"file": "017.mp3", "caption": "17. Roncadoras and Drums – Musicians from Ancash"},
        {"file": "018.mp3", "caption": "18. Melancholy Blues – Louis Armstrong and His Hot Seven (Marty Bloom / Walter Melrose)"},
        {"file": "019.mp3", "caption": "19. Muğam – Kamil Jalilov"},
        {"file": "020.mp3", "caption": "20. The Rite of Spring (Le Sacre du Printemps), Part II—The Sacrifice: VI. Sacrificial Dance (The Chosen One) – Columbia Symphony Orchestra / Igor Stravinsky"},
        {"file": "021.mp3", "caption": "21. The Well-Tempered Clavier, Book II: Prelude & Fugue No. 1 in C Major, BWV 870 – Glenn Gould (Johann Sebastian Bach)"},
        {"file": "022.mp3", "caption": "22. Symphony No. 5 in C Minor, Opus 67: I. Allegro Con Brio – Philharmonia Orchestra / Otto Klemperer (Ludwig Van Beethoven)"},
        {"file": "023.mp3", "caption": "23. Izlel e Delyu Haydutin – Valya Balkanska"},
        {"file": "024.mp3", "caption": "24. Navajo Night Chant, Yeibichai Dance – Ambrose Roan Horse, Chester Roan and Tom Roan"},
        {"file": "025.mp3", "caption": "25. The Fairie Round – Early Music Consort of London / David Munrow (Anthony Holborne)"},
        {"file": "026.mp3", "caption": "26. Naranaratana Kookokoo (The Cry of the Megapode Bird) – Maniasinimae and Taumaetarau Chieftain Tribe of Oloha and Palasu'u Village Community"},
        {"file": "027.mp3", "caption": "27. Wedding Song – Young girl of Huancavelica"},
        {"file": "028.mp3", "caption": "28. Liu Shui (Flowing Streams) – Guan Pinghu"},
        {"file": "029.mp3", "caption": "29. Bhairavi: Jaat Kahan Ho – Kesarbai Kerkar"},
        {"file": "030.mp3", "caption": "30. Dark Was the Night, Cold Was the Ground – Blind Willie Johnson"},
        {"file": "031.mp3", "caption": "31. String Quartet No. 13 in B-flat Major, Opus 130: V. Cavatina – Budapest String Quartet (Ludwig Van Beethoven)"},
    ]
    
    # Randomly select one image and one audio.
    selected_image = random.choice(images)
    selected_audio = random.choice(audios)
    
    # Build the custom content with the selected image and audio.
    custom_content = f"""
Quantized to `{formatted_qtype}` using [SpongeQuant](https://github.com/SpongeEngine/SpongeQuant), the Oobabooga of LLM quantization. Chat & support at [Sponge Engine](https://discord.gg/azNmr2Gdgy).

<figure>
  <img src="https://huggingface.co/spaces/SpongeEngine/README/resolve/main/{selected_image['file']}" alt="{selected_image['caption']}">
  <figcaption>{selected_image['caption']}</figcaption>
</figure>

<figure>
  <audio controls>
    <source src="https://huggingface.co/spaces/SpongeEngine/README/resolve/main/{selected_audio['file']}" type="audio/mp3">
    Your browser does not support the audio element.
  </audio>
  <figcaption>{selected_audio['caption']}</figcaption>
</figure>
"""
    merged_yaml = yaml.dump(custom_metadata, default_flow_style=False)
    full_card = f"---\n{merged_yaml}---\n\n{custom_content}"
    return full_card

def format_quant_type(qtype: str) -> str:
    """
    Formats the quantization method string for the repository name.
    """
    if qtype.lower().startswith("i1-"):
        parts = qtype.split("-", 1)
        if len(parts) == 2:
            return "i1-" + parts[1].upper()
        else:
            return qtype.lower()
    else:
        return qtype.upper()

def upload_quant(model_id, base_model_name, quantization_type, save_folder, hf_token, username, **kwargs):
    repo_id = f"{username}/{base_model_name}-{format_quant_type(quantization_type)}"
    log = f"[INFO] Preparing to upload quantized model to repo: {repo_id}\n"
    
    try:
        card_content = generate_custom_model_card(model_id, base_model_name, quantization_type, username, save_folder)
        card_path = os.path.join(save_folder, "README.md")
        with open(card_path, "w", encoding="utf-8") as f:
            f.write(card_content)
        log += f"[INFO] Created custom model card for {repo_id} at {card_path}\n"
    except Exception as e:
        log += f"[ERROR] Error creating custom model card: {e}\n"
    
    try:
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=hf_token)
        log += f"[INFO] Repo {repo_id} is ready. Uploading folder {save_folder}...\n"
        api = HfApi()
        api.upload_folder(
            folder_path=save_folder,
            repo_id=repo_id,
            ignore_patterns=["*.bf16.gguf"],
            token=hf_token
        )
        log += f"[INFO] Uploaded quantized model to {repo_id}\n"
    except Exception as e:
        log += f"[ERROR] Error uploading model: {e}\n"
    
    return log

def build_llama_cmd(script_name: str, *args):
    """
    Build command with the appropriate path (either Python script or compiled executable).
    """
    if script_name.endswith(".py"):
        script_path = os.path.join("llama_cpp", script_name)
        return f'"{sys.executable}" "{script_path}" ' + " ".join(str(arg) for arg in args)
    else:
        exec_path = os.path.join("llama_cpp", "build", "bin", script_name)
        return f'"{exec_path}"' + " ".join(str(arg) for arg in args)

# ---------------------------
# Quantization Method Implementations
# ---------------------------
def quantize_gguf(model_id: str, additional_param: str, hf_token: str, username: str,
                  use_imatrix: bool, imatrix_file: str, calibration_file: str, recompute_imatrix: bool,
                  imatrix_process_output: bool, imatrix_verbosity: int, imatrix_no_ppl: bool,
                  imatrix_chunk: int, imatrix_output_frequency: int, imatrix_save_frequency: int,
                  imatrix_in_files: str, imatrix_ngl: int):
    base_model_name = model_id.split("/")[-1].strip()
    model_dir = os.path.join("models", base_model_name)
    save_folder = os.path.join("quantized_models", f"{base_model_name}-GGUF")
    os.makedirs(save_folder, exist_ok=True)
    out_file = os.path.join(save_folder, f"{base_model_name.lower()}.bf16.gguf")
    yield f"=== GGUF Quantization for {base_model_name} ===\n"
    yield f"[INFO] Expected output file: {out_file}\n"
    
    if not os.path.exists(out_file):
        cmd = build_llama_cmd("convert_hf_to_gguf.py", model_dir, "--outtype", "bf16", "--outfile", out_file)
        yield f"[INFO] Running conversion command:\n  {cmd}\n"
        for line in run_command(cmd):
            yield line
    else:
        yield f"[INFO] File {out_file} already exists. Skipping conversion.\n"
    
    if use_imatrix:
        # Parse additional in-files (comma-separated) into a list.
        in_files_list = [s.strip() for s in imatrix_in_files.split(",") if s.strip()] if imatrix_in_files.strip() else []
        if recompute_imatrix or not os.path.exists(imatrix_file):
            for line in compute_imatrix_file(out_file, calibration_file, imatrix_file,
                                             process_output=imatrix_process_output,
                                             verbosity=imatrix_verbosity,
                                             no_ppl=imatrix_no_ppl,
                                             chunk=imatrix_chunk,
                                             output_frequency=imatrix_output_frequency,
                                             save_frequency=imatrix_save_frequency,
                                             in_files=in_files_list,
                                             ngl=imatrix_ngl):
                yield line

    quant_methods = additional_param.replace(" ", "").split(",")
    for method in quant_methods:
        method_str = method.strip().upper()
        if method_str.startswith("IQ") and not use_imatrix:
            yield f"[WARN] Skipping {method_str} quantization because imatrix is not enabled.\n"
            continue
        if use_imatrix:
            qtype = os.path.join(save_folder, f"{base_model_name.lower()}-i1-{method_str}.gguf")
            cmd = build_llama_cmd("llama-quantize", "--imatrix", imatrix_file, out_file, qtype, method_str)
        else:
            qtype = os.path.join(save_folder, f"{base_model_name.lower()}-{method_str}.gguf")
            cmd = build_llama_cmd("llama-quantize", out_file, qtype, method_str)
        yield f"[INFO] Quantizing with method '{method_str}':\n  {cmd}\n"
        for line in run_command(cmd):
            yield line

    repo_quant_type = "i1-GGUF" if use_imatrix else "GGUF"
    yield "[INFO] Uploading GGUF quantized model...\n"
    for line in upload_quant(model_id, base_model_name, repo_quant_type, save_folder, hf_token, username):
         yield line


def quantize_gptq(model_id: str, additional_param: str, hf_token: str, username: str):
    try:
        from transformers import AutoTokenizer, AutoConfig, GPTQConfig, AutoModelForCausalLM
    except ImportError:
        yield "[ERROR] GPU-specific quantization (GPTQ) requires the 'transformers' package. Please use a GPU container or install the dependency.\n"
        return

    # Now continue with the rest of your logic:
    yield "=== GPTQ Quantization ===\n"
    defaults = [4, 128, 0.1]
    if additional_param.strip():
        parts = [p.strip() for p in additional_param.split(",")]
        if len(parts) >= 3:
            bits = int(parts[0])
            group_size = int(parts[1])
            damp_percent = float(parts[2])
        else:
            bits, group_size, damp_percent = defaults
            yield f"[WARN] Insufficient GPTQ parameters provided. Using defaults: {defaults}\n"
    else:
        bits, group_size, damp_percent = defaults

    yield f"[INFO] Using GPTQ parameters: bits={bits}, group_size={group_size}, damp_percent={damp_percent}\n"

    base_model_name = model_id.split("/")[-1]
    local_dir = os.path.join("models", base_model_name)
    yield "[INFO] Loading tokenizer from local model directory...\n"
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    yield "[INFO] Loading patched configuration from local model directory...\n"
    patch_model_config(local_dir)
    config = AutoConfig.from_pretrained(local_dir, trust_remote_code=True)
    yield "[DEBUG] Patched config rope_scaling: " + str(config.rope_scaling) + "\n"
    yield "[INFO] Initializing GPTQ configuration...\n"
    quantization_config = GPTQConfig(
        bits=bits,
        dataset="c4",
        tokenizer=tokenizer,
        damp_percent=damp_percent,
        rope_scaling=config.rope_scaling
    )
    yield "[INFO] Loading model with integrated GPTQ configuration...\n"
    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        config=config,
        device_map="auto",
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    yield "[DEBUG] Loaded model config rope_scaling: " + str(model.config.rope_scaling) + "\n"
    model.config.rope_scaling = config.rope_scaling
    yield "[DEBUG] After override, model config rope_scaling: " + str(model.config.rope_scaling) + "\n"
    
    save_folder = os.path.join("quantized_models", f"{base_model_name}-GPTQ")
    yield f"[INFO] Saving quantized model to {save_folder}...\n"
    model.save_pretrained(save_folder, use_safetensors=True)
    tokenizer.save_pretrained(save_folder)
    yield "[INFO] GPTQ quantization completed.\n"
    for line in upload_quant(model_id, base_model_name, "GPTQ", save_folder, hf_token, username):
        yield line

def quantize_exllamav2(model_id: str, additional_param: str, hf_token: str, username: str):
    try:
        yield "=== ExLlamaV2 Quantization ===\n"
        bpw = float(additional_param) if additional_param.strip() else 4.5
        base_model_name = model_id.split("/")[-1]
        model_dir = os.path.join("models", base_model_name)
        save_folder = os.path.join("quantized_models", f"{base_model_name}-EXL2")
        cmd = f'"{sys.executable}" "/app/exllamav2/convert.py" -i {model_dir} -o {save_folder} -b {bpw}'
        yield f"[INFO] Running ExLlamaV2 command:\n  {cmd}\n"
        for line in run_command(cmd):
            yield line
        yield "[INFO] ExLlamaV2 quantization completed.\n"
        for line in upload_quant(model_id, base_model_name, "exl2", save_folder, hf_token, username, bpw=bpw):
            yield line
    except Exception as e:
        yield f"[ERROR] Error during ExLlamaV2 quantization: {e}\n"

def quantize_awq(model_id: str, additional_param: str, hf_token: str, username: str):
    try:
        yield "=== AWQ Quantization ===\n"
        defaults = [4, 128, "GEMM", True]
        if additional_param.strip():
            parts = [p.strip() for p in additional_param.split(",")]
            if len(parts) >= 4:
                bits = int(parts[0])
                group_size = int(parts[1])
                version = parts[2]
                zero_point = parts[3].lower() in ["true", "1", "yes"]
            else:
                bits, group_size, version, zero_point = defaults
                yield f"[WARN] Insufficient AWQ parameters provided. Using defaults: {defaults}\n"
        else:
            bits, group_size, version, zero_point = defaults

        yield f"[INFO] Using AWQ parameters: bits={bits}, group_size={group_size}, version={version}, zero_point={zero_point}\n"
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer, AutoConfig

        quant_config = {
            "w_bit": bits,
            "q_group_size": group_size,
            "version": version,
            "zero_point": zero_point
        }

        base_model_name = model_id.split("/")[-1]
        save_folder = os.path.join("quantized_models", f"{base_model_name}-AWQ")
        local_dir = os.path.join("models", base_model_name)
        yield "[INFO] Loading model and tokenizer for AWQ from local directory...\n"
        patch_model_config(local_dir)
        config = AutoConfig.from_pretrained(local_dir, trust_remote_code=True)
        if isinstance(config.rope_scaling, dict):
            config.rope_scaling = {"type": "linear", "factor": 1.0}
            yield f"[DEBUG] Overridden config rope_scaling: {config.rope_scaling}\n"
        model = AutoAWQForCausalLM.from_pretrained(local_dir, config=config, safetensors=True, low_cpu_mem_usage=True)
        tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
        
        yield "[INFO] Quantizing model using AWQ...\n"
        model.quantize(tokenizer, quant_config=quant_config)
        model.save_quantized(save_folder)
        tokenizer.save_pretrained(save_folder)
        yield f"[INFO] AWQ quantization completed. Saved to {save_folder}\n"
        for line in upload_quant(model_id, base_model_name, "AWQ", save_folder, hf_token, username):
            yield line
    except Exception as e:
        yield f"[ERROR] Error during AWQ quantization: {e}\n"

def quantize_hqq(model_id: str, additional_param: str, hf_token: str, username: str):
    try:
        yield "=== HQQ Quantization ===\n"
        defaults = [2, 128]
        if additional_param.strip():
            parts = [p.strip() for p in additional_param.split(",")]
            if len(parts) >= 2:
                bits = int(parts[0])
                group_size = int(parts[1])
            else:
                bits, group_size = defaults
                yield f"[WARN] Insufficient HQQ parameters provided. Using defaults: {defaults}\n"
        else:
            bits, group_size = defaults

        yield f"[INFO] Using HQQ parameters: bits={bits}, group_size={group_size}\n"
        from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer
        from hqq.core.quantize import BaseQuantizeConfig
        quant_config = BaseQuantizeConfig(nbits=bits, group_size=group_size)
        base_model_name = model_id.split("/")[-1]
        save_folder = os.path.join("quantized_models", f"{base_model_name}-HQQ")
        yield "[INFO] Downloading HQQ model and tokenizer...\n"
        model = HQQModelForCausalLM.from_pretrained(
            model_id, cache_dir=".", attn_implementation="flash_attention_2"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        yield "[INFO] Quantizing model using HQQ...\n"
        model.quantize_model(quant_config=quant_config, device='cuda')
        yield f"[INFO] HQQ quantization completed. Saving to {save_folder}...\n"
        model.save_quantized(save_folder)
        tokenizer.save_pretrained(save_folder)
        for line in upload_quant(model_id, base_model_name, "HQQ", save_folder, hf_token, username):
            yield line
    except Exception as e:
        yield f"[ERROR] Error during HQQ quantization: {e}\n"

# ---------------------------
# Main Orchestration Function
# ---------------------------
def quant_tavern_ui(model_ids: str, hf_token: str, username: str,
                    gguf_sel: bool, gguf_param: str,
                    gptq_sel: bool, gptq_param: str,
                    exllamav2_sel: bool, exllamav2_param: str,
                    awq_sel: bool, awq_param: str,
                    hqq_sel: bool, hqq_param: str,
                    enable_imatrix: bool, imatrix_file: str,
                    calibration_file: str,
                    recompute_imatrix: bool,
                    imatrix_process_output: bool, imatrix_verbosity: int, imatrix_no_ppl: bool,
                    imatrix_chunk: int, imatrix_output_frequency: int, imatrix_save_frequency: int,
                    imatrix_in_files: str, imatrix_ngl: int,
                    delete_original: bool, delete_quantized: bool):
    full_log = "=== Starting SpongeQuant Quantization Process ===\n"
    # Split the input into a list of model IDs (one per non-empty line)
    model_list = [m.strip() for m in model_ids.splitlines() if m.strip()]
    
    for model_id in model_list:
        full_log += f"\n=== Processing model: {model_id} ===\n"
        for line in download_model(model_id, hf_token):
            full_log += line
            yield full_log

        # Build a list of selected methods with their parameters.
        selected_methods = []
        if gguf_sel:
            selected_methods.append(("GGUF", gguf_param))
        if gptq_sel:
            selected_methods.append(("GPTQ", gptq_param))
        if exllamav2_sel:
            selected_methods.append(("ExLlamaV2", exllamav2_param))
        if awq_sel:
            selected_methods.append(("AWQ", awq_param))
        if hqq_sel:
            selected_methods.append(("HQQ", hqq_param))
        
        if not selected_methods:
            full_log += "[ERROR] No quantization method selected for model. Please select at least one method.\n"
            yield full_log
            continue

        for method, param in selected_methods:
            full_log += f"[INFO] Running {method} quantization...\n"
            yield full_log
            if method == "GGUF":
                # Pass the additional Imatrix parameters to quantize_gguf
                for line in quantize_gguf(model_id, gguf_param, hf_token, username,
                                          enable_imatrix, imatrix_file, calibration_file, recompute_imatrix,
                                          imatrix_process_output, imatrix_verbosity, imatrix_no_ppl,
                                          imatrix_chunk, imatrix_output_frequency, imatrix_save_frequency,
                                          imatrix_in_files, imatrix_ngl):
                    full_log += line
                    yield full_log
            elif method == "GPTQ":
                for line in quantize_gptq(model_id, param, hf_token, username):
                    full_log += line
                    yield full_log
            elif method == "ExLlamaV2":
                for line in quantize_exllamav2(model_id, param, hf_token, username):
                    full_log += line
                    yield full_log
            elif method == "AWQ":
                for line in quantize_awq(model_id, param, hf_token, username):
                    full_log += line
                    yield full_log
            elif method == "HQQ":
                for line in quantize_hqq(model_id, param, hf_token, username):
                    full_log += line
                    yield full_log
            else:
                full_log += f"[ERROR] Unknown quantization method: {method}\n"
                yield full_log

        # After processing methods for the current model, handle deletion if requested.
        base_model_name = model_id.split("/")[-1]
        if delete_original:
            original_path = os.path.join("models", base_model_name)
            if os.path.exists(original_path):
                try:
                    shutil.rmtree(original_path)
                    full_log += f"[INFO] Deleted original model folder: {original_path}\n"
                except Exception as e:
                    full_log += f"[ERROR] Could not delete original model folder {original_path}: {e}\n"
        if delete_quantized:
            quant_dir = os.path.join("quantized_models")
            if os.path.exists(quant_dir):
                for folder in os.listdir(quant_dir):
                    if folder.startswith(base_model_name + "-"):
                        full_path = os.path.join(quant_dir, folder)
                        try:
                            shutil.rmtree(full_path)
                            full_log += f"[INFO] Deleted quantized model folder: {full_path}\n"
                        except Exception as e:
                            full_log += f"[ERROR] Could not delete quantized folder {full_path}: {e}\n"
        yield full_log

    full_log += "\n=== Quantization Process Completed ===\n"
    yield full_log

# ---------------------------
# Build the UI using Gradio (with individual method selection)
# ---------------------------
with gr.Blocks(title="SpongeQuant") as iface:
    gr.Markdown("# SpongeQuant")
    with gr.Row():
        # Multi-line textbox for multiple model IDs (one per line)
        model_ids_input = gr.Textbox(label="Model IDs (one per line)", 
                                     value="mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated", 
                                     lines=3)
        hf_token_input = gr.Textbox(label="Hugging Face Token", type="password")
        username_input = gr.Textbox(label="Hugging Face Username", value="SpongeEngine")
    gr.Markdown("### Select Quantization Methods")
    with gr.Accordion("Quantization Methods", open=True):
        with gr.Row():
            gguf_checkbox = gr.Checkbox(label="GGUF", value=True)
            gguf_param = gr.Textbox(label="GGUF Additional Parameter", value=DEFAULT_PARAMS["GGUF"])
        with gr.Row():
            gptq_checkbox = gr.Checkbox(label="GPTQ", value=False)
            gptq_param = gr.Textbox(label="GPTQ Additional Parameter", value=DEFAULT_PARAMS["GPTQ"])
        with gr.Row():
            exllamav2_checkbox = gr.Checkbox(label="ExLlamaV2", value=False)
            exllamav2_param = gr.Textbox(label="ExLlamaV2 Additional Parameter", value=DEFAULT_PARAMS["ExLlamaV2"])
        with gr.Row():
            awq_checkbox = gr.Checkbox(label="AWQ", value=False)
            awq_param = gr.Textbox(label="AWQ Additional Parameter", value=DEFAULT_PARAMS["AWQ"])
        with gr.Row():
            hqq_checkbox = gr.Checkbox(label="HQQ", value=False)
            hqq_param = gr.Textbox(label="HQQ Additional Parameter", value=DEFAULT_PARAMS["HQQ"])
    gr.Markdown("### Imatrix Advanced Parameters")
    with gr.Row():
        imatrix_process_output_checkbox = gr.Checkbox(label="Process output.weight", value=False)
        imatrix_verbosity_input = gr.Number(label="Verbosity", value=1, precision=0)
        imatrix_no_ppl_checkbox = gr.Checkbox(label="Disable PPL", value=False)
    with gr.Row():
        imatrix_chunk_input = gr.Number(label="Chunk size", value=64, precision=0)
        imatrix_output_freq_input = gr.Number(label="Output Frequency", value=10, precision=0)
        imatrix_save_freq_input = gr.Number(label="Save Frequency", value=0, precision=0)
    with gr.Row():
        imatrix_in_files_input = gr.Textbox(label="Additional in-files (comma-separated)", value="")
        imatrix_ngl_input = gr.Number(label="GPU offload (-ngl)", value=80, precision=0)
    gr.Markdown("### Imatrix Calibration (GGUF Only)")
    with gr.Row():
        enable_imatrix_checkbox = gr.Checkbox(label="Enable Imatrix", value=True)
        imatrix_file_input = gr.Textbox(label="Imatrix File Output Path", value=DEFAULT_IMATRIX_FILE)
    with gr.Row():
        calibration_file_input = gr.Textbox(label="Calibration Data File Path", value=DEFAULT_CALIBRATION_FILE)
        recompute_imatrix_checkbox = gr.Checkbox(label="Compute Imatrix", value=True)
    gr.Markdown("### Cleanup Options")
    with gr.Row():
        delete_original_checkbox = gr.Checkbox(label="Delete Original Model after quantization", value=False)
        delete_quantized_checkbox = gr.Checkbox(label="Delete Quantization Output after upload", value=False)
    with gr.Row():
        run_button = gr.Button("Run Quantization")
    quant_output = gr.Textbox(label="Output Log", interactive=False, lines=20)
    
    run_button.click(
    fn=quant_tavern_ui, 
    inputs=[
        model_ids_input, hf_token_input, username_input,
        gguf_checkbox, gguf_param,
        gptq_checkbox, gptq_param,
        exllamav2_checkbox, exllamav2_param,
        awq_checkbox, awq_param,
        hqq_checkbox, hqq_param,
        enable_imatrix_checkbox, imatrix_file_input,
        calibration_file_input, recompute_imatrix_checkbox,
        # Advanced imatrix parameters:
        imatrix_process_output_checkbox, imatrix_verbosity_input, imatrix_no_ppl_checkbox,
        imatrix_chunk_input, imatrix_output_freq_input, imatrix_save_freq_input,
        imatrix_in_files_input, imatrix_ngl_input,
        delete_original_checkbox, delete_quantized_checkbox
    ], 
    outputs=quant_output
    )

if __name__ == "__main__":
    iface.queue()
    iface.launch(server_name="0.0.0.0", server_port=7860)