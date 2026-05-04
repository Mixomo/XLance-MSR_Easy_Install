import os
import torch
import gradio as gr
import tempfile
import numpy as np
try:
    import spaces
except ImportError:
    # Not running on HF Spaces — create a no-op decorator
    class spaces:
        @staticmethod
        def GPU(fn):
            return fn
from inference_full import inference_main, load_audio, save_audio
from huggingface_hub import hf_hub_download

# ===== Basic config =====
USE_CUDA = torch.cuda.is_available()
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "12"))
REPO_ID = os.getenv("MODEL_REPO_ID", "chenxie95/xlance-msr-ckpt")

# Instrument to checkpoint mapping
INSTRUMENT_MAP = {
    'vox': {
        'pre': ['denoise.pth'],
        'mss': ['vox_mss.pth'],
        'post': ['dereverb.pth']
    },
    'gtr': {
        'pre': ['denoise.pth'],
        'mss': ['gtr_mss.pth'],
        'post': []
    },
    'key': {
        'pre': ['denoise.pth'],
        'mss': ['key_mss.pth'],
        'post': []
    },
    'syn': {
        'pre': ['denoise.pth'],
        'mss': ['syn_mss.pth', 'syn_mss1.pth'],
        'post': []
    },
    'bass': {
        'pre': ['denoise.pth'],
        'mss': ['bass_mss.pth'],
        'post': []
    },
    'drums': {
        'pre': ['denoise.pth'],
        'mss': ['drums_mss.pth', 'drums_mss1.pth'],
        'post': []
    },
    'perc': {
        'pre': ['denoise.pth'],
        'mss': ['perc_mss.pth', 'perc_mss1.pth'],
        'post': []
    },
    'orch': {
        'pre': ['denoise.pth'],
        'mss': ['orch_mss.pth', 'orch_mss1.pth'],
        'post': []
    }
}

# Cache for downloaded models
MODEL_CACHE = {}

def download_model(filename):
    """Download and cache a model file in the local 'checkpoints' directory using modern hf_hub_download"""
    if filename not in MODEL_CACHE:
        # Save to a local folder instead of the default cache
        local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
        os.makedirs(local_dir, exist_ok=True)
        
        print(f"Checking/Downloading {filename} to {local_dir}...")
        MODEL_CACHE[filename] = hf_hub_download(
            repo_id=REPO_ID, 
            filename=filename,
            local_dir=local_dir
        )
    return MODEL_CACHE[filename]

def prepare_models(instrument):
    """Download all required models for the instrument"""
    config = INSTRUMENT_MAP[instrument]
    pre_models = [download_model(m) for m in config['pre']]
    mss_models = [download_model(m) for m in config['mss']]
    post_models = [download_model(m) for m in config['post']]
    return pre_models, mss_models, post_models

@spaces.GPU
def process_audio_batch(audio_path, selected_stems, export_options, progress=gr.Progress()):
    if not audio_path:
        return None, None, "Please upload an audio file first."
    if not selected_stems:
        return None, None, "Please select at least one instrument."
    if not export_options:
        return None, None, "Please select at least one export mode."
    
    try:
        orig_audio, sr = load_audio(audio_path)
    except Exception as e:
        return None, None, f"Error loading audio: {str(e)}"
    
    results = []
    # Usar ruta absoluta para evitar confusiones con carpetas temporales
    project_root = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    print(f"\n--- Iniciando procesamiento: {base_name} ---")
    print(f"Directorio de salida: {output_dir}")
    
    combined_stems = None
    intermediate_files = []
    
    for instrument in progress.tqdm(selected_stems, desc="Processing instruments"):
        pre_models, mss_models, post_models = prepare_models(instrument)
        
        # Archivo para el stem extraído
        stem_filename = f"{base_name}_{instrument}_stem.wav"
        stem_path = os.path.join(output_dir, stem_filename)
        
        class Args:
            checkpoint_pre = pre_models
            checkpoint = mss_models
            checkpoint_post = post_models
            input_dir = audio_path
            output_dir = stem_path
            device = "cuda" if torch.cuda.is_available() else "cpu"
            batch_size = BATCH_SIZE
        
        print(f"Extracting '{instrument}'...")
        inference_main(Args())
        
        # Cargar el stem para la lógica de "Minus Stem"
        sep_audio, _ = load_audio(stem_path)
        
        if combined_stems is None:
            combined_stems = np.zeros_like(sep_audio)
        
        # Acumular stems
        min_len = min(combined_stems.shape[1], sep_audio.shape[1])
        combined_stems[:, :min_len] += sep_audio[:, :min_len]

        if "Only Stems" in export_options:
            results.append(stem_path)
        else:
            # Si no se pidió exportar el stem, lo guardamos para borrarlo después de la resta
            intermediate_files.append(stem_path)
            
    if "Minus Stems (Karaoke)" in export_options and combined_stems is not None:
        print("Generating Karaoke track (Original - Selected stems)...")
        min_len = min(orig_audio.shape[1], combined_stems.shape[1])
        minus_audio = orig_audio[:, :min_len] - combined_stems[:, :min_len]
        
        # Evitar distorsión
        minus_audio = np.clip(minus_audio, -1.0, 1.0)
        
        stems_suffix = "_".join(selected_stems)
        backing_filename = f"{base_name}_minus_{stems_suffix}.wav"
        backing_path = os.path.join(output_dir, backing_filename)
        save_audio(minus_audio, sr, backing_path)
        results.append(backing_path)
    
    # Cleanup: delete files that were not explicitly requested
    if "Only Stems" not in export_options:
        for f in intermediate_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except:
                pass
            
    if not results:
        return [], "No files generated. Check export options."
        
    print(f"Processing finished. {len(results)} files generated in {output_dir}")
    return results, f"Completed! Files saved in 'outputs' folder."

# ===== Gradio UI =====
with gr.Blocks(title="XLance-MSR Pro") as demo:
    gr.HTML("<h1 style='text-align: center;'>🎵 XLance-MSR Audio Master</h1>")
    gr.HTML("<p style='text-align: center;'>Professional-grade AI Audio Separation using Multi-Stem Reconstruction</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Input Audio Source"
            )
            
            with gr.Group():
                stems_selection = gr.CheckboxGroup(
                    choices=[
                        ("Vocals", "vox"), 
                        ("Guitars", "gtr"), 
                        ("Keyboards", "key"), 
                        ("Synthesizers", "syn"), 
                        ("Bass", "bass"), 
                        ("Drums", "drums"), 
                        ("Percussion", "perc"), 
                        ("Orchestra", "orch")
                    ],
                    value=["vox"],
                    label="1. Stems to extract",
                    info="Select what you want to extract from the song"
                )
                
                export_options = gr.CheckboxGroup(
                    choices=["Only Stems", "Minus Stems (Karaoke)"],
                    value=["Only Stems"],
                    label="2. Export options",
                    info="Stems = separate tracks | Karaoke = song without those instruments"
                )
            
            process_btn = gr.Button("🚀 Start Separation", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            status = gr.Textbox(
                label="Status",
                value="Ready",
                interactive=False
            )
            
            # State to hold the list of generated file paths
            results_state = gr.State([])
            
            @gr.render(inputs=results_state)
            def render_audio_results(files):
                if not files:
                    gr.Markdown("### ⌛ Processing...")
                    return
                
                with gr.Column():
                    gr.Markdown(f"### ✨ Generated Files ({len(files)})")
                    for path in files:
                        label = "Instrument" if "_stem.wav" in path else "Karaoke Mix"
                        if "_vox_stem" in path: label = "🎤 Vocals"
                        elif "_drums_stem" in path: label = "🥁 Drums"
                        elif "_bass_stem" in path: label = "🎸 Bass"
                        elif "_gtr_stem" in path: label = "🎸 Guitar"
                        elif "minus_" in path: label = "🎹 Karaoke Track (Minus)"
                        
                        gr.Audio(
                            value=path, 
                            label=f"{label}: {os.path.basename(path)}", 
                            interactive=False,
                            waveform_options=gr.WaveformOptions(
                                waveform_color="#4b6cb7",
                                waveform_progress_color="#182848",
                            )
                        )
                    gr.File(value=files, label="Download all files")
    
    # Button click
    process_btn.click(
        fn=process_audio_batch,
        inputs=[audio_input, stems_selection, export_options],
        outputs=[results_state, status]
    )
    
    # Examples
    gr.Examples(
        examples=[
            ["examples/forget.mp3", ["vox"], ["Only Stems"]],
            ["examples/forget.mp3", ["drums"], ["Only Stems", "Minus Stems (Karaoke)"]],
            ["examples/sonata.mp3", ["key", "orch"], ["Only Stems"]],
        ],
        inputs=[audio_input, stems_selection, export_options],
        label="Quick Examples",
        examples_per_page=3,
    )

# Queue: keep a small queue to avoid OOM
demo.queue(max_size=8)
demo.launch(css=".gradio-container {max-width: 1100px !important}")