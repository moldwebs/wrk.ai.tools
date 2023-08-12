import os, subprocess
import torch

def setup():
    install_cmds = [
        ['pip', 'install', 'ftfy', 'gradio', 'regex', 'tqdm', 'transformers==4.21.2', 'timm', 'fairscale', 'requests'],
        ['pip', 'install', 'open_clip_torch'],
        ['pip', 'install', '-e', 'git+https://github.com/pharmapsychotic/BLIP.git@lib#egg=blip'],
        ['git', 'clone', '-b', 'open-clip', 'https://github.com/pharmapsychotic/clip-interrogator.git']
    ]
    for cmd in install_cmds:
        print(subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode('utf-8'))

setup()

# download cache files
print("Download preprocessed cache files...")
CACHE_URLS = [
    'https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-H-14_laion2b_s32b_b79k_artists.pkl',
    'https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-H-14_laion2b_s32b_b79k_flavors.pkl',
    'https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-H-14_laion2b_s32b_b79k_mediums.pkl',
    'https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-H-14_laion2b_s32b_b79k_movements.pkl',
    'https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-H-14_laion2b_s32b_b79k_trendings.pkl',
]
os.makedirs('cache', exist_ok=True)
for url in CACHE_URLS:
    print(subprocess.run(['wget', url, '-P', 'cache'], stdout=subprocess.PIPE).stdout.decode('utf-8'))

import sys
sys.path.append('src/blip')
sys.path.append('clip-interrogator')

import gradio as gr
from clip_interrogator import Config, Interrogator

config = Config()
config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
config.blip_offload = False if torch.cuda.is_available() else True
config.chunk_size = 2048
config.flavor_intermediate_count = 512
config.blip_num_beams = 64

ci = Interrogator(config)

def inference(image, mode, best_max_flavors):
    image = image.convert('RGB')
    if mode == 'best':
        
        prompt_result = ci.interrogate(image, max_flavors=int(best_max_flavors))
        
        print("mode best: " + prompt_result)
        
        return prompt_result
    
    elif mode == 'classic':
        
        prompt_result = ci.interrogate_classic(image)
        
        print("mode classic: " + prompt_result)
        
        return prompt_result
    
    else:
        
        prompt_result = ci.interrogate_fast(image)
        
        print("mode fast: " + prompt_result)
        
        return prompt_result

with gr.Blocks() as block:
    with gr.Column(elem_id="col-container"):

        input_image = gr.Image(type='pil', elem_id="input-img")
        with gr.Row():
            mode_input = gr.Radio(['best', 'classic', 'fast'], label='Select mode', value='best')
            flavor_input = gr.Slider(minimum=2, maximum=24, step=2, value=4, label='best mode max flavors')
        
        submit_btn = gr.Button("Submit")
        
        output_text = gr.Textbox(label="Description Output", elem_id="output-txt")
                
    submit_btn.click(fn=inference, inputs=[input_image,mode_input,flavor_input], outputs=[output_text], api_name="clipi2")
    
block.queue(max_size=32,concurrency_count=10).launch(server_name="0.0.0.0", debug=True)