import gradio as gr
import os
from ctransformers import AutoModelForCausalLM, AutoConfig, Config #import for GGML models
import datetime

#MODEL SETTINGS also for DISPLAY
convHistory = ''
modelfile = "model/claude2-alpaca-7b.Q4_K_M.gguf"
repetitionpenalty = 1.15
contextlength=4096
logfile = 'Claude2Alpaca_logs.txt'
print("loading model...")
stt = datetime.datetime.now()
conf = AutoConfig(Config(temperature=0.3, 
                         repetition_penalty=repetitionpenalty, 
                         batch_size=64,
                        max_new_tokens=2048, 
                        context_length=contextlength))
llm = AutoModelForCausalLM.from_pretrained(modelfile,
                                        model_type="llama", 
                                        config = conf)
dt = datetime.datetime.now() - stt
print(f"Model loaded in {dt}")

def writehistory(text):
    with open(logfile, 'a') as f:
        f.write(text)
        f.write('\n')
    f.close()

"""
gr.themes.Base()
gr.themes.Default()
gr.themes.Glass()
gr.themes.Monochrome()
gr.themes.Soft()
"""
def combine(a, b, c, d):
    global convHistory
    import datetime
    SYSTEM_PROMPT = f"""{a}


    """        
    temperature = c
    max_new_tokens = d
    prompt = a + "\n\n### Instruction:\n" + b + "\n\n### Response:\n"
    start = datetime.datetime.now()
    generation = ""
    delta = ""
    prompt_tokens = f"Prompt Tokens: {len(llm.tokenize(prompt))}"
    answer_tokens = ''
    total_tokens = ''   
    for character in llm(prompt, 
                 temperature = temperature, 
                 repetition_penalty = 1.15, 
                 max_new_tokens=max_new_tokens,
                 stream = True):
        generation += character
        answer_tokens = f"Out Tkns: {len(llm.tokenize(generation))}"
        total_tokens = f"Total Tkns: {len(llm.tokenize(prompt)) + len(llm.tokenize(generation))}"
        delta = datetime.datetime.now() - start
        yield generation, delta, prompt_tokens, answer_tokens, total_tokens
    timestamp = datetime.datetime.now()
    logger = f"""time: {timestamp}\n Temp: {temperature} - MaxNewTokens: {max_new_tokens} - RepPenalty: 1.5 \nPROMPT: \n{prompt}\nClaude2Alpaca-7B: {generation}\nGenerated in {delta}\nPromptTokens: {prompt_tokens}   Output Tokens: {answer_tokens}  Total Tokens: {total_tokens}\n\n---\n\n"""
    writehistory(logger)
    convHistory = convHistory + prompt + "\n" + generation + "\n"
    print(convHistory)
    return generation, delta, prompt_tokens, answer_tokens, total_tokens    
    #return generation, delta


# MAIN GRADIO INTERFACE
with gr.Blocks(theme='remilia/Ghostly') as demo:   #theme=gr.themes.Glass()
    #TITLE SECTION
    with gr.Row(variant='compact'):
            with gr.Column(scale=12):
                gr.HTML("<center>"
                + "<h3>Prompt Engineering Playground!</h3>"
                + "<h1>🧠🦙 Claude2-Alpaca-7b 4K context window</h2>"
                + "<p>Test your favourite LLM for advanced inferences</p></center>")  
            gr.Image(value='./claude2alpaca_logo.png', width=80, show_label = False, 
                     show_download_button = False, container = False)    
    # INTERACTIVE INFOGRAPHIC SECTION
    with gr.Row():
        with gr.Column(min_width=80):
            gentime = gr.Textbox(value="", placeholder="Generation Time:", min_width=50, show_label=False)                          
        with gr.Column(min_width=80):
            prompttokens = gr.Textbox(value="", placeholder="Prompt Tkn:", min_width=50, show_label=False)
        with gr.Column(min_width=80):
            outputokens = gr.Textbox(value="", placeholder="Output Tkn:", min_width=50, show_label=False)            
        with gr.Column(min_width=80):
            totaltokens = gr.Textbox(value="", placeholder="Total Tokens:", min_width=50, show_label=False)  

    # PLAYGROUND INTERFACE SECTION
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
            f"""
            ### Tunning Parameters""")
            temp = gr.Slider(label="Temperature",minimum=0.0, maximum=1.0, step=0.01, value=0.1)
            max_len = gr.Slider(label="Maximum output lenght", minimum=10,maximum=2048,step=2, value=1024)
            gr.Markdown(
            """
            Fill the System Prompt and User Prompt
            And then click the Button below
            """)
            btn = gr.Button(value="🧠🦙 Generate", variant='primary')
            gr.Markdown(
            f"""
            - **Prompt Template**: Alpaca 🦙
            - **Repetition Penalty**: {repetitionpenalty}
            - **Context Lenght**: {contextlength} tokens
            - **LLM Engine**: CTransformers
            - **Model**: 🧠🦙 Claude2-Alpaca-7b
            - **Log File**: {logfile}
            """) 


        with gr.Column(scale=4):
            txt = gr.Textbox(label="System Prompt", lines=2)
            txt_2 = gr.Textbox(label="User Prompt", lines=5)
            txt_3 = gr.Textbox(value="", label="Output", lines = 10, show_copy_button=True)
            btn.click(combine, inputs=[txt, txt_2,temp,max_len], outputs=[txt_3,gentime,prompttokens,outputokens,totaltokens])


if __name__ == "__main__":
    demo.launch(inbrowser=True)