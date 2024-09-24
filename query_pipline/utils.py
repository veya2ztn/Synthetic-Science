import re
def multispaces_into_singlespace(text:str):
    return re.sub(r'\s+', ' ', text)

def better_latex_sentense_string(latex_string:str):
    if latex_string is None:return None
    latex_string = latex_string.replace('\n'," ")
    latex_string = multispaces_into_singlespace(latex_string)
    return latex_string.strip()

class QuestionMachine_vLLM:
    def __init__(self, model_path='pretrain_weights/Llama3-8B/origin/'):
        logging.info("loading model...........")
        llm       = LLM(model=model_path, enable_prefix_caching=True)
        self.llm  = llm
        self.model_path = model_path
        self.model_flag = os.path.basename(model_path.rstrip('/')).lower()
        logging.info("loading model........ done~!")
    
    def ask_question(self,*args,**kargs):
        raise NotImplementedError