from .query_methods import generate_with_start_kvcache
from fastchat.model.model_adapter import (
        load_model,
        get_conversation_template,
    )
import logging
from transformers import LlamaForCausalLM
from tqdm.auto import tqdm
class FullPaperQuestionMachine:
    information_need_to_export = ['Result','Abstract','Introduction','Methodology','Literature Review','Discussion']
    def __init__(self, model_path='pretrain_weights/vicuna/vicuna-7b-v1.5-16k'):
        logging.info("loading model...........")
        model, tokenizer = load_model(
            model_path,
            "cuda",
            1,
            load_8bit=False,
            attn_implementation = 'flash_attention_2' ## <--- the quilty via using flash_attn is better than sdpa.  
        )
        self.model = model
        self.tokenizer = tokenizer
        self.model_path = model_path
        logging.info("loading model........ done~!")

    def format_question_context(self, title, abstract, content):
        qs  = f"""
        Here's a research paper titled as "{title}". 
        Find its abstract below: "{abstract}".
        Based on the abstract, provide a brief summary.  Now, please proceed to review the main content of the paper: 
        \"\"\"\n{content}"\n\"\"\"\n 
        Create a comprehensive outline for this paper, detailing its logical structure. Take a step-by-step approach to this task.
        """
        conv = get_conversation_template(self.model_path)
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        history = generate_with_start_kvcache(self.model,self.tokenizer,
                                            {'prompt':conv.get_prompt(),
                                            'max_new_tokens':5000,'temperature':0.7},
                                            'cuda',context_len=16000,return_kvcache=True)
        return qs, history
    

    def ask_question(self, title, abstract, content):
        for iii in tqdm(range(len(self.information_need_to_export)+1), leave=False, position=2):
            if iii ==0:
                qs, context = self.format_question_context(title, abstract, content)
                output_dict = {'outlines':context["text"]}
                continue
            cluster = self.information_need_to_export[iii - 1]
            conv = get_conversation_template(self.model_path)
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], context["text"])
            conv.append_message(conv.roles[0], f"""
                I need you formulate five insightful question about the "{cluster}" part of this paper based on the information given above. Make the question as short as possible. 
                The question should not be general like 'What is the main purpose'. 
                I need at least one 'Why' question, one 'What' question and one 'How' question. 
                """)
            conv.append_message(conv.roles[1], None)
            downstringquestion = generate_with_start_kvcache(self.model, self.tokenizer,
                                                             {'prompt':conv.get_prompt(),'max_new_tokens':2000,'temperature':0.7},
                                                             'cuda',
                                                             context_len=16000,
                                                             start_kvcache=context['last_kvcache'])
            output_dict[f'question_for_{cluster}'] = downstringquestion['text']
        return output_dict

