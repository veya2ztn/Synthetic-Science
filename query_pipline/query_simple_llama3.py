from vllm import LLM,SamplingParams
import logging
from fastchat.model.model_adapter import Llama3Adapter,get_conversation_template
import numpy as np
from tqdm.auto import tqdm
from typing import List
from .utils import better_latex_sentense_string

Instractions = """
### Prompt to Generate Specific Questions Based on Content

**Objective:** The task is to generate specific and meaningful questions that are highly relevant to the content provided. These questions should delve into the details and nuances of the text, focusing on its specific aspects rather than overarching themes.

**Instructions:**

1. **Read the Text:** Carefully read through the provided content to fully understand the material, including its context, key points, arguments, and any data or conclusions presented.

2. **Identify Key Elements:** Focus on important elements such as:
   - Specific statements or claims made by the author.
   - Data, examples, or case studies mentioned.
   - Theories, models, or frameworks discussed.
   - Any assumptions or implications that underpin the text.

3. **Formulate Questions:** Based on your understanding, formulate questions that:
   - Probe deeper into specific statements or claims: Ask for clarification, justification, or further details.
   - Challenge or explore the validity of data or examples: Inquire about sources, methodologies, or contrasting views.
   - Examine theories, models, or frameworks: Ask how these apply in different contexts or how they compare to alternative approaches.
   - Investigate assumptions or implications: Explore the basis of assumptions or potential impacts and consequences.

4. **Ensure Specificity:** Each question should be directly related to the content, detailed, and specific. Avoid general questions that could apply to any text.

5. **Maintain Relevance:** All questions must be relevant to the text and should help in further understanding or critiquing the content.

This question is specific, probes into the data mentioned, and asks for further critical examination, which aligns with the objective of generating meaningful and specific questions. Below are some positive and negative examples:

**Example of Positive Task Execution:**
    - What is the purpose of cross-correlation analyses in the study of X-ray emissions from active galactic nuclei?"*
    - Why are the cross-correlation analyses in the study of X-ray emissions commonly utilized?
    - Why is the emission from helium-like is not affected by detailed plasma diagnostics as the [Mg tenth] ground to upper level was well shielded and hydrogen had a minimum contribution to the production of the ground state of magnesium. 
    - Why does the plasma not affect the production of the ground state of magnesium?
    - What is the significance of x-ray emission line diagnostics in the study of ionized plasma?

**Example of Negative Task Execution:**
    - What's the capital of U.K?
    - What is the meaning of the word  `column density`?
    - What is the unit of the number 1.8\\*10^-6?
    - How do you define the RG-Ratio?
    - What are the different ways in which the source can be explained?
"""
import re,os

class QuestionMachine_vLLM:
    def __init__(self, model_path='pretrain_weights/Llama3-8B/origin/', enable_prefix_caching=True,trust_remote_code=False ):
        logging.info("loading model...........")
        llm       = LLM(model=model_path, enable_prefix_caching=enable_prefix_caching,trust_remote_code=trust_remote_code)
        self.llm  = llm
        self.model_path = model_path
        self.model_flag = os.path.basename(model_path.rstrip('/')).lower()
        logging.info("loading model........ done~!")
    
    def ask_question(self,*args,**kargs):
        raise NotImplementedError

class SimpleSentenseQvLLM(QuestionMachine_vLLM):
    query_format = 'title+abstract+sentense.detail'
    def format_question_context(self, title, abstract, sentense):
        assert isinstance(title, str)
        assert isinstance(abstract, str)
        title    = better_latex_sentense_string(title)
        abstract = better_latex_sentense_string(abstract)
        sentense = better_latex_sentense_string(sentense)
        conv = Llama3Adapter.get_default_conv_template(None,self.model_path)
        #qs = f"""Read below sentence and tell me its type. The answer should be one word and is one of type from ['Author List', 'Reference', 'Content', 'Meaningless']. There is the sentence \"{sentense}\" """
        qs = f"""Now read the paper:
Ppaer_Name:\n\"\"\"\n{title}\n\"\"\"\n
Paper_Abstract:\n\"\"\"\n{abstract}\n\"\"\"\n
Give me one good question relate to one sentense from the paper. 
The sentense is \n\"\"\"\n{sentense}\n\"\"\"\n
Notice: 
    - please only provide one question.
    - the question must end with `?`
    - Do not return other response except the question itself.
    - A question usually start with How, Who, Why, What and etc.
    - Do not appear more than one question in the response, even it is split via ','.

"""     
        conv.system_message=Instractions
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()
    

    def ask_question(self, title, abstract, sentense,sentence_id):
        context = self.format_question_context(title, abstract, sentense)
        sampling_params = SamplingParams(stop=["?"],max_tokens=150)
        
        result = self.llm.generate(context, sampling_params, use_tqdm=False)
        result = result[0].outputs[0]
        output_string = result.text
        if str(result.finish_reason) == "stop":
            output_string = output_string + "?"
        
        return { 'sentence_id': int(sentence_id), 'result': output_string}

    def ask_question_bulk(self, titles:str|List[str], abstracts:str|List[str], sentenses:List[str], sentence_ids:List[int]):
        assert isinstance(sentenses, list)
        if isinstance(titles, str):
            titles = [titles]*len(sentenses)
        if isinstance(abstracts, str):
            abstracts = [abstracts]*len(sentenses)
        context         = [self.format_question_context(title, abstract,sentense) for title, abstract, sentense in zip(titles, abstracts,sentenses)]
        sampling_params = SamplingParams(stop=["?"],max_tokens=150)
        results= self.llm.generate(context, sampling_params, use_tqdm=False)
        
        results= [r.outputs[0] for r in results]
        assert len(results) == len(sentence_ids)
        outputs = []
        for sentence_id, sentense, result in zip(sentence_ids,sentenses, results):
            output_string = result.text
            if str(result.finish_reason) == "stop":
                output_string = output_string + "?"
            #print(output_string)
            outputs.append( {'sentence_id':int(sentence_id),
                             #'sentence': sentense,  ### if we store the sentense directly, this always means we build a copy of origin paper.
                             'model':self.model_flag, 
                             'prompt_version': self.query_format,
                             'result':output_string })

        return outputs

    def ask_question_batch(self, title, abstracts,sentenses,sentence_ids, batch_size):
        totally_batch_num = int(np.ceil(len(title)/batch_size))
        outputs = []
        for i in tqdm(range(totally_batch_num), leave=False, position=2):
            start_index        = i*batch_size
            end_index          = (i+1)*batch_size
            batch_title        = title[start_index:end_index]
            batch_abstract     = abstracts[start_index:end_index]
            batch_sentenses    = sentenses[start_index:end_index]
            batch_sentence_ids = sentence_ids[start_index:end_index]
            outputs.extend(self.ask_question_bulk(batch_title, batch_abstract,batch_sentenses, batch_sentence_ids))
        return outputs