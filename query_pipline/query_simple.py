from vllm import LLM,SamplingParams
import logging
from fastchat.model.model_adapter import get_conversation_template
import numpy as np
from tqdm.auto import tqdm
class QuestionMachine_vLLM:
    def __init__(self, model_path='pretrain_weights/vicuna/vicuna-7b-v1.5-16k'):
        logging.info("loading model...........")
        llm       = LLM(model=model_path)
        self.llm  = llm
        self.model_path = model_path
        logging.info("loading model........ done~!")
    
    def ask_question(self,*args,**kargs):
        raise NotImplementedError

class SimpleSentenseQvLLM(QuestionMachine_vLLM):
    def format_question_context(self, title, sentense):
        conv = get_conversation_template(self.model_path)
        #qs = f"""Read below sentence and tell me its type. The answer should be one word and is one of type from ['Author List', 'Reference', 'Content', 'Meaningless']. There is the sentence \"{sentense}\" """
        qs = f"""
        I have a specific paragraph from a scholarly paper named <{title}>. 
        I need your help to formulate an insightful question based on the information given. 
        The specific paragraph from the paper is:
        \"\"\"\n{sentense} \n\"\"\"
        Following the question, provide a one-sentence response that succinctly answers it. 
        The response should be start with "What is". Make the response as short as possible. 
        The response should not be general like "What is the main purpose" 
        """
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()
    

    def ask_question(self, title, sentense,sentence_id):
        context = self.format_question_context(title, sentense)
        sampling_params = SamplingParams(stop=["?"],max_tokens=150)
        result = self.llm.generate(context, sampling_params, use_tqdm=False)
        result = result[0].outputs[0]
        output_string = result.text
        if str(result.finish_reason) == "stop":
            output_string = output_string + "?"
        return { 'sentence_id': int(sentence_id), 'result': output_string}

    def ask_question_bulk(self, title, sentenses,sentence_ids):
    
        context         = [self.format_question_context(title, sentense) for sentense in sentenses]
        sampling_params = SamplingParams(stop=["?"],max_tokens=150)
        results= self.llm.generate(context, sampling_params, use_tqdm=False)
        
        results= [r.outputs[0] for r in results]
        assert len(results) == len(sentence_ids)
        outputs = []
        for sentence_id, result in zip(sentence_ids,results):
            output_string = result.text
            if str(result.finish_reason) == "stop":
                output_string = output_string + "?"
            #print(output_string)
            outputs.append( { 'sentence_id': int(sentence_id), 'result': output_string})
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
            outputs.extend(self.ask_question_bulk(batch_title, batch_sentenses, batch_sentence_ids))
        return outputs