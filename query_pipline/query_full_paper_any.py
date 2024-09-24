from tqdm.auto import tqdm
from .utils import better_latex_sentense_string
from .query_simple_llama3 import QuestionMachine_vLLM

from vllm import LLM,SamplingParams
Instractions = """
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
from fastchat.conversation import register_conv_template, Conversation, SeparatorStyle,get_conv_template
register_conv_template(
            Conversation(
                name="phi3",
                roles=("<|user|>", "<|assistant|>"),
                sep_style=SeparatorStyle.DEFAULT,
                sep="<|end|>",
            )
        )
def obtain_conv_template(model_path):
    if 'llama3' in model_path:
        from fastchat.model.model_adapter import Llama3Adapter
        return Llama3Adapter.get_default_conv_template(None,model_path)
    if 'phi-3-mini' in model_path:
        
        
        return get_conv_template('phi3')
    raise NotImplementedError

class FullPaperQuestionMachine(QuestionMachine_vLLM):
    information_need_to_export = ['Result','Abstract','Introduction','Methodology','Literature Review','Discussion']
    query_format = 'title+content'
    def ask_question(self, title, abstract, content):
        title    = better_latex_sentense_string(title)
        abstract = better_latex_sentense_string(abstract)
        output_dict={ 
            'model':self.model_flag, 
            'prompt_version': self.query_format
            }
        for iii in tqdm(range(len(self.information_need_to_export)), leave=False, position=2):
            conv = obtain_conv_template(self.model_path)
            conv.system_message= Instractions
            cluster = self.information_need_to_export[iii]
            qs = f"""Now read the paper:
Ppaer_Name:\n\"\"\"\n{title}\n\"\"\"\n
And its content below:\n\"\"\"\n{content}\n\"\"\"\n
I need you formulate five insightful question about the view from "{cluster}" part of this paper based on the information given above. 
I need at least one 'Why' question, one 'What' question and one 'How' question. 
You only need directly give the five question and end each question with "?".
"""     

            conv.system_message=Instractions
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            sampling_params = SamplingParams(stop=['<|eot_id|>'], max_tokens=1000)
            result = self.llm.generate(prompt, sampling_params, use_tqdm=False)
            result = result[0].outputs[0]
            output_string = result.text

            output_dict[f'question_for_{cluster}'] = output_string
        return output_dict

