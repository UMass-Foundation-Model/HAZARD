HAZARD provide two versions of LLM-based agent. The overall framework of LLM_agent can be summarized as follows:
 ![[LLM_framework.pdf]]
 Compared with version 1.0, the prompt of version 2.0 additionally includes the history of object information.
### Version 1.0
* Usage: `--agent_name llm --api_key_file <a file with your openai api key in it>`
* Parameter:`--lm_id gpt-3.5-turbo` for GPT-3.5 backbone or `--lm_id gpt-4` for GPT-4 backbone
* Prompt example
![[detail_prompt_v1.pdf]]

### Version 2.0
* Usage: `--agent_name llm --api_key_file <a file with your openai api key in it>`
* Parameter:`--lm_id gpt-3.5-turbo` for GPT-3.5 backbone or `--lm_id gpt-4` for GPT-4 backbone
* Prompt example
![[detail_prompt_v2.pdf]]