#Baseline action planner

# Standard library imports
import sys
from os import path
import yaml
import logging
import json # Used for logging and debugging

# Third-party imports
import jinja2
from dotenv import dotenv_values
from openai import OpenAI
from tenacity import retry, stop_after_attempt

# Add parent directories dynamically
sys.path.append(
    path.dirname(path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__))))))
)
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

# Local application/library specific imports
from AIDojoCoordinator.game_components import Action, Observation
from NetSecGameAgents.agents.llm_utils import create_action_from_response, create_status_from_state
import validate_responses


class ConfigLoader:
    """Class to handle loading YAML configurations."""
    
    @staticmethod
    def load_config(file_name: str = 'prompts.yaml') -> dict:
        possible_paths = [
            path.join(path.dirname(__file__), file_name),
            path.join(path.dirname(path.dirname(__file__)), file_name),
            path.join(path.dirname(path.dirname(path.dirname(__file__))), file_name),
        ]
        for yaml_file in possible_paths:
            if path.exists(yaml_file):
                with open(yaml_file, 'r') as file:
                    return yaml.safe_load(file)
        raise FileNotFoundError(f"{file_name} not found in expected directories.")


class LLMActionPlanner:
    def __init__(self, model_name: str, goal: str, memory_len: int = 10, api_url=None, config: dict = None):
        self.model = model_name
        self.config = config or ConfigLoader.load_config()

        if "gpt" in self.model:
            env_config = dotenv_values(".env")
            self.client = OpenAI(api_key=env_config["OPENAI_API_KEY"])
        else:
            self.client = OpenAI(base_url=api_url, api_key="ollama")
        
        self.memory_len = memory_len
        self.logger = logging.getLogger("REACT-agent")
        self.update_instructions(goal.lower())
        self.prompts = []
        self.states = []
        self.responses = []

    def get_prompts(self) -> list:
        return self.prompts

    def get_responses(self) -> list:
        return self.responses
    
    def get_states(self) -> list:
        return self.states
    
    def update_instructions(self, new_goal: str) -> None:
        template_str = self.config['prompts']['INSTRUCTIONS_TEMPLATE']
        
        # Compile all action examples into a single string
        examples_text = ""
        for example in self.config['action_examples'].values():
            examples_text += f"{example}\n"
            
        template = jinja2.Environment().from_string(template_str)
        self.instructions = template.render(goal=new_goal, action_examples=examples_text)

    def create_mem_prompt(self, memory_list: list) -> str:
        prompt = ""
        for memory, goodness in memory_list:
            prompt += f"You have taken action {memory} in the past. This action was {goodness}.\n"
        return prompt

    @retry(stop=stop_after_attempt(3))
    def openai_query(self, msg_list: list, max_tokens: int = 60, model: str = None, fmt=None):
        llm_response = self.client.chat.completions.create(
            model=model or self.model,
            messages=msg_list,
            max_tokens=max_tokens,
            temperature=0.0,
            response_format=fmt or {"type": "text"},
        )
        return llm_response.choices[0].message.content

    def parse_response(self, llm_response: str, state: Observation.state):
        response_dict = {"action": None, "parameters": None}
        valid = False
        action = None
        try:
            validated_response, error_msg = validate_responses.validate_agent_response(llm_response)
            if validated_response is None:
                self.logger.error(f"Validation failed: {error_msg}")
                response_dict["action"] = "InvalidResponse"
                response_dict["parameters"] = {"error": error_msg, "original": llm_response}
                return valid, response_dict, action
            
            response = validated_response
            action_str = response.get("action", None)
            action_params = response.get("parameters", None)
            
            if action_str and action_params is not None:
                valid, action = create_action_from_response(response, state)
                response_dict["action"] = action_str
                response_dict["parameters"] = action_params
            else:
                self.logger.warning("Missing action or parameters in LLM response.")
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            response_dict["action"] = "InvalidJSON"
            response_dict["parameters"] = llm_response
        except Exception as e:
            self.logger.error(f"Unexpected error in parse_response: {e}")
        return valid, response_dict, action

    def get_action_from_obs_react(self, observation: Observation, memory_buf: list) -> tuple:
        """
        Plans an action using a three-stage LLM reasoning process.
        - Stage 0: Iterates through all possible action types to find which ones are currently valid.
        - Stage 1: Asks the LLM to create a tactical plan given the list of valid action types.
        - Stage 2: Asks the LLM to generate the final JSON action based on the plan from Stage 1.
        """
        self.states.append(observation.state.as_json())
        status_prompt = create_status_from_state(observation.state)
        memory_prompt = self.create_mem_prompt(memory_buf)

        # ----------- STAGE 0: FIND ALL VALID ACTION TYPES -----------
        print("\n[STAGE 0] Finding all available valid action types...")
        q0_config = self.config['questions'][0]
        q0_template = q0_config['text']
        q0_rules = q0_config['rules']
        
        available_action_types = []

        for action_type, specific_rule in q0_rules.items():
            q0_formatted = q0_template.format(action_type=action_type, specific_rule=specific_rule)
            
            messages_stage0 = [
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": status_prompt},
                {"role": "user", "content": q0_formatted},
            ]
            
            try:
                response_stage0 = self.openai_query(messages_stage0, max_tokens=10)
                clean_response = response_stage0.strip()
                self.logger.info(f"Stage 0 check for '{action_type}': response '{clean_response}'")
                
                num_valid_actions = int(clean_response)
                if num_valid_actions > 0:
                    print(f"  - '{action_type}' is valid ({num_valid_actions} targets).")
                    available_action_types.append(action_type)
                else:
                    print(f"  - '{action_type}' is not valid.")
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Could not parse Stage 0 response for '{action_type}' as integer: '{clean_response}'. Skipping.")
                print(f"  - Could not parse response for '{action_type}'. Skipping.")
            except Exception as e:
                self.logger.error(f"An unexpected error in Stage 0 check for '{action_type}': {e}. Skipping.")
                print(f"  - Error checking '{action_type}'. Skipping.")

        if not available_action_types:
            self.logger.error("Failed to find any viable action types.")
            print("[STAGE 0] No valid action types found. Cannot proceed.")
            return False, {"action": "NoAction", "parameters": "No viable action types found."}, None
        
        print(f"[STAGE 0] Available actions for next step: {available_action_types}")

        # ----------- STAGE 1: REASONING & PLANNING -----------
        print(f"\n[STAGE 1] Generating tactical plan from options: {available_action_types}...")

        q1_template = self.config['questions'][1]['text']
        q1_formatted = q1_template.format(available_action_types=', '.join(available_action_types))
        
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": f"{status_prompt}\n{memory_prompt}"},
            {"role": "user", "content": q1_formatted}
        ]
        
        self.logger.info(f"Stage 1 Full Prompt:\n{json.dumps(messages, indent=2)}")
        
        reasoning_response = self.openai_query(messages, max_tokens=512)
        self.logger.info(f"Stage 1 Reasoning Response:\n{reasoning_response}")
        print(f"[STAGE 1] LLM Tactical Plan:\n{reasoning_response}")

        # ----------- STAGE 2: FINAL JSON GENERATION -----------
        print(f"\n[STAGE 2] Generating final JSON action from plan...")

        q2_template = self.config['questions'][2]['text']
        
        messages.append({"role": "assistant", "content": reasoning_response})
        messages.append({"role": "user", "content": q2_template})

        self.logger.info(f"Stage 2 Full Prompt:\n{json.dumps(messages, indent=2)}")
        
        final_json_response = self.openai_query(messages, max_tokens=200, fmt={"type": "json_object"})
        
        self.logger.info(f"Stage 2 Final JSON Response: {final_json_response}")
        print(f"[STAGE 2] Final JSON Response: {final_json_response}")
        
        return self.parse_response(final_json_response, observation.state)