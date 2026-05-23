# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" Class for sampling new program skeletons. """
from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Collection, Sequence, Type
import numpy as np
import time

from litesr.solver_agent import evaluator
from litesr.solver_agent import buffer
from litesr import config as config_lib
import requests
import json
import http.client
import os
from pathlib import Path
from litesr.literature_agent.rag import LiteratureAgent
from litesr.clients import get_anthropic_client, get_ollama_client, get_openai_client, get_hf_client

try:
    from openai import OpenAI
    _OPENAI_SDK_AVAILABLE = True
except ImportError:
    _OPENAI_SDK_AVAILABLE = False


class LLM(ABC):
    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = samples_per_prompt

    def _draw_sample(self, prompt: str) -> str:
        raise NotImplementedError('Must provide a language model.')

    @abstractmethod
    def draw_samples(self, prompt: str) -> Collection[str]:
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]


class Sampler:
    """ Node that samples program skeleton continuations and sends them for analysis. """
    _global_samples_nums: int = 1
    _epoch_num: int = 0
    _rag_active: bool = True
    _rag_stall_count: int = 0
    _rag_last_best: float = -float('inf')

    def __init__(
            self,
            database: buffer.ExperienceBuffer,
            evaluators: Sequence[evaluator.Evaluator],
            samples_per_prompt: int,
            config: config_lib.Config,
            max_sample_nums: int | None = None,
            llm_class: Type[LLM] = LLM,
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluators = evaluators
        self._llm = llm_class(samples_per_prompt)
        self._max_sample_nums = max_sample_nums
        self.config = config

    def sample(self, **kwargs):
        """ Continuously gets prompts, samples programs, sends them for analysis. """
        while True:
            if self._max_sample_nums and self.__class__._global_samples_nums >= self._max_sample_nums:
                break
            if self.__class__._epoch_num == 202:
                break

            self.__class__._epoch_num += 1
            self._on_epoch_start(self.__class__._epoch_num)

            # ── Adaptive RAG toggle ────────────────────────────────────────────
            current_best = max(self._database._best_score_per_island)
            threshold = getattr(self.config, 'rag_improvement_threshold', 0.01)
            if current_best - self.__class__._rag_last_best >= threshold:
                self.__class__._rag_last_best = current_best
                self.__class__._rag_stall_count = 0
            else:
                self.__class__._rag_stall_count += 1
                patience = getattr(self.config, 'rag_toggle_patience', 10)
                if self.__class__._rag_stall_count >= patience:
                    self.__class__._rag_active = not self.__class__._rag_active
                    self.__class__._rag_stall_count = 0
                    state = "ON" if self.__class__._rag_active else "OFF"
                    print(f"[RAG] No improvement for {patience} epochs → RAG {state}")

            prompt = self._database.get_prompt()
            reset_time = time.time()

            samples = self._llm.draw_samples(
                prompt.code, self.config, self._database._last_reset_time,
                rag_active=self.__class__._rag_active,
            )
            sample_time = (time.time() - reset_time) / self._samples_per_prompt

            for sample_index, sample in enumerate(samples):
                self._global_sample_nums_plus_one()
                cur_global_sample_nums = self._get_global_sample_nums()
                chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
                chosen_evaluator.analyse(
                    sample,
                    prompt.island_id,
                    prompt.version_generated,
                    **kwargs,
                    global_sample_nums=cur_global_sample_nums,
                    epoch_num=self.__class__._epoch_num,
                    sample_time=sample_time,
                )

    def _on_epoch_start(self, epoch_num: int) -> None:
        """Hook called at the start of every evolutionary step."""
        pass

    def _get_global_sample_nums(self) -> int:
        return self.__class__._global_samples_nums

    def set_global_sample_nums(self, num):
        self.__class__._global_samples_nums = num

    def _global_sample_nums_plus_one(self):
        self.__class__._global_samples_nums += 1


def _extract_body(sample: str, config: config_lib.Config) -> str:
    """Extract the function body from a response sample, removing the function signature."""
    lines = sample.splitlines()
    func_body_lineno = 0
    find_def_declaration = False

    for lineno, line in enumerate(lines):
        if line[:3] == 'def':
            func_body_lineno = lineno
            find_def_declaration = True
            break

    if find_def_declaration:
        if config.use_api:
            code = ''
            for line in lines[func_body_lineno + 1:]:
                code += line + '\n'
        else:
            code = ''
            indent = '    '
            for line in lines[func_body_lineno + 1:]:
                if line[:4] != indent:
                    line = indent + line
                code += line + '\n'
        return code

    return sample


class LocalLLM(LLM):
    def __init__(self, samples_per_prompt: int, batch_inference: bool = True, trim=True) -> None:
        super().__init__(samples_per_prompt)
        self._batch_inference = batch_inference
        self._url = "http://127.0.0.1:5000/completions"
        self._instruction_prompt = (
            "You are a helpful assistant tasked with discovering mathematical function structures "
            "for scientific systems. Complete the 'equation' function below, considering the "
            "physical meaning and relationships of inputs.\n\n"
        )
        self._trim = trim

    @staticmethod
    def combine_n_texts(data_list, n, key='text'):
        selected_texts = [item[key] for item in data_list[:n] if key in item]
        return " \n".join(selected_texts)

    def draw_samples(self, prompt: str, config: config_lib.Config, last_reset_time: float, rag_active: bool = True) -> Collection[str]:
        """Returns multiple equation program skeleton hypotheses for the given `prompt`."""
        rag_context = ""
        if rag_active:
            print("[RAG] Local RAG activated")
            rag = LiteratureAgent(
                milvus_uri=os.environ.get("MILVUS_URI", "http://localhost:19530"),
                local_embedding=True,
                use_ollama=getattr(config, 'rag_use_ollama', True),
                ollama_model=getattr(config, 'rag_ollama_model', 'mistral:latest'),
            )
            papers_dir = "./papers"
            all_paths = [str(p) for p in Path(papers_dir).glob("*.pdf") if p.is_file()]
            if all_paths:
                rag.index_pdfs(all_paths, force_update=False)

            rag_result = rag.agentic_query(prompt, return_sources=True)
            rag_context = rag_result["answer"]

        if rag_context:
            SYNTHESIS_PROMPT_TEMPLATE = """\
            You are a scientific code generation assistant.

            TASK: Complete the Python function below Update.
            Output the COMPLETE Python function — starting with the def signature, then the full body.
            No explanation, no prose, no markdown, no helper functions, no imports, NO COMMENTS.

            ════════════════════════════════════════════════════════════════════
            CONTEXT FOR THE EQUATION:
            ════════════════════════════════════════════════════════════════════
            {rag_context}

            IMPORTANT:
            - Improved the function if improvement required according to the context or else keep the same version.
            - The equations above use LaTeX mathematical notation — they are NOT code.
            - Translate them into numpy expressions inside the single function below.
            - Do NOT define any new functions, classes, or imports.
            - Do NOT write any comments (no # lines, no inline comments, no docstrings).
            - Use only the variables in the function signature and the `params` array for all constants.
            - Your response must begin with the def signature and end with the return statement. Nothing else.
            ════════════════════════════════════════════════════════════════════

            {function_with_signature}
            """
            prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
                rag_context=rag_context,
                function_with_signature=prompt,
            )

        if config.use_api:
            if getattr(config, 'use_hf_api', False):
                return self._draw_samples_hf(prompt, config)
            elif getattr(config, 'use_openai_sdk', False):
                return self._draw_samples_sdk(prompt, config)
            elif getattr(config, 'use_ollama', False):
                return self._draw_samples_ollama(prompt, config)
            elif getattr(config, 'use_anthropic', False):
                return self._draw_samples_anthropic(prompt, config)
            else:
                return self._draw_samples_api(prompt, config)
        else:
            return self._draw_samples_local(prompt, config)

    def _draw_samples_local(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        prompt = '\n'.join([self._instruction_prompt, prompt])
        while True:
            try:
                raw_samples = []
                if self._batch_inference:
                    response = self._do_request(prompt)
                    for res in response:
                        raw_samples.append(res)
                else:
                    for _ in range(self._samples_per_prompt):
                        response = self._do_request(prompt)
                        raw_samples.append(response)

                if self._trim:
                    return [_extract_body(s, config) for s in raw_samples]
                return raw_samples
            except Exception:
                continue

    def _draw_samples_ollama(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        """Draw samples from a local Ollama instance using its OpenAI-compatible API."""
        if not _OPENAI_SDK_AVAILABLE:
            raise ImportError("openai package is required: pip install openai")

        base_url = getattr(config, 'ollama_base_url', "http://localhost:11434/v1")
        model    = getattr(config, 'ollama_model', "mistral:latest")
        client   = get_ollama_client(base_url)
        prompt_with_instruction = '\n'.join([self._instruction_prompt, prompt])
        all_samples = []

        for _ in range(self._samples_per_prompt):
            while True:
                try:
                    response = client.chat.completions.create(
                        model=model,
                        max_tokens=4096,
                        messages=[{"role": "user", "content": prompt_with_instruction}],
                    )
                    raw = response.choices[0].message.content
                    all_samples.append(_extract_body(raw, config) if self._trim else raw)
                    break
                except Exception as e:
                    print(f"[Ollama] request failed: {e}, retrying...")
                    continue

        return all_samples

    def _draw_samples_api(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        """Draw samples via raw HTTPS to the OpenAI API."""
        prompt_with_instruction = '\n'.join([self._instruction_prompt, prompt])
        all_samples = []

        for _ in range(self._samples_per_prompt):
            while True:
                try:
                    conn = http.client.HTTPSConnection("api.openai.com")
                    payload = json.dumps({
                        "max_tokens": 512,
                        "model": config.api_model,
                        "messages": [{"role": "user", "content": prompt_with_instruction}],
                    })
                    headers = {
                        'Authorization': f"Bearer {os.environ['API_KEY']}",
                        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                        'Content-Type': 'application/json',
                    }
                    conn.request("POST", "/v1/chat/completions", payload, headers)
                    res  = conn.getresponse()
                    data = json.loads(res.read().decode("utf-8"))
                    raw  = data['choices'][0]['message']['content']
                    all_samples.append(_extract_body(raw, config) if self._trim else raw)
                    break
                except Exception:
                    continue

        return all_samples

    def _draw_samples_hf(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        """Draw samples using HuggingFace Inference Router via the OpenAI-compatible SDK."""
        if not _OPENAI_SDK_AVAILABLE:
            raise ImportError("openai package is required: pip install openai")

        hf_base_url = getattr(config, 'hf_api_url', "https://router.huggingface.co/v1")
        hf_model    = getattr(config, 'hf_model', "meta-llama/Llama-3.1-8B-Instruct:cerebras")
        client      = get_hf_client(base_url=hf_base_url)
        prompt_with_instruction = '\n'.join([self._instruction_prompt, prompt])
        all_samples = []

        for _ in range(self._samples_per_prompt):
            while True:
                try:
                    response = client.chat.completions.create(
                        model=hf_model,
                        max_tokens=512,
                        messages=[{"role": "user", "content": prompt_with_instruction}],
                    )
                    raw = response.choices[0].message.content
                    all_samples.append(_extract_body(raw, config) if self._trim else raw)
                    break
                except Exception:
                    continue

        return all_samples

    def _draw_samples_sdk(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        """Draw samples using the OpenAI Python SDK."""
        if not _OPENAI_SDK_AVAILABLE:
            raise ImportError("openai package is required: pip install openai")

        client = get_openai_client()
        prompt_with_instruction = '\n'.join([self._instruction_prompt, prompt])
        all_samples = []

        for _ in range(self._samples_per_prompt):
            while True:
                try:
                    response = client.chat.completions.create(
                        model=config.api_model,
                        max_tokens=512,
                        messages=[{"role": "user", "content": prompt_with_instruction}],
                    )
                    raw = response.choices[0].message.content
                    all_samples.append(_extract_body(raw, config) if self._trim else raw)
                    break
                except Exception:
                    continue

        return all_samples

    def _draw_samples_anthropic(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        """Draw samples using the Anthropic Claude API."""
        client = get_anthropic_client()
        model  = getattr(config, 'anthropic_model', "claude-sonnet-4-6")
        prompt_with_instruction = '\n'.join([self._instruction_prompt, prompt])
        all_samples = []

        for _ in range(self._samples_per_prompt):
            while True:
                try:
                    response = client.messages.create(
                        model=model,
                        max_tokens=4096,
                        messages=[{"role": "user", "content": prompt_with_instruction}],
                    )
                    raw    = response.content[0].text
                    sample = _extract_body(raw, config) if self._trim else raw
                    all_samples.append(sample)
                    break
                except Exception as e:
                    print(f"[Anthropic] request failed: {e}, retrying...")
                    continue

        return all_samples

    def _do_request(self, content: str) -> str:
        content = content.strip('\n').strip()
        repeat_prompt: int = self._samples_per_prompt if self._batch_inference else 1

        data = {
            'prompt': content,
            'repeat_prompt': repeat_prompt,
            'params': {
                'do_sample': True,
                'temperature': None,
                'top_k': None,
                'top_p': None,
                'add_special_tokens': False,
                'skip_special_tokens': True,
            }
        }

        headers = {'Content-Type': 'application/json'}
        response = requests.post(self._url, data=json.dumps(data), headers=headers)

        if response.status_code == 200:
            response = response.json()["content"]
            return response if self._batch_inference else response[0]
