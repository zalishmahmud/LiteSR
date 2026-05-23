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

"""Configuration of a LiteSR experiment."""
from __future__ import annotations

import dataclasses
from typing import Type

from litesr.solver_agent import sampler
from litesr.solver_agent import evaluator


@dataclasses.dataclass(frozen=True)
class ExperienceBufferConfig:
    """Configures Experience Buffer parameters.
    
    Args:
        functions_per_prompt (int): Number of previous hypotheses to include in prompts
        num_islands (int): Number of islands in experience buffer for diversity
        reset_period (int): Seconds between weakest island resets
        cluster_sampling_temperature_init (float): Initial cluster softmax sampling temperature
        cluster_sampling_temperature_period (int): Period for temperature decay
    """
    functions_per_prompt: int = 2 
    num_islands: int = 10 
    reset_period: int = 30 * 60
    cluster_sampling_temperature_init: float = 0.1
    cluster_sampling_temperature_period: int = 30_000


@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for LiteSR experiments."""
    experience_buffer: ExperienceBufferConfig = dataclasses.field(default_factory=ExperienceBufferConfig)
    num_samplers: int = 1
    num_evaluators: int = 1
    samples_per_prompt: int = 4
    evaluate_timeout_seconds: int = 30
    use_api: bool = False
    use_hf_api: bool = False
    use_openai_sdk: bool = False
    use_ollama: bool = False
    use_anthropic: bool = True
    ollama_model: str = "mistral:latest"
    anthropic_model: str = "claude-sonnet-4-6"
    api_model: str = "gpt-4o"
    # ── RAG pipeline settings ─────────────────────────────────────────────────
    rag_use_ollama: bool = True
    rag_ollama_model: str = "mistral:latest"
    rag_use_equation_indexer: bool = False
    rag_toggle_patience: int = 20
    rag_improvement_threshold: float = 1e-4
    # ── EquationIndexer settings (vision OCR during indexing) ─────────────────
    eq_idx_use_ollama: bool = False
    eq_idx_vision_model: str = "llava:latest"
    eq_idx_anthropic_model: str = "claude-sonnet-4-6"


@dataclasses.dataclass()
class ClassConfig:
    llm_class: Type[sampler.LLM]
    sandbox_class: Type[evaluator.Sandbox]
