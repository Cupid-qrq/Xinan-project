# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import pathlib
from typing import Literal, Optional, get_args

from pyrit.common.apply_defaults import REQUIRED_VALUE, apply_defaults
from pyrit.common.path import CONVERTER_SEED_PROMPT_PATH
from pyrit.identifiers import ComponentIdentifier
from pyrit.models import SeedPrompt
from pyrit.prompt_converter.llm_generic_text_converter import LLMGenericTextConverter
from pyrit.prompt_target import PromptChatTarget

logger = logging.getLogger(__name__)


# Supported translation modes
ChineseTranslationMode = Literal["academic", "technical", "smiles", "math", "research", "reaction", "combined"]
CHINESE_TRANSLATION_MODES = set(get_args(ChineseTranslationMode))

# Mapping from mode to Chinese YAML file name
CHINESE_MODE_YAML_FILES: dict[str, str] = {
    "academic": "chinese_academic_science_converter.yaml",
    "technical": "chinese_technical_science_converter.yaml",
    "smiles": "chinese_smiles_science_converter.yaml",
    "math": "chinese_math_science_converter.yaml",
    "research": "chinese_research_science_converter.yaml",
    "reaction": "chinese_reaction_science_converter.yaml",
    "combined": "chinese_combined_science_converter.yaml",
}


class ChineseScientificTranslationConverter(LLMGenericTextConverter):
    """
    Uses an LLM to rewrite prompts into Chinese scientific/technical style.

    This converter keeps the same mode behavior as ScientificTranslationConverter,
    but applies Chinese prompt templates.
    """

    @apply_defaults
    def __init__(
        self,
        *,
        converter_target: PromptChatTarget = REQUIRED_VALUE,  # type: ignore[assignment]
        mode: str = "combined",
        prompt_template: Optional[SeedPrompt] = None,
    ) -> None:
        """
        Initialize the Chinese scientific translation converter.

        Args:
            converter_target (PromptChatTarget): The LLM target to perform conversion.
            mode (str): Conversion mode, one of built-in options:
                academic, technical, smiles, math, research, reaction, combined.
            prompt_template (SeedPrompt, Optional): Custom prompt template.
                Required if using a custom mode not in the built-in list.

        Raises:
            ValueError: If custom mode is used without prompt_template.
        """
        if prompt_template is not None:
            resolved_template = prompt_template
        elif mode in CHINESE_TRANSLATION_MODES:
            yaml_file = CHINESE_MODE_YAML_FILES[mode]
            resolved_template = SeedPrompt.from_yaml_file(pathlib.Path(CONVERTER_SEED_PROMPT_PATH) / yaml_file)
        else:
            raise ValueError(
                f"Custom mode '{mode}' requires a prompt_template. "
                f"Either use a built-in mode from {sorted(CHINESE_TRANSLATION_MODES)} "
                "or provide a prompt_template."
            )

        super().__init__(
            converter_target=converter_target,
            system_prompt_template=resolved_template,
        )
        self._mode = mode

    def _build_identifier(self) -> ComponentIdentifier:
        return self._create_identifier(
            params={
                "mode": self._mode,
            },
            children={"converter_target": self._converter_target.get_identifier()},
        )
