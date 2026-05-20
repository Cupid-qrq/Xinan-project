# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import pathlib
from typing import Optional

from pyrit.common.apply_defaults import REQUIRED_VALUE, apply_defaults
from pyrit.common.path import CONVERTER_SEED_PROMPT_PATH
from pyrit.models import SeedPrompt
from pyrit.prompt_converter.llm_generic_text_converter import LLMGenericTextConverter
from pyrit.prompt_target import PromptChatTarget

logger = logging.getLogger(__name__)


class ClassicalChineseConverter(LLMGenericTextConverter):
    """
    Rewrites prompts into Classical Chinese style (Wenyanwen) using an LLM.

    An existing ``PromptChatTarget`` is used to perform the conversion.
    """

    @apply_defaults
    def __init__(
        self,
        *,
        converter_target: PromptChatTarget = REQUIRED_VALUE,  # type: ignore[assignment]
        prompt_template: Optional[SeedPrompt] = None,
    ) -> None:
        """
        Initialize the converter with a target and optional prompt template.

        Args:
            converter_target (PromptChatTarget): The target chat support for conversion.
            prompt_template (SeedPrompt, Optional): Optional custom prompt template.
        """
        prompt_template = (
            prompt_template
            if prompt_template
            else SeedPrompt.from_yaml_file(pathlib.Path(CONVERTER_SEED_PROMPT_PATH) / "classical_chinese_converter.yaml")
        )

        super().__init__(
            converter_target=converter_target,
            system_prompt_template=prompt_template,
        )
