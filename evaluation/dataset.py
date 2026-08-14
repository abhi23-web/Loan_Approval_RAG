"""Golden dataset loading and validation.

The dataset is the regression suite. If it can be loaded with a typo in a field
name, a case silently stops being graded and the suite quietly gets weaker, so
the schema forbids unknown keys and every case is validated on load.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import PROJECT_ROOT
from app.core.exceptions import EvaluationError
from app.core.logging_config import get_logger

_logger = get_logger(__name__)

DEFAULT_DATASET_PATH = PROJECT_ROOT / "evaluation" / "golden_dataset.json"


class GoldenCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    question: str
    question_type: str
    category: str
    expected_answer: str
    acceptable_answer_variations: list[str] = Field(default_factory=list)
    expected_answer_patterns: list[str] = Field(default_factory=list)
    forbidden_answer_patterns: list[str] = Field(default_factory=list)
    expected_source: str
    expected_document_version: int
    expected_context_keywords: list[str]
    # Present only on cases that deliberately query a superseded version.
    version_override: dict[str, int] | None = None
    policy_reference: str = ""
    why_it_matters: str = ""

    @property
    def is_historical_lookup(self) -> bool:
        return self.version_override is not None


class GradingPrecondition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_active_version: dict[str, int]
    how_to_reach_it: list[str] = Field(default_factory=list)
    note: str = ""


class ReproducibilityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repeat_runs: int = Field(default=3, ge=1, le=20)
    measures: list[str] = Field(default_factory=list)


class GoldenDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    dataset_version: str
    created_on: str
    description: str
    why_this_corpus: str = ""
    grading_precondition: GradingPrecondition
    categories: dict[str, str]
    reproducibility: ReproducibilityConfig
    cases: list[GoldenCase]

    @classmethod
    def load(cls, dataset_path: Path | None = None) -> GoldenDataset:
        path = dataset_path or DEFAULT_DATASET_PATH
        if not path.exists():
            raise EvaluationError(f"golden dataset not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            raw_dataset = json.load(handle)

        dataset = cls.model_validate(raw_dataset)
        case_ids = [case.case_id for case in dataset.cases]
        if len(set(case_ids)) != len(case_ids):
            raise EvaluationError("duplicate case_id in the golden dataset")

        _logger.info(
            "loaded golden dataset '%s' v%s: %d case(s), %d repeat run(s)",
            dataset.dataset_id,
            dataset.dataset_version,
            len(dataset.cases),
            dataset.reproducibility.repeat_runs,
        )
        return dataset

    def case(self, case_id: str) -> GoldenCase:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise EvaluationError(f"unknown case_id '{case_id}'")
