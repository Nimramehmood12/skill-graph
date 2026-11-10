"""Pydantic v2 schemas — strict typed output contracts."""
from pydantic import BaseModel, Field
from typing import List, Dict


class SkillGap(BaseModel):
    skill: str
    prereqs_met: List[str] = Field(default_factory=list)
    missing_prereqs: List[str] = Field(default_factory=list)
    bridge_path: List[str] = Field(default_factory=list)
    estimated_hours: int = 0
    readiness: float = Field(0.0, ge=0.0, le=1.0)


class RoadmapStep(BaseModel):
    order: int
    skill: str
    prerequisite_chain: List[str]
    estimated_hours: int


class AnalysisResult(BaseModel):
    fit_score: float = Field(..., ge=0.0, le=1.0)
    matched_skills: List[str] = Field(default_factory=list)
    transferable_strengths: List[str] = Field(default_factory=list)
    critical_blockers: List[str] = Field(default_factory=list)
    gaps: List[SkillGap] = Field(default_factory=list)
    roadmap: List[RoadmapStep] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict)