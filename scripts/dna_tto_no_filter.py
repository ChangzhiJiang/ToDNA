"""Shared DNA_TTO selection policy: validate/deduplicate, but do not reject by GC/run length."""
from __future__ import annotations

from dataclasses import replace
import time
from typing import Sequence

import torch

from evo2_nabla_dna import CandidateRecord, Evo2TestTimeDesigner, SearchConfig
from evo2_nabla_dna.dto import RewardFn, validate_sequence


def sequence_metrics(sequence: str, evo2_nll: float) -> dict[str, float]:
    sequence = validate_sequence(sequence)
    gc_fraction = sum(base in "GC" for base in sequence) / len(sequence)
    max_run = current_run = 1
    for previous, current in zip(sequence, sequence[1:]):
        current_run = current_run + 1 if current == previous else 1
        max_run = max(max_run, current_run)
    return {
        "gc_fraction": float(gc_fraction),
        "max_homopolymer": float(max_run),
        "evo2_nll": float(evo2_nll),
    }


class NoSequenceConstraintDesigner(Evo2TestTimeDesigner):
    """Retain projection/edit controls while disabling GC/homopolymer rejection."""

    def search(
        self,
        prompt: str,
        length: int,
        reward_fn: RewardFn,
        config: SearchConfig | None = None,
        initial_sequences: Sequence[str] | None = None,
    ) -> list[CandidateRecord]:
        config = config or SearchConfig()
        if config.constraints is not None:
            raise ValueError("DNA_TTO requires constraints=None")
        if initial_sequences is None:
            raise ValueError("formal DNA_TTO runs require locked initial sequences")
        sequences = [validate_sequence(sequence) for sequence in initial_sequences]
        if not sequences or any(len(sequence) != length for sequence in sequences):
            raise ValueError("all initial sequences must match the requested length")

        records: list[CandidateRecord] = []
        seen: set[str] = set()
        duplicates = 0
        device = torch.device(self.dto.device)
        for candidate_index, initial_sequence in enumerate(sequences, start=1):
            dto_config = config.dto
            if dto_config.initial_logit_noise_std:
                base_seed = dto_config.initial_logit_noise_seed
                if base_seed is None:
                    raise ValueError("trajectory seed base is required for logit noise")
                dto_config = replace(
                    dto_config,
                    initial_logit_noise_seed=(base_seed + candidate_index - 1)
                    % (2**63 - 1),
                )
            started = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            result = self.dto.optimize(initial_sequence, reward_fn, dto_config)
            elapsed = time.perf_counter() - started
            peak_memory = 0
            if device.type == "cuda":
                peak_memory = int(torch.cuda.max_memory_allocated(device))
            optimized_sequence = validate_sequence(result.optimized_sequence)
            if optimized_sequence in seen:
                duplicates += 1
                continue
            seen.add(optimized_sequence)
            log_probability = self.dto.score_sequence(optimized_sequence)
            evo2_nll = -log_probability
            records.append(CandidateRecord(
                initial_sequence=initial_sequence,
                optimized_sequence=optimized_sequence,
                initial_reward=result.initial_reward,
                optimized_reward=result.final_reward,
                evo2_log_probability=log_probability,
                evo2_nll=evo2_nll,
                ranking_score=result.final_reward,
                accepted=True,
                constraint_metrics=sequence_metrics(optimized_sequence, evo2_nll),
                hamming_edits=result.hamming_edits,
                trust_kl=result.final_trust_kl,
                expected_edit_fraction=result.final_expected_edit_fraction,
                history=result.history,
                design_oracle_calls=int(dto_config.steps),
                wall_clock_seconds=float(elapsed),
                peak_gpu_memory_bytes=peak_memory,
            ))
            if candidate_index % 32 == 0 or candidate_index == len(sequences):
                print(
                    f"SEARCH_PROGRESS attempted={candidate_index}/{len(sequences)} "
                    f"unique={len(records)} accepted={len(records)}",
                    flush=True,
                )
        records.sort(key=lambda record: record.ranking_score, reverse=True)
        self.last_search_summary = {
            "attempted": len(sequences),
            "unique_decoded": len(records),
            "unique_projected": len(records),
            "accepted": len(records),
            "duplicates": duplicates,
            "rejection_counts": {},
            "hard_constraints_enabled": False,
        }
        return records[: config.top_k]
