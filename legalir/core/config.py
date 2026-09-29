"""Cấu hình pipeline (Config) và sơ đồ đường dẫn artifact (Paths).

Tách từ ex9488.py; logic giữ nguyên."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    data: str = "/kaggle/input/datasets/hnhhctiliu/dsclegal26/LegalIR - Public Test"
    work_dir: str = "runs/legalir_bgem3"
    base_index: str = ""
    # No default: chunk_document's rewrite (article/Khoản-atomic packing,
    # 2026-08-19) produces different chunk_id boundaries than whatever
    # produced the old attached train_groups.jsonl/checkpoints, so silently
    # reusing either would score/mine against stale chunk offsets. Pass
    # --train-groups-file/--resume-artifacts-dir explicitly only once you
    # have artifacts regenerated under the new chunking.
    train_groups_file: str = ""
    resume_artifacts_dir: str = ""
    reuse_trained_embeddings: bool = True
    # Test-only is the safe default when this whole file is pasted into a
    # Kaggle cell. Training remains available via --stage retrain-c-r2.
    stage: str = "loadmodel"
    cr2_source: str = "/kaggle/input/datasets/tntduong/legali26/runs/legalir_bgem3"
    # ``loadmodel`` reads an arbitrary raw LegalIR query file without labels,
    # loads the completed C/R2 run, and writes predictions without training.
    test_file: str = (
        "/kaggle/input/datasets/nguyendinhhaduong/uitdscprivate26/"
        "private-official.json"
    )
    loadmodel_output_dir: str = ""
    loadmodel_model_dir: str = (
        "/kaggle/input/notebooks/nguyendinhhaduong/"
        "fork-of-reranker-dsc-96"
    )
    loadmodel_weight_set: str = "all"
    loadmodel_dry_run: bool = False
    cr2_prepare_only: bool = False
    cr2_eval_only: bool = False
    cr2_remine_c: bool = True
    cr2_c_evidence: int = 4
    cr2_c_negatives: int = 24
    cr2_c_positive_shortlist: int = 32
    cr2_c_learning_rate: float = 2e-6
    cr2_c_replay_fraction: float = 0.25
    cr2_c_epochs: int = 3
    cr2_c_steps: int = 1000
    cr2_r2_epochs: int = 5
    cr2_r2_steps: int = 2000
    cr2_r2_lr_decay: bool = True
    cr2_r2_boundary_weight: float = 0.0
    cr2_remine_r2: bool = True
    cr2_r2_refresh_steps: int = 250
    cr2_learning_rate: float = 2e-6
    cr2_negatives: int = 16
    seed: int = 2026
    eval_size: int = 1000
    embedding_model: str = "/kaggle/input/models/andreasbis/baai-bge-m3/transformers/default/1"
    # Root containing config.json/tokenizer/model weights. The nested
    # .cache/huggingface/.gitignore file is download metadata, not the model.
    reranker_model: str = (
        "/kaggle/input/notebooks/nguyendinhhaduong/bge-m3-reranker/"
        "models/bge-reranker-v2-m3"
    )
    cache_dir: str = ""
    # Chunk boundaries always align with an Điều (or, when an Điều exceeds
    # this, a Khoản) boundary -- see chunk_document. 1000 was chosen over
    # 1800 from a 300-document sample of the real corpus: median Điều length
    # is ~900 chars, and article-atomic packing at 1000 puts ~70% of chunks
    # at exactly one whole Điều vs. 25% under the old raw-block greedy pack
    # at 1800 (scratchpad chunking analysis, 2026-08-19). No char-count
    # overlap is added between chunks any more: since chunks never split an
    # Điều/Khoản mid-sentence, a fixed-length tail would only reintroduce
    # the boundary-crossing noise this redesign removes.
    # A 1,400-character soft target keeps most medium-sized legal articles
    # intact while remaining far below BGE-M3's token window.  The chunker
    # still splits at Dieu/Khoan boundaries, so this adds useful local context
    # without returning to arbitrary fixed-width chunks.
    chunk_chars: int = 1400
    max_chunks_per_doc: int = 0
    embedding_max_length: int = 1024
    embedding_dim: int = 1024
    embed_batch_size: int = 16
    query_batch_size: int = 64
    # Every A/B epoch contains 1,500 micro-steps to keep training practical.
    # micro-steps. Early stopping still retains the best held-out Hit@5 epoch.
    embed_epochs: int = 6
    embed_steps_per_epoch: int = 1500
    embed_max_steps: int = 9000
    embed_min_epochs: int = 2
    embed_early_stopping_patience: int = 2
    embed_train_batch_size: int = 4
    embed_grad_accum: int = 8
    embed_learning_rate: float = 1e-5
    embed_stage2_learning_rate: float = 5e-6
    # B continues A with a lower LR and teacher/lexical hard negatives. At
    # inference it is one member of the union, not the sole pool retriever.
    embed_stage2_epochs: int = 6
    embed_stage2_steps_per_epoch: int = 1500
    embed_stage2_max_steps: int = 9000
    embed_stage2_min_epochs: int = 2
    # C is intentionally independent from A/B so the final ensemble retains
    # three genuinely different embedding views.
    embed_stage3_epochs: int = 5
    embed_stage3_steps_per_epoch: int = 5000
    embed_stage3_min_epochs: int = 2
    embed_stage3_learning_rate: float = 5e-6
    # C uses the same compact rank as A/B; its diversity comes from curriculum
    # and independent initialization rather than excessive adapter capacity.
    embed_stage3_hard_negatives: int = 12
    # C sees every A/B boundary failure plus a deterministic rehearsal sample.
    # Rehearsal prevents the specialist from forgetting ordinary queries.
    embed_stage3_rehearsal_ratio: float = 0.35
    # A/B select C's curriculum and negatives, but their scores are never
    # distilled. This preserves C as a third representation view.
    embed_stage3_kd_weight: float = 0.0
    hard_negatives: int = 8
    # B mines negatives from its own candidate pool; too few negatives left
    # most near-miss distractors unseen during stage-2 training. Fourteen
    # negatives (still well under candidate_docs, so the hardest/lexical/random
    # stratified split in teacher_hard_negative_rows keeps a real pool to
    # sample from) covers more of the pool's hard cases without turning most of
    # the pool into per-group negatives.
    embed_stage2_hard_negatives: int = 14
    # Legal text repeats near-identical boilerplate (same điều/khoản numbers,
    # same statute name) across unrelated articles, so a false lexical
    # positive is the classic legal-IR failure mode: the reranker-teacher
    # score alone doesn't guarantee those token-overlap decoys get selected
    # as negatives. This carves out a dedicated stratum (see
    # teacher_hard_negative_rows) of the highest lexical-overlap non-gold
    # candidates the teacher-hardest stratum didn't already pick, so B is
    # explicitly trained to separate lexical near-duplicates in embedding
    # space rather than relying on the teacher to have already ranked them
    # low.
    embed_stage2_lexical_negative_ratio: float = 0.20
    embedding_temperature: float = 0.02
    embedding_kd_weight: float = 0.30
    train_groups: int = 0
    dense_chunk_k: int = 450
    # A 60-document union covers complementary A/B/C near misses before the
    # cross-encoders optimize the top-five boundary.
    candidate_docs: int = 60
    # Reserve a small number of documents from each retriever before filling
    # the rest by calibrated RRF. This is what makes A/B/C a real retrieval
    # ensemble even if calibration gives one member a near-zero weight.
    dense_member_quota: int = 5
    chunks_per_doc: int = 4
    ensemble_stage1_weight: float = 0.75
    # Score the entire candidate pool, not merely its dense top five.
    # Score that whole high-recall pool so the reranker can actually promote a
    # relevant item from below dense rank five instead of merely reordering it.
    rerank_top_k: int = 60
    reranker_max_length: int = 1536
    rerank_batch_size: int = 16
    # Zero means use every non-holdout group. Each epoch samples a fresh 3,000
    # groups, rather than permanently discarding roughly half the training set.
    rerank_train_groups: int = 0
    rerank_mining_docs: int = 50
    rerank_hard_negatives: int = 15
    rerank_epochs: int = 4
    rerank_steps_per_epoch: int = 6000
    rerank_max_steps: int = 24000
    rerank_min_epochs: int = 2
    rerank_early_stopping_patience: int = 2
    rerank_grad_accum: int = 8
    rerank_learning_rate: float = 2e-5
    rerank_pairwise_weight: float = 0.40
    rerank_margin: float = 0.20
    rerank_stage2_epochs: int = 4
    rerank_stage2_min_epochs: int = 2
    rerank_stage2_steps_per_epoch: int = 5000
    rerank_stage2_learning_rate: float = 5e-6
    rerank_stage2_hard_negatives: int = 15
    rerank_stage2_margin: float = 0.15
    # Train against the same multi-chunk max used at inference without making
    # each micro-step prohibitively large on the 0.6B cross-encoder.
    rerank_train_chunks_per_doc: int = 2
    # Checkpoint selection must measure the competition objective, not the
    # one-positive mining-group proxy. Five hundred queries gives ~0.002
    # Recall resolution while avoiding ~100 minutes of extra validation seen
    # when scoring all 1,000 queries after every R1/R2 epoch.
    rerank_checkpoint_eval_queries: int = 500
    # Forward-only checkpoint evaluation can safely use a larger batch than
    # gradient-bearing training. Keep the training/mining batch at 16.
    rerank_checkpoint_batch_size: int = 32
    # R2 is a boundary repair stage: retain every actual top-5 failure and a
    # deterministic slice of already-correct queries to prevent forgetting.
    rerank_stage2_rehearsal_ratio: float = 0.25
    calibration_size: int = 500
    target_recall_at_5: float = 0.96
    # How many of the best calibration weight vectors to persist to
    # calibration.json / print at evaluate() end for visibility. Only the
    # single best (rank 1) is actually used to build the submission.
    top_configs_logged: int = 5
    bm25_enabled: bool = True
    # A/B/C/R1/R2 share rank 64; diversity comes from data/objective/init.
    lora_rank: int = 64
    embed_stage3_lora_rank: int = 64
    # rsLoRA stabilizes high-rank adapters. With alpha=2*sqrt(rank), its
    # alpha/sqrt(rank) scale remains equal to the old alpha/rank=2 scale.
    lora_use_rslora: bool = True
    # Adapt attention plus output/FFN projections. "attention" remains a
    # lower-memory fallback selectable from the command line.
    lora_target_profile: str = "attention_ffn"
    warmup_ratio: float = 0.10
    no_train_embedding: bool = False
    no_train_reranker: bool = False
    rebuild: bool = False
    self_test: bool = False


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def raw(self) -> Path: return self.root / "raw"
    @property
    def prepared(self) -> Path: return self.root / "prepared"
    @property
    def train(self) -> Path: return self.prepared / "train.jsonl"
    @property
    def validation(self) -> Path: return self.prepared / "validation_1000.jsonl"
    @property
    def test(self) -> Path: return self.prepared / "public_test.jsonl"
    @property
    def documents(self) -> Path: return self.prepared / "documents.jsonl"
    @property
    def chunks(self) -> Path: return self.prepared / "chunks.jsonl"
    @property
    def chunk_meta(self) -> Path: return self.prepared / "chunk_meta.json"
    @property
    def base_index(self) -> Path: return self.root / "dense_base.faiss"
    @property
    def final_index(self) -> Path: return self.root / "dense_finetuned.faiss"
    @property
    def embed_adapter(self) -> Path: return self.root / "embedding_adapter"
    @property
    def second_index(self) -> Path: return self.root / "dense_finetuned_stage2.faiss"
    @property
    def second_embed_adapter(self) -> Path: return self.root / "embedding_adapter_stage2"
    @property
    def third_index(self) -> Path: return self.root / "dense_finetuned_stage3.faiss"
    @property
    def third_embed_adapter(self) -> Path: return self.root / "embedding_adapter_stage3"
    @property
    def dense_calibration(self) -> Path: return self.root / "dense_ensemble_calibration.json"
    @property
    def rerank_adapter(self) -> Path: return self.root / "bge_reranker_v2_m3_adapter"
    @property
    def second_rerank_adapter(self) -> Path: return self.root / "bge_reranker_v2_m3_adapter_stage2"
    @property
    def groups(self) -> Path: return self.root / "train_groups.jsonl"
    @property
    def second_stage_groups(self) -> Path: return self.root / "reranker_groups_bgem3.jsonl"
    @property
    def reranker_groups(self) -> Path: return self.root / "reranker_groups_ensemble.jsonl"
    @property
    def third_stage_groups(self) -> Path: return self.root / "embedding_groups_stage3.jsonl"
    @property
    def second_reranker_groups(self) -> Path: return self.root / "reranker_groups_stage2.jsonl"
    @property
    def reranker_checkpoint_pool(self) -> Path: return self.root / "reranker_checkpoint_pool.jsonl"
    @property
    def reranker_checkpoint_pool_meta(self) -> Path: return self.root / "reranker_checkpoint_pool_meta.json"
    @property
    def reranker_checkpoint_r1_scores(self) -> Path: return self.root / "reranker_checkpoint_r1_scores.jsonl"
    @property
    def reranker_checkpoint_r1_scores_meta(self) -> Path: return self.root / "reranker_checkpoint_r1_scores_meta.json"
    @property
    def calibration(self) -> Path: return self.root / "calibration.json"
    @property
    def submission_json(self) -> Path: return self.root / "submission.json"
    @property
    def submission_zip(self) -> Path: return self.root / "submission.zip"
    @property
    def public_debug_jsonl(self) -> Path: return self.root / "public_test_top10_debug.jsonl"
    @property
    def public_debug_zip(self) -> Path: return self.root / "public_test_top10_debug.zip"
    @property
    def eval_submission_json(self) -> Path: return self.root / "eval_submission.json"
    @property
    def eval_wrong_predictions(self) -> Path: return self.root / "eval_wrong_predictions.jsonl"
    @property
    def baseline_submission_zip(self) -> Path: return self.root / "submission_baseline_09028.zip"
    @property
    def ablation_report(self) -> Path: return self.root / "ablation_report.json"
