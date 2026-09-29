"""Namespace phẳng chứa toàn bộ API pipeline (thay cho `import ex9488 as api`).

Lớp `Experiment` nhận module này làm `api` và gọi `api.<hàm>`; vì vậy thứ tự
import dưới đây đi theo thứ tự phụ thuộc giữa các module."""

from __future__ import annotations

from legalir._deps import *  # noqa: F401,F403
from legalir.core.constants import *  # noqa: F401,F403
from legalir.core.config import *  # noqa: F401,F403
from legalir.core.utils import *  # noqa: F401,F403
from legalir.core.artifacts import *  # noqa: F401,F403
from legalir.models.lora import *  # noqa: F401,F403
from legalir.data.text import *  # noqa: F401,F403
from legalir.data.chunking import *  # noqa: F401,F403
from legalir.data.loading import *  # noqa: F401,F403
from legalir.models.dense_encoder import *  # noqa: F401,F403
from legalir.data.chunk_store import *  # noqa: F401,F403
from legalir.retrieval.dense_index import *  # noqa: F401,F403
from legalir.retrieval.dense_fusion import *  # noqa: F401,F403
from legalir.training.negatives import *  # noqa: F401,F403
from legalir.retrieval.fused_order import *  # noqa: F401,F403
from legalir.models.reranker import *  # noqa: F401,F403
from legalir.training.mining import *  # noqa: F401,F403
from legalir.training.mining_repair import *  # noqa: F401,F403
from legalir.rerank.cache import *  # noqa: F401,F403
from legalir.rerank.candidates import *  # noqa: F401,F403
from legalir.evaluation.metrics import *  # noqa: F401,F403
from legalir.training.embedder import *  # noqa: F401,F403
from legalir.evaluation.dense_eval import *  # noqa: F401,F403
from legalir.training.reranker_checkpoints import *  # noqa: F401,F403
from legalir.training.reranker import *  # noqa: F401,F403
from legalir.rerank.repair_fusion import *  # noqa: F401,F403
from legalir.rerank.meta_ranker import *  # noqa: F401,F403
from legalir.rerank.continuous_opt import *  # noqa: F401,F403
from legalir.evaluation.evaluate import *  # noqa: F401,F403
from legalir.pipeline.submission import *  # noqa: F401,F403
from legalir.pipeline.validation import *  # noqa: F401,F403
from legalir.selftest import *  # noqa: F401,F403
from legalir.experiment.constants import *  # noqa: F401,F403
from legalir.experiment.utils import *  # noqa: F401,F403
from legalir.experiment.losses import *  # noqa: F401,F403
from legalir.experiment.c_helpers import *  # noqa: F401,F403
from legalir.rerank.fusion import *  # noqa: F401,F403
from legalir.rerank.weight_search import *  # noqa: F401,F403
from legalir.experiment.base import *  # noqa: F401,F403
from legalir.experiment.pools import *  # noqa: F401,F403
from legalir.experiment.mining import *  # noqa: F401,F403
from legalir.experiment.train_c import *  # noqa: F401,F403
from legalir.experiment.train_r2 import *  # noqa: F401,F403
from legalir.experiment.export import *  # noqa: F401,F403
from legalir.experiment.experiment import *  # noqa: F401,F403
from legalir.experiment.run import *  # noqa: F401,F403
from legalir.pipeline.loadmodel import *  # noqa: F401,F403
from legalir.cli import *  # noqa: F401,F403
from legalir.pipeline.loadmodel import (  # noqa: F401  (tên bắt đầu bằng _ không đi qua import *)
    _loadmodel_config, _loadmodel_experiment, _loadmodel_output_root, _loadmodel_require_dir,
    _loadmodel_require_file, _loadmodel_validate_artifacts, _loadmodel_weights,
    _resolve_loadmodel_model_root, _resolve_loadmodel_test_file, _write_loadmodel_outputs,
)
from legalir.rerank.continuous_opt import (  # noqa: F401
    _repair_config_from_vector, _softmax, _softplus, _vector_from_repair_config,
)
