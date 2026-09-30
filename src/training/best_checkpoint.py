"""Keep immutable best adapters and atomically publish the current minimum."""
import json
import math
import shutil
import uuid
from datetime import datetime,timezone
from pathlib import Path
from .budget import atomic_json


def publish_best(run,checkpoint,step,validation_loss):
    run=Path(run).resolve();checkpoint=Path(checkpoint).resolve()
    if checkpoint.parent!=run:raise ValueError('Checkpoint must belong to this run')
    if not math.isfinite(validation_loss):raise ValueError('Validation loss must be finite')
    source=checkpoint/'adapter'
    if not (source/'adapter_model.safetensors').is_file():raise ValueError('Adapter weights missing')
    pointer=run/'best.json'
    if pointer.exists():
        previous=json.loads(pointer.read_text(encoding='utf-8'))
        if validation_loss>=previous['validation_loss']:return previous
    best_root=run/'best';best_root.mkdir(exist_ok=True)
    destination=best_root/f'step-{step:06d}'
    temporary=best_root/f'.step-{step:06d}-{uuid.uuid4().hex}.partial'
    if not destination.exists():
        temporary.mkdir(exist_ok=False)
        shutil.copytree(source,temporary/'adapter')
        atomic_json(temporary/'metadata.json',{'step':step,'validation_loss':validation_loss,
            'checkpoint':checkpoint.name,'metric':'token_weighted_validation_loss'})
        temporary.rename(destination)
    metadata={'step':step,'validation_loss':validation_loss,'checkpoint':checkpoint.name,
        'adapter':(destination/'adapter').relative_to(run).as_posix(),
        'metric':'token_weighted_validation_loss','updated_utc':datetime.now(timezone.utc).isoformat(),
        'resume_state':(checkpoint/'state.pt').relative_to(run).as_posix()}
    atomic_json(pointer,metadata)
    return metadata
