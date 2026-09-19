"""Functional CPU coverage for ``--resume`` training progress.

The tests use real Accelerate prepared objects and the real
``NetworkTrainer._run_training_loop``. The model and batch processor are tiny,
so no model or dataset downloads are involved.
"""

import argparse
import asyncio
import copy
import random
import os
import socket
import subprocess
import sys
from datetime import timedelta
from multiprocessing import Value
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Dataset, Sampler

from musubi_tuner.dataset.bucket import BucketBatchManager
from musubi_tuner.dataset.image_video_dataset import BaseDataset, ImageDataset
from musubi_tuner.training import trainer_base as trainer_base_module
from musubi_tuner.training.trainer_base import NetworkTrainer
from musubi_tuner.utils import train_utils


class OrderedToyDataset(Dataset):
    def __init__(self, length=5):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        value = float(index)
        return {"x": torch.tensor([value]), "target": torch.tensor([2.0 * value + 1.0]), "latents": torch.tensor([value])}


class BucketedToyDataset(OrderedToyDataset):
    """Exercise Musubi's real lazy, epoch-dependent in-place bucket shuffling."""

    def __init__(self, length, shared_epoch):
        super().__init__(length)
        self.shared_epoch, self.current_epoch, self.seed = shared_epoch, 0, 4321
        self.batch_manager = BucketBatchManager({(1, 1): list(range(length))}, batch_size=1)

    shuffle_buckets = ImageDataset.shuffle_buckets

    def __getitem__(self, index):
        BaseDataset.__getitem__(self, index)
        resolution, offset = self.batch_manager.bucket_batch_indices[index]
        return super().__getitem__(self.batch_manager.buckets[resolution][offset])


class RecordingSampler(Sampler):
    def __init__(self, length):
        self.length = length
        self.epochs = []

    def __iter__(self):
        return iter(range(self.length))

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


class ToyDatasetInfo:
    batch_size = 1

    def get_metadata(self):
        return {"name": "ordered-toy"}


class ToyDatasetGroup:
    def __init__(self, length):
        self.num_train_items = length
        self.datasets = [ToyDatasetInfo()]


class ToyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

    def on_epoch_start(self, transformer):
        del transformer

    def on_step_start(self):
        pass

    def prepare_grad_etc(self, transformer):
        pass

    def get_trainable_params(self):
        return list(self.parameters())

    def save_weights(self, path, dtype, metadata):
        del dtype, metadata
        torch.save(self.state_dict(), path)


class ToyTrainer(NetworkTrainer):
    @property
    def architecture(self):
        return "hv"

    @property
    def architecture_full_name(self):
        return "HunyuanVideo toy"

    def __init__(self):
        super().__init__()
        self.seen = []
        self.noises = []
        self.noisy_loss = False
        self.sample_calls = []

    def scale_shift_latents(self, latents):
        return latents

    def process_batch(
        self,
        args,
        accelerator,
        transformer,
        network,
        batch,
        latents,
        noise,
        noise_scheduler,
        dit_dtype,
        network_dtype,
        vae,
        global_step,
    ):
        del args, accelerator, transformer, latents, noise_scheduler, dit_dtype, network_dtype, vae, global_step
        self.seen.append(int(batch["x"][0, 0].item()))
        self.noises.append(noise.detach().cpu().clone())
        target = batch["target"] + noise + random.random() + np.random.random() if self.noisy_loss else batch["target"]
        return torch.nn.functional.mse_loss(network(batch["x"]), target), {}

    def sample_images(self, accelerator, args, epoch, steps, vae, transformer, sample_parameters, dit_dtype):
        del accelerator, args, vae, transformer, sample_parameters, dit_dtype
        self.sample_calls.append((epoch, steps))


def _args(output_dir, *, max_steps, accumulation=1, resume=None, save_steps=None, save_epochs=None, sample_at_first=False):
    """Build the broad Namespace surface consumed by the base loop."""
    return argparse.Namespace(
        resume=None if resume is None else str(resume),
        resume_from_huggingface=False,
        huggingface_token=None,
        max_train_steps=max_steps,
        max_train_epochs=None,
        gradient_accumulation_steps=accumulation,
        output_dir=str(output_dir),
        output_name="toy",
        network_args=None,
        network_module="toy",
        network_dim=1,
        network_alpha=1.0,
        network_dropout=0.0,
        network_weights=None,
        learning_rate=0.05,
        optimizer_type="AdamW",
        optimizer_args=None,
        lr_scheduler="constant",
        lr_scheduler_type=None,
        lr_scheduler_args=None,
        lr_warmup_steps=0,
        lr_decay_steps=None,
        lr_scheduler_num_cycles=1,
        lr_scheduler_power=1.0,
        lr_scheduler_timescale=None,
        lr_scheduler_min_lr_ratio=None,
        mixed_precision="no",
        full_fp16=False,
        full_bf16=False,
        weighting_scheme="none",
        logit_mean=0.0,
        logit_std=1.0,
        mode_scale=1.29,
        guidance_scale=1.0,
        timestep_sampling="uniform",
        sigmoid_scale=1.0,
        discrete_flow_shift=1.0,
        gradient_checkpointing=False,
        gradient_checkpointing_cpu_offload=False,
        seed=1234,
        training_comment=None,
        max_grad_norm=0.0,
        fp8_base=False,
        fp8_scaled=False,
        dit=None,
        vae=None,
        no_metadata=False,
        metadata_title=None,
        metadata_reso=None,
        metadata_author=None,
        metadata_description=None,
        metadata_license=None,
        metadata_tags=None,
        metadata_arch=None,
        huggingface_repo_id=None,
        save_precision=None,
        save_every_n_steps=save_steps,
        save_every_n_epochs=save_epochs,
        save_last_n_steps=None,
        save_last_n_steps_state=None,
        save_last_n_epochs=None,
        save_last_n_epochs_state=None,
        save_state=True,
        save_state_on_train_end=True,
        save_state_to_huggingface=False,
        sample_prompts=None,
        sample_at_first=sample_at_first,
        sample_every_n_steps=None,
        sample_every_n_epochs=None,
        scale_weight_norms=False,
        log_grad_metrics=False,
        log_config=False,
        log_tracker_config=None,
        log_tracker_name=None,
        wandb_run_name=None,
        compile=False,
        blocks_to_swap=0,
        split_attn=False,
        sdpa=True,
        flash_attn=False,
        sage_attn=False,
        xformers=False,
        flash3=False,
        dit_dtype=None,
        vae_dtype=None,
        min_timestep=None,
        max_timestep=None,
        num_timestep_buckets=None,
    )


def _cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def _assert_nested_equal(left, right):
    assert type(left) is type(right)
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def _training_state_dict(trainer):
    state = trainer._training_state
    if state is None:
        return None
    return _cpu_copy(state.state_dict() if hasattr(state, "state_dict") else state)


def _state_file(state_dir):
    return Path(state_dir) / "trainer_state.pt"


def _read_state(state_dir):
    path = _state_file(state_dir)
    assert path.exists(), f"missing resume sidecar: {path}"
    return torch.load(path, map_location="cpu", weights_only=True)


def _state_version_key(state):
    return "version" if "version" in state else "schema_version"


def _config_length(config):
    for key in ("prepared_len", "length", "len", "num_batches"):
        if key in config:
            return config[key]
    raise AssertionError(f"no prepared dataloader length in {config}")


def _run_training(
    output_dir,
    *,
    max_steps,
    data_length=5,
    accumulation=1,
    optimizer_kind="adamw",
    resume=None,
    save_steps=None,
    save_epochs=None,
    sample_at_first=False,
    shuffle=False,
    noisy_loss=False,
    num_workers=0,
    bucketed=False,
    persistent_workers=False,
    distributed=False,
):
    """Run the actual preparation and base loop with tiny CPU models."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(1234)
    trainer = ToyTrainer()
    trainer.noisy_loss = noisy_loss
    network = ToyNetwork()
    transformer = torch.nn.Linear(1, 1)
    parameters = list(network.parameters())
    if optimizer_kind == "sgd":
        optimizer, optimizer_name = torch.optim.SGD(parameters, lr=0.05, momentum=0.9), "torch.optim.SGD"
    elif optimizer_kind == "schedulefree":
        schedulefree = pytest.importorskip("schedulefree")
        optimizer = schedulefree.AdamWScheduleFree(parameters, lr=0.05, foreach=False)
        optimizer_name = "schedulefree.AdamWScheduleFree"
    else:
        optimizer, optimizer_name = torch.optim.AdamW(parameters, lr=0.05), "torch.optim.AdamW"
    scheduler = (
        trainer.get_dummy_scheduler(optimizer)
        if optimizer_kind == "schedulefree"
        else torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 - min(step, 10) * 0.01)
    )
    train_fn = getattr(optimizer, "train", lambda: None)
    eval_fn = getattr(optimizer, "eval", lambda: None)
    sampler = None if shuffle else RecordingSampler(data_length)
    current_epoch = Value("i", 0)
    dataset = BucketedToyDataset(data_length, current_epoch) if bucketed else OrderedToyDataset(data_length)
    dataloader = DataLoader(
        dataset, batch_size=1, sampler=sampler, shuffle=shuffle, num_workers=num_workers, persistent_workers=persistent_workers
    )
    kwargs_handlers = None
    if distributed:
        kwargs_handlers = [
            InitProcessGroupKwargs(backend="gloo", init_method="env://?use_libuv=False", timeout=timedelta(seconds=60))
        ]
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=accumulation, kwargs_handlers=kwargs_handlers)
    args = _args(
        output_dir,
        max_steps=max_steps,
        accumulation=accumulation,
        resume=resume,
        save_steps=save_steps,
        save_epochs=save_epochs,
        sample_at_first=sample_at_first,
    )
    transformer, network, optimizer, dataloader, scheduler, training_model, network_dtype = trainer._prepare_with_accelerator(
        args, accelerator, transformer, network, optimizer, dataloader, scheduler, torch.float32, torch.float32, None
    )
    trainer._register_hooks_and_resume(args, accelerator, network)
    trainer._run_training_loop(
        args,
        accelerator,
        session_id=1,
        training_started_at=0.0,
        train_dataset_group=ToyDatasetGroup(data_length),
        train_dataloader=dataloader,
        current_epoch=current_epoch,
        transformer=transformer,
        network=network,
        training_model=network,
        optimizer=optimizer,
        optimizer_name=optimizer_name,
        optimizer_args="",
        optimizer_train_fn=train_fn,
        optimizer_eval_fn=eval_fn,
        lr_scheduler=scheduler,
        lr_descriptions=["toy"],
        vae=None,
        sample_parameters=None,
        dit_dtype=torch.float32,
        network_dtype=torch.float32,
    )
    last_state_dir = output_dir / train_utils.LAST_STATE_NAME.format("toy")
    state_dir = last_state_dir if _state_file(last_state_dir).exists() else None
    if state_dir is None and resume is not None and _state_file(resume).exists():
        state_dir = Path(resume)
    result = SimpleNamespace(
        trainer=trainer,
        seen=list(trainer.seen),
        noises=list(trainer.noises),
        sample_calls=list(trainer.sample_calls),
        sampler_epochs=[] if sampler is None else list(sampler.epochs),
        model_state=_cpu_copy(accelerator.unwrap_model(network).state_dict()),
        optimizer_state=_cpu_copy(optimizer.state_dict()),
        scheduler_state=_cpu_copy(scheduler.state_dict()) if hasattr(scheduler, "state_dict") else None,
        training_state=_training_state_dict(trainer),
        state_dir=state_dir,
    )
    return result


def _step_state_dir(output_dir, step):
    return Path(output_dir) / train_utils.STEP_STATE_NAME.format("toy", step)


def _epoch_state_dir(output_dir, epoch):
    return Path(output_dir) / train_utils.EPOCH_STATE_NAME.format("toy", epoch)


def test_local_resume_helper_loads_requested_state(tmp_path):
    trainer = NetworkTrainer()
    calls = []

    class FakeAccelerator:
        def load_state(self, path):
            calls.append(path)

    args = _args(tmp_path, max_steps=1, resume=tmp_path / "checkpoint")
    assert trainer.resume_from_local_or_hf_if_specified(FakeAccelerator(), args) is True
    assert calls == [str(tmp_path / "checkpoint")]


def test_huggingface_resume_helper_is_mocked_without_network(monkeypatch, tmp_path):
    trainer = NetworkTrainer()
    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()
    calls = []
    files = [SimpleNamespace(rfilename="checkpoint/optimizer.bin"), SimpleNamespace(rfilename="checkpoint/model.safetensors")]
    monkeypatch.setattr(trainer_base_module.huggingface_utils, "list_dir", lambda **kwargs: files)
    monkeypatch.setattr(
        trainer_base_module.huggingface_hub,
        "hf_hub_download",
        lambda **kwargs: str(downloaded / Path(kwargs["filename"]).name),
    )

    class FakeAccelerator:
        def load_state(self, path):
            calls.append(path)

    args = _args(tmp_path, max_steps=1, resume="owner/repo/checkpoint")
    args.resume_from_huggingface = True
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        assert trainer.resume_from_local_or_hf_if_specified(FakeAccelerator(), args) is True
    finally:
        loop.close()
        asyncio.set_event_loop(None)
    assert calls == [str(downloaded)]


@pytest.mark.parametrize(("optimizer_kind", "accumulation"), [("sgd", 1), ("adamw", 2)])
def test_mid_epoch_resume_matches_uninterrupted_training(tmp_path, optimizer_kind, accumulation):
    reference = _run_training(
        tmp_path / "reference", max_steps=7, data_length=5, accumulation=accumulation, optimizer_kind=optimizer_kind
    )
    partial = _run_training(
        tmp_path / "partial", max_steps=2, data_length=5, accumulation=accumulation, optimizer_kind=optimizer_kind, save_steps=1
    )
    resumed = _run_training(
        tmp_path / "resumed",
        max_steps=7,
        data_length=5,
        accumulation=accumulation,
        optimizer_kind=optimizer_kind,
        resume=_step_state_dir(tmp_path / "partial", 2),
    )
    _assert_nested_equal(reference.model_state, resumed.model_state)
    _assert_nested_equal(reference.optimizer_state, resumed.optimizer_state)
    _assert_nested_equal(reference.scheduler_state, resumed.scheduler_state)
    _assert_nested_equal(reference.training_state, resumed.training_state)
    assert partial.seen + resumed.seen == reference.seen


def test_epoch_end_resume_uses_next_epoch_cursor(tmp_path):
    _run_training(tmp_path / "epoch-source", max_steps=8, data_length=4, optimizer_kind="sgd", save_epochs=1)
    epoch_state_dir = _epoch_state_dir(tmp_path / "epoch-source", 1)
    state = _read_state(epoch_state_dir)
    assert state["global_step"] == 4 and state["epoch"] == 1 and state["batch_index"] == 0
    reference = _run_training(tmp_path / "epoch-reference", max_steps=12, data_length=4, optimizer_kind="sgd")
    resumed = _run_training(tmp_path / "epoch-resumed", max_steps=12, data_length=4, optimizer_kind="sgd", resume=epoch_state_dir)
    _assert_nested_equal(reference.model_state, resumed.model_state)
    _assert_nested_equal(reference.optimizer_state, resumed.optimizer_state)
    _assert_nested_equal(reference.training_state, resumed.training_state)
    assert resumed.seen == reference.seen[4:]
    assert resumed.sampler_epochs[0] == 1


def test_stepwise_state_normalizes_last_batch_to_next_epoch(tmp_path):
    result = _run_training(tmp_path / "stepwise", max_steps=3, data_length=5, accumulation=2, save_steps=1)
    step1, step2, step3 = (_read_state(_step_state_dir(tmp_path / "stepwise", step)) for step in (1, 2, 3))
    assert (step1["global_step"], step1["epoch"], step1["batch_index"]) == (1, 0, 2)
    assert (step2["global_step"], step2["epoch"], step2["batch_index"]) == (2, 0, 4)
    assert (step3["global_step"], step3["epoch"], step3["batch_index"]) == (3, 1, 0)
    assert result.seen == [0, 1, 2, 3, 4]
    assert not _step_state_dir(tmp_path / "stepwise", 4).exists()
    assert result.state_dir is not None


def test_last_state_preserves_partial_cursor(tmp_path):
    _run_training(tmp_path / "partial", max_steps=2, data_length=5, accumulation=2, save_steps=1)
    state = _read_state(tmp_path / "partial" / train_utils.LAST_STATE_NAME.format("toy"))
    assert (state["global_step"], state["epoch"], state["batch_index"]) == (2, 0, 4)


def test_resume_does_not_repeat_initial_sample(tmp_path):
    partial = _run_training(tmp_path / "sample-source", max_steps=1, data_length=3, sample_at_first=True, save_steps=1)
    assert partial.sample_calls == [(0, 0)]
    resumed = _run_training(
        tmp_path / "sample-resumed",
        max_steps=2,
        data_length=3,
        resume=_step_state_dir(tmp_path / "sample-source", 1),
        sample_at_first=True,
    )
    assert resumed.sample_calls == []


def test_already_complete_resume_is_a_noop(tmp_path):
    source = _run_training(tmp_path / "source", max_steps=3, data_length=4, save_steps=1)
    complete = _run_training(tmp_path / "complete", max_steps=3, data_length=4, resume=source.state_dir, sample_at_first=True)
    assert complete.seen == [] and complete.sample_calls == []


def test_sidecar_schema_and_prepared_dataloader_config(tmp_path):
    _run_training(tmp_path / "schema", max_steps=2, data_length=5, accumulation=2, save_steps=1)
    state = _read_state(_step_state_dir(tmp_path / "schema", 1))
    assert state[_state_version_key(state)] == 1
    assert {"global_step", "epoch", "batch_index", "dataloader_config"}.issubset(state)
    config = state["dataloader_config"]
    assert _config_length(config) == 5 and config["gradient_accumulation_steps"] == 2
    assert config["num_processes"] == 1 and config["split_batches"] is False
    assert config["num_timestep_buckets"] is None
    assert "timestep_range_pool" in state


def test_validate_epoch_end_cursor_is_side_effect_free(tmp_path):
    trainer = NetworkTrainer()
    config = {"prepared_len": 4, "gradient_accumulation_steps": 2, "num_processes": 1, "split_batches": False, "seed": None}
    trainer._training_state.load_state_dict(
        {"version": 1, "global_step": 2, "epoch": 0, "batch_index": 4, "dataloader_config": config}
    )
    trainer._training_state.validate(4, 2, 1, False, None, None)
    assert (trainer._training_state.epoch, trainer._training_state.batch_index) == (0, 4)


def test_inconsistent_resume_cursor_is_rejected(tmp_path):
    trainer = NetworkTrainer()
    config = {"prepared_len": 5, "gradient_accumulation_steps": 2, "num_processes": 1, "split_batches": False, "seed": None}
    trainer._training_state.load_state_dict(
        {"version": 1, "global_step": 1, "epoch": 100, "batch_index": 2, "dataloader_config": config}
    )
    with pytest.raises(ValueError, match="inconsistent|cursor|config"):
        trainer._training_state.validate(5, 2, 1, False, None, None)


@pytest.mark.parametrize("bad_field", ["version", "global_step", "batch_index"])
def test_malformed_resume_sidecar_is_rejected(tmp_path, bad_field):
    source = tmp_path / "malformed-source"
    _run_training(source, max_steps=2, data_length=4, save_steps=1)
    state_dir = _step_state_dir(source, 1)
    path, state = _state_file(state_dir), _read_state(state_dir)
    key = _state_version_key(state)
    if bad_field == "version":
        state[key] = 999
    elif bad_field == "global_step":
        state.pop("global_step")
    else:
        state["batch_index"] = 999
    torch.save(state, path)
    with pytest.raises(ValueError, match="state|version|cursor|batch|missing|Unsupported"):
        _run_training(tmp_path / "malformed-resumed", max_steps=3, data_length=4, resume=state_dir)


def test_huggingface_resume_without_seed_is_rejected(tmp_path):
    args = _args(tmp_path, max_steps=1, resume="owner/repo/state")
    args.seed = None
    args.resume_from_huggingface = True
    with pytest.raises(ValueError, match="resume_from_huggingface|seed"):
        NetworkTrainer()._init_session(args)


def test_resume_without_seed_uses_checkpoint_seed(tmp_path):
    source = tmp_path / "seed-source"
    _run_training(source, max_steps=2, data_length=4, save_steps=1)
    args = _args(tmp_path / "seed-resumed", max_steps=3, resume=_step_state_dir(source, 1))
    args.seed = None
    trainer = NetworkTrainer()
    trainer._init_session(args)
    assert args.seed == _read_state(_step_state_dir(source, 1))["dataloader_config"]["seed"]


def test_resume_rejects_dataloader_config_mismatch(tmp_path):
    source = tmp_path / "config-source"
    _run_training(source, max_steps=2, data_length=4, save_steps=1)
    state_dir = _step_state_dir(source, 1)
    state = _read_state(state_dir)
    state["dataloader_config"]["gradient_accumulation_steps"] = 2
    torch.save(state, _state_file(state_dir))
    with pytest.raises(ValueError, match="config|accumulation|dataloader|mismatch"):
        _run_training(tmp_path / "config-resumed", max_steps=3, data_length=4, resume=state_dir)


def _legacy_args(tmp_path, *, accumulation=1):
    args = _args(tmp_path, max_steps=3000, accumulation=accumulation, resume=tmp_path / "legacy")
    Path(args.resume).mkdir()
    return args


@pytest.mark.parametrize(
    ("split_batches", "expected_step", "expected_epoch", "expected_batch"),
    [(False, 966, 966 // 8, 966 % 8), (True, 1932, 1932 // 8, 1932 % 8)],
)
def test_legacy_scheduler_progress_uses_accelerated_scheduler_units(
    tmp_path, split_batches, expected_step, expected_epoch, expected_batch
):
    trainer = NetworkTrainer()
    trainer._resumed_from_state = True
    trainer._resume_state_present = False
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    scheduler_state = scheduler.state_dict()
    scheduler_state["last_epoch"] = 1932
    scheduler.load_state_dict(scheduler_state)
    accelerator = SimpleNamespace(num_processes=2, split_batches=split_batches)
    progress = trainer._restore_training_progress(
        args=_legacy_args(tmp_path), accelerator=accelerator, lr_scheduler=scheduler, train_dataloader=list(range(8))
    )
    assert progress == (expected_step, expected_epoch, expected_batch)
    state = trainer._training_state
    assert (state.global_step, state.epoch, state.batch_index) == progress


def test_legacy_schedule_free_scheduler_without_sidecar_fails_explicitly(tmp_path):
    trainer = NetworkTrainer()
    trainer._resumed_from_state = True
    trainer._resume_state_present = False
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=0.1)
    scheduler = trainer.get_dummy_scheduler(optimizer)
    accelerator = SimpleNamespace(num_processes=1, split_batches=False)
    with pytest.raises(ValueError, match="scheduler|state|legacy"):
        trainer._restore_training_progress(
            args=_legacy_args(tmp_path), accelerator=accelerator, lr_scheduler=scheduler, train_dataloader=list(range(4))
        )


def test_repeat_resume_from_same_checkpoint_is_deterministic(tmp_path):
    source = tmp_path / "source"
    _run_training(source, max_steps=2, data_length=5, accumulation=2, optimizer_kind="adamw", save_steps=1)
    first = _run_training(
        tmp_path / "resume-a", max_steps=7, data_length=5, accumulation=2, optimizer_kind="adamw", resume=_step_state_dir(source, 2)
    )
    second = _run_training(
        tmp_path / "resume-b", max_steps=7, data_length=5, accumulation=2, optimizer_kind="adamw", resume=_step_state_dir(source, 2)
    )
    _assert_nested_equal(first.model_state, second.model_state)
    _assert_nested_equal(first.optimizer_state, second.optimizer_state)
    _assert_nested_equal(first.scheduler_state, second.scheduler_state)
    _assert_nested_equal(first.training_state, second.training_state)
    assert first.seen == second.seen


@pytest.mark.parametrize("accumulation", [1, 2])
@pytest.mark.parametrize("cut_step", [2, 6])
def test_shuffled_resume_preserves_samples_and_noise(tmp_path, accumulation, cut_step):
    options = dict(max_steps=9, data_length=5, accumulation=accumulation, shuffle=True, noisy_loss=True, save_steps=1)
    reference = _run_training(tmp_path / "reference", **options)
    resumed = _run_training(tmp_path / "resumed", resume=_step_state_dir(tmp_path / "reference", cut_step), **options)
    per_epoch = (5 + accumulation - 1) // accumulation
    consumed = (cut_step // per_epoch) * 5 + (cut_step % per_epoch) * accumulation
    assert resumed.seen == reference.seen[consumed:]
    _assert_nested_equal(resumed.noises, reference.noises[consumed:])
    _assert_nested_equal(resumed.model_state, reference.model_state)
    _assert_nested_equal(resumed.optimizer_state, reference.optimizer_state)
    _assert_nested_equal(resumed.scheduler_state, reference.scheduler_state)


@pytest.mark.parametrize("num_workers,persistent_workers", [(0, False), (1, False), (1, True)])
def test_bucketed_shuffled_resume_preserves_samples_and_training_rng(tmp_path, num_workers, persistent_workers):
    options = dict(
        max_steps=13,
        data_length=5,
        shuffle=True,
        noisy_loss=True,
        bucketed=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        save_steps=1,
    )
    reference = _run_training(tmp_path / "reference", **options)
    resumed = _run_training(tmp_path / "resumed", resume=_step_state_dir(tmp_path / "reference", 7), **options)
    assert resumed.seen == reference.seen[7:]
    _assert_nested_equal(resumed.noises, reference.noises[7:])
    _assert_nested_equal(resumed.model_state, reference.model_state)
    _assert_nested_equal(resumed.optimizer_state, reference.optimizer_state)


def test_timestep_range_pool_round_trip():
    trainer = NetworkTrainer()
    trainer.timestep_range_pool = [(0.5, 1.0)]
    trainer._training_state.configure(4, 1, 1, False, 1234, 4)
    state = trainer._training_state
    state.timestep_range_pool = list(trainer.timestep_range_pool)
    restored = NetworkTrainer()
    restored._training_state.load_state_dict(state.state_dict())
    restored.timestep_range_pool = list(restored._training_state.timestep_range_pool)
    assert restored.timestep_range_pool == [(0.5, 1.0)]
    assert restored._training_state.dataloader_config["num_timestep_buckets"] == 4


def test_legacy_resume_initializes_state_before_next_optimizer_step(tmp_path):
    trainer = NetworkTrainer()
    trainer._resumed_from_state = True
    scheduler = SimpleNamespace(last_epoch=1932)
    args = _args(tmp_path, max_steps=5520, resume=tmp_path / "legacy")
    accelerator = SimpleNamespace(num_processes=1, split_batches=False)
    progress = trainer._restore_training_progress(args, accelerator, scheduler, list(range(276)))
    state = trainer._training_state
    assert progress == (1932, 7, 0)
    assert (state.global_step, state.epoch, state.batch_index) == progress


@pytest.mark.parametrize(("last_epoch", "accumulation", "expected"), [(5, 1, (5, 1, 0)), (3, 2, (3, 1, 0))])
def test_legacy_epoch_end_cursor_is_normalized(tmp_path, last_epoch, accumulation, expected):
    trainer = NetworkTrainer()
    trainer._resumed_from_state = True
    scheduler = SimpleNamespace(last_epoch=last_epoch)
    args = _args(tmp_path, max_steps=10, accumulation=accumulation, resume=tmp_path / "legacy")
    accelerator = SimpleNamespace(num_processes=1, split_batches=False)
    progress = trainer._restore_training_progress(args, accelerator, scheduler, list(range(5)))
    assert progress == expected


def test_legacy_resume_save_resume_round_trip(tmp_path):
    options = dict(max_steps=8, data_length=5, accumulation=2, save_steps=1)
    reference = _run_training(tmp_path / "reference", **options)
    legacy = _step_state_dir(tmp_path / "reference", 2)
    _state_file(legacy).unlink()  # a genuine Accelerate state without the new sidecar
    resumed = _run_training(tmp_path / "resume", resume=legacy, **options)
    next_state = _step_state_dir(tmp_path / "resume", 3)
    assert (_read_state(next_state)["global_step"], _read_state(next_state)["epoch"]) == (3, 1)
    resumed_again = _run_training(tmp_path / "resume-again", resume=next_state, **options)
    _assert_nested_equal(reference.model_state, resumed.model_state)
    _assert_nested_equal(reference.model_state, resumed_again.model_state)
    _assert_nested_equal(reference.optimizer_state, resumed_again.optimizer_state)


@pytest.mark.parametrize("cut_step", [2, 3])
def test_schedulefree_shuffled_resume_restores_optimizer_and_cursor(tmp_path, cut_step):
    options = dict(
        max_steps=7, data_length=5, accumulation=2, shuffle=True, noisy_loss=True, save_steps=1, optimizer_kind="schedulefree"
    )
    reference = _run_training(tmp_path / "reference", **options)
    state_dir = _step_state_dir(tmp_path / "reference", cut_step)
    assert not (state_dir / "scheduler.bin").exists()
    resumed = _run_training(tmp_path / "resumed", resume=state_dir, **options)
    consumed = (cut_step // 3) * 5 + (cut_step % 3) * 2
    assert resumed.seen == reference.seen[consumed:]
    _assert_nested_equal(reference.model_state, resumed.model_state)
    _assert_nested_equal(reference.optimizer_state, resumed.optimizer_state)


class SaveTrackingAccelerator:
    def __init__(self, is_main_process=True):
        self.is_main_process = is_main_process
        self.save_calls = []
        self.wait_calls = 0

    def save_state(self, path, **kwargs):
        self.save_calls.append((path, kwargs))
        Path(path).mkdir(parents=True, exist_ok=True)

    def wait_for_everyone(self):
        self.wait_calls += 1


def _state_helper_args(output_dir, **overrides):
    values = dict(
        output_dir=str(output_dir),
        output_name="toy",
        save_state_to_huggingface=False,
        save_every_n_steps=1,
        save_last_n_steps=None,
        save_last_n_steps_state=None,
        save_every_n_epochs=1,
        save_last_n_epochs=None,
        save_last_n_epochs_state=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_state_helpers_save_on_all_processes_with_synchronization(tmp_path):
    accelerator = SaveTrackingAccelerator(is_main_process=False)
    args = _state_helper_args(tmp_path)
    train_utils.save_and_remove_state_stepwise(args, accelerator, 2, sync_processes=True)
    train_utils.save_and_remove_state_on_epoch_end(args, accelerator, 1, sync_processes=True)
    train_utils.save_state_on_train_end(args, accelerator, sync_processes=True)
    assert len(accelerator.save_calls) == 3
    assert accelerator.wait_calls >= 6


def test_legacy_main_only_state_helpers_do_not_introduce_collectives(tmp_path):
    accelerator = SaveTrackingAccelerator(is_main_process=True)
    args = _state_helper_args(tmp_path)
    train_utils.save_and_remove_state_stepwise(args, accelerator, 2)
    train_utils.save_and_remove_state_on_epoch_end(args, accelerator, 1)
    train_utils.save_state_on_train_end(args, accelerator)
    assert len(accelerator.save_calls) == 3
    assert accelerator.wait_calls == 0


def test_state_helper_uploads_and_removes_only_on_main_process(tmp_path, monkeypatch):
    old_state = tmp_path / train_utils.STEP_STATE_NAME.format("toy", 1)
    old_state.mkdir()
    uploads = []
    monkeypatch.setattr(train_utils.huggingface_utils, "upload", lambda *args, **kwargs: uploads.append((args, kwargs)))
    args = _state_helper_args(tmp_path, save_state_to_huggingface=True, save_last_n_steps_state=1)
    accelerator = SaveTrackingAccelerator(is_main_process=True)
    train_utils.save_and_remove_state_stepwise(args, accelerator, 3, sync_processes=True)
    assert uploads and not old_state.exists()
    assert accelerator.wait_calls >= 2


def test_state_helper_non_main_does_not_upload_or_remove(tmp_path, monkeypatch):
    old_state = tmp_path / train_utils.STEP_STATE_NAME.format("toy", 1)
    old_state.mkdir()
    uploads = []
    monkeypatch.setattr(train_utils.huggingface_utils, "upload", lambda *args, **kwargs: uploads.append((args, kwargs)))
    args = _state_helper_args(tmp_path, save_state_to_huggingface=True, save_last_n_steps_state=1)
    accelerator = SaveTrackingAccelerator(is_main_process=False)
    train_utils.save_and_remove_state_stepwise(args, accelerator, 3, sync_processes=True)
    assert not uploads and old_state.exists()
    assert accelerator.wait_calls >= 2


def _distributed_case(root, phase):
    root = Path(root)
    rank = int(os.environ["RANK"])
    resume = None
    if phase == "mid":
        resume = _step_state_dir(root / "reference", 4)
    elif phase == "epoch":
        resume = _epoch_state_dir(root / "reference", 1)
    elif phase == "legacy":
        resume = _step_state_dir(root / "reference", 3)
    result = _run_training(
        root / phase,
        max_steps=6,
        data_length=10,
        accumulation=2,
        shuffle=True,
        noisy_loss=True,
        save_steps=1,
        save_epochs=1,
        resume=resume,
        distributed=True,
    )
    torch.save(
        {
            key: getattr(result, key)
            for key in ("seen", "noises", "model_state", "optimizer_state", "scheduler_state", "training_state")
        },
        root / f"{phase}-rank{rank}.pt",
    )


def _launch_two_processes(root, phase):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    children = []
    try:
        for rank in range(2):
            env = dict(
                os.environ,
                RANK=str(rank),
                LOCAL_RANK=str(rank),
                WORLD_SIZE="2",
                LOCAL_WORLD_SIZE="2",
                MASTER_ADDR="127.0.0.1",
                MASTER_PORT=str(port),
                USE_LIBUV="0",
                OMP_NUM_THREADS="1",
                ACCELERATE_USE_CPU="true",
                PYTHONIOENCODING="utf-8",
            )
            children.append(
                subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--distributed-case", str(root), phase],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
            )
        for child in children:
            output, _ = child.communicate(timeout=100)
            assert child.returncode == 0, output
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.communicate()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="requires torch.distributed Gloo")
def test_two_process_shuffled_checkpoint_round_trip(tmp_path):
    _launch_two_processes(tmp_path, "reference")
    for state_dir in (
        _step_state_dir(tmp_path / "reference", 4),
        _epoch_state_dir(tmp_path / "reference", 1),
        tmp_path / "reference/toy-state",
    ):
        assert (state_dir / "trainer_state.pt").exists()
        assert (state_dir / "model.safetensors").exists()
        assert (state_dir / "optimizer.bin").exists()
        assert (state_dir / "random_states_0.pkl").exists()
        assert (state_dir / "random_states_1.pkl").exists()
    for phase, consumed in (("mid", 7), ("epoch", 5), ("legacy", 5)):
        if phase == "legacy":
            _state_file(_step_state_dir(tmp_path / "reference", 3)).unlink()
        _launch_two_processes(tmp_path, phase)
        for rank in range(2):
            reference = torch.load(tmp_path / f"reference-rank{rank}.pt", weights_only=False)
            resumed = torch.load(tmp_path / f"{phase}-rank{rank}.pt", weights_only=False)
            assert resumed["seen"] == reference["seen"][consumed:]
            _assert_nested_equal(resumed["noises"], reference["noises"][consumed:])
            for key in ("model_state", "optimizer_state", "scheduler_state", "training_state"):
                _assert_nested_equal(resumed[key], reference[key])


if __name__ == "__main__" and sys.argv[1:2] == ["--distributed-case"]:
    _distributed_case(sys.argv[2], sys.argv[3])
