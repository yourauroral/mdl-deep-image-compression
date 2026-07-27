import sys
from pathlib import Path

from scripts.run_formal_evaluations import (
    ROOT,
    _evaluation_command,
    main,
)


def _evaluation(tmp_path: Path, *, nproc_per_node: int, attachment=None):
    return _evaluation_command(
        config=tmp_path / "config.yaml",
        checkpoint=tmp_path / "best.pth",
        result_json=tmp_path / "teacher_forced.json",
        batch_size=8,
        nproc_per_node=nproc_per_node,
        bootstrap_samples=100,
        codec_verification_json=attachment,
    )


def test_single_process_evaluation_command_uses_formal_protocol(tmp_path):
    command = _evaluation(tmp_path, nproc_per_node=1)

    assert command[:2] == [
        sys.executable,
        str(ROOT / "scripts/evaluate.py"),
    ]
    assert "--formal" in command
    assert "--result_json" in command
    assert "--codec_verification_json" not in command


def test_multi_process_evaluation_command_uses_torchrun_module(tmp_path):
    command = _evaluation(tmp_path, nproc_per_node=4)

    assert command[:5] == [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
    ]
    assert command[5].endswith("scripts/evaluate.py")


def test_evaluation_command_attaches_explicit_roundtrip_manifest(tmp_path):
    attachment = tmp_path / "sequential_roundtrip.json"
    command = _evaluation(
        tmp_path,
        nproc_per_node=1,
        attachment=attachment,
    )

    option = command.index("--codec_verification_json")
    assert command[option + 1] == str(attachment)


def test_dry_run_defaults_to_both_datasets(capsys, tmp_path):
    main(["--dry_run", "--output_dir", str(tmp_path)])

    output = capsys.readouterr().out
    assert output.count("scripts/evaluate.py") == 2
    assert "ccigpt_cifar10_s_rgb_ronly_v2.yaml" in output
    assert "ccigpt_imagenet64_v1.yaml" in output
    assert "scripts/verify_lossless.py" not in output


def test_dry_run_verification_precedes_and_attaches_to_evaluation(capsys, tmp_path):
    main([
        "cifar10",
        "--dry_run",
        "--output_dir", str(tmp_path),
        "--verify_cifar_images", "2",
    ])

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert "scripts/verify_lossless.py" in lines[0]
    assert "--num_images 2" in lines[0]
    assert "scripts/evaluate.py" in lines[1]
    assert "--codec_verification_json" in lines[1]


def test_dry_run_does_not_implicitly_reuse_stale_verification(capsys, tmp_path):
    stale = tmp_path / "cifar10/sequential_roundtrip.json"
    stale.parent.mkdir()
    stale.write_text("{}")

    main([
        "cifar10",
        "--dry_run",
        "--output_dir", str(tmp_path),
    ])

    output = capsys.readouterr().out
    assert "--codec_verification_json" not in output


def test_dry_run_can_explicitly_reuse_verification(capsys, tmp_path):
    manifest = tmp_path / "roundtrip.json"
    main([
        "cifar10",
        "--dry_run",
        "--output_dir", str(tmp_path / "results"),
        "--cifar_verification_json", str(manifest),
    ])

    output = capsys.readouterr().out
    assert f"--codec_verification_json {manifest}" in output
