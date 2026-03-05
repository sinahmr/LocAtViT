import logging

import os
import subprocess
from datetime import datetime, timedelta

_logger = logging.getLogger('slurm')

def get_slurm_end_time(job_id, print_info=False):
    try:
        if not job_id:
            return None
        job_info = subprocess.check_output(f"scontrol show job {job_id}", shell=True).decode()
        if print_info:
            _logger.info(f"\nSLURM job info: {job_info.strip()}\n")

        end_time_str = next(s.split("=")[1] for s in job_info.split() if "EndTime" in s)
        return datetime.strptime(end_time_str, "%Y-%m-%dT%H:%M:%S")
    except Exception as e:
        _logger.error('Error in get_slurm_end_time: %s', e)

def get_slurm_remaining_time(end_time):
    if not end_time:
        _logger.info("The job is not being run using SLURM, ignoring time check...")
        return float("inf")
    return max((end_time - datetime.now()).total_seconds(), 0)

def get_slurm_arguments(job_id):
    if not job_id:
        return ''
    time_limit = str(timedelta(seconds=int(os.environ.get("SLURM_JOB_END_TIME")) - int(os.environ.get("SLURM_JOB_START_TIME"))))
    time_limit = time_limit.replace(' ', '').replace('days,', '-')

    job_info = subprocess.check_output(f"scontrol show job {job_id}", shell=True).decode()
    out_dir = '/'.join(job_info.split('StdOut=')[1].split()[0].split('/')[:-1])

    gpu_per_node = os.environ.get("SLURM_GPUS_PER_NODE", "")
    if ":" not in gpu_per_node:
        gpu_per_node = job_info.split('gres/gpu:')[1].split()[0].replace("=", ":")

    return (f'--account={os.environ.get("SLURM_JOB_ACCOUNT")} '
            f'--output={out_dir}/%x.%j.out '
            f'--gpus-per-node={gpu_per_node} '
            f'--ntasks-per-node={os.environ.get("SLURM_NTASKS_PER_NODE")} '
            f'--cpus-per-task={os.environ.get("SLURM_CPUS_PER_TASK")} '
            f'--time={time_limit} '
            f'--job-name={os.environ.get("SLURM_JOB_NAME")}')

def reschedule_job(job_id, config, checkpoint, wandb_run):
    try:
        job_info = subprocess.check_output(f"scontrol show job {job_id}", shell=True).decode()
        job_script = next(s.split("=")[1] for s in job_info.split() if "Command" in s)
        slurm_args = get_slurm_arguments(job_id)

        arguments = f"--config {config} --resume {checkpoint} --slurm-reschedule"
        n_proc = os.environ.get("NPROC_PER_NODE", 1)
        dataset = os.environ.get("DATASET_NAME", "imagenet")
        if wandb_run:
            arguments += f" --wandb-resume-id {wandb_run.id}"
        command = ["sbatch", slurm_args, job_script, n_proc, dataset, arguments]
        _logger.info(" ".join(command))
        return " ".join(command)
    except Exception as e:
        _logger.error('Error in reschedule_job: %s', e)
