#!/bin/bash

#SBATCH --job-name slurmjob                                        # Job name

### Logging
#SBATCH --output=slurm_logs/slurmjob_%j.out                    # Name of stdout output file (%j expands to jobId)
#SBATCH --error=slurm_logs/slurmjob_%j.err                        # Name of stderr output file (%j expands to jobId)
#SBATCH --mail-user=zifan@cs.utexas.edu  # Email of notification
#SBATCH --mail-type=END,FAIL,REQUEUE                                      

### Node info
#SBATCH --partition dgx                        # Queue name - current options are titans and dgx
#SBATCH --nodes=1                                                            # Always set to 1 when using the cluster
#SBATCH --ntasks-per-node=1                                        # Number of tasks per node
#SBATCH --time 1:00:00                                                     # Run time (hh:mm:ss)

#SBATCH --gres=gpu:1                                                       # Number of gpus needed
#SBATCH --mem=5G                                                         # Memory requirements
#SBATCH --cpus-per-task=8                                              # Number of cpus needed per task

./executable/run_central_learner.sh configs/motion.yaml
