mkdir -p buffer_folder
pwd=$(pwd)
echo $pwd

if [[ $1 -eq 1 ]]
then
    export BUFFER_PATH=$pwd/buffer_folder && ./singularity_run.sh python3 td3/train.py --config_path ./configs/motion_laser.yaml
else
    export BUFFER_PATH=$pwd/buffer_folder && python3 scripts/train_condor.py --local_update --config_path ./configs/motion_laser.yaml
fi

