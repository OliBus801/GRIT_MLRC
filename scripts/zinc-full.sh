#!/bin/bash
#SBATCH --time=06:30:00
#SBATCH --account=def-chgag196
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=50G
#SBATCH --output=results/zinc_full/zinc-full-GRIT-RRWP/log/slurm_%j.out

# Vérification que N_EPOCH est défini
if [ -z "$N_EPOCH" ]; then
  echo "Erreur : la variable d'environnement N_EPOCH n'est pas définie."
  exit 1
fi

if [ -z "$RAND_SEED" ]; then
  echo "Erreur : la variable d'environnement RAND_SEED n'est pas définie."
  exit 1
fi

# Set repository for librairies installation
cd /home/olbus4/scratch/GRIT/env

# Creating virtual environment
module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate


# Installing necessary librairies
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
pip install deepsnap-0.2.0-py3-none-any.whl ogb-1.3.6-py3-none-any.whl graphgym-0.4.0-py3-none-any.whl outdated-0.2.2-py2.py3-none-any.whl littleutils-0.2.4-py3-none-any.whl

# Go back to root repository
cd /home/olbus4/scratch/GRIT

echo "Launching experiments with N_EPOCH=$N_EPOCH and RAND_SEED=$RAND_SEED"

# Run 4 training runs with different seeds on 4 GPUs
for i in 0; do
  seed=$(($RAND_SEED + i))
  python main.py --cfg configs/GRIT/zinc-full-GRIT-RRWP.yaml accelerator "cuda:$i" optim.max_epoch "$N_EPOCH" seed "$seed" &
done
wait

echo "Test Experiment ZINC-120k done !"

# Calcul de la nouvelle valeur de N_EPOCH
NEXT_EPOCH=$((N_EPOCH + 100))
export N_EPOCH=$NEXT_EPOCH

# Vérification de la limite
if [ "$NEXT_EPOCH" -gt 2000 ]; then
    echo "Fin des 2000 epochs"
    exit 0
fi

# Lancement du prochain script avec la nouvelle valeur
echo "Launching next script with N_EPOCH=$N_EPOCH AND RAND_SEED=$RAND_SEED"
sbatch --export=ALL scripts/zinc.sh
