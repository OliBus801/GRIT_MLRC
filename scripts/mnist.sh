#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --account=aip-chgag196
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=50000M
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=results/mnist-GRIT-RRWP/log/slurm_%j.out

# On définit les variables d'environnements
export DATASET=mnist

# Paramètres d'entraînement (split epochs)
SEED_BASE=${RAND_SEED:-0}
DELTA_EPOCH=${DELTA_EPOCH:-50}
EPOCHS=${EPOCH_GOAL:-${DELTA_EPOCH:-0}}
MAX_EPOCH=${MAX_EPOCH:-200}

# Vérification qu'on a epoch > 0
if [ "$EPOCHS" -le 0 ]; then
  echo "[ERREUR] Ni EPOCH_GOAL ni DELTA_EPOCH n'ont été spécifiés. Impossible de déterminer le nombre d'époques à exécuter."
  exit 1
fi

# Set repository for librairies installation
cd /home/o/olbus4/links/scratch/GRIT_MLRC/env/

# Creating virtual environment
module load python/3.10
module load cuda
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate


# Installing necessary librairies
pip install --no-index --upgrade pip
pip install --no-index -r requirements_tamia.txt
pip install deepsnap-0.2.0-py3-none-any.whl ogb-1.3.6-py3-none-any.whl graphgym-0.4.0-py3-none-any.whl outdated-0.2.2-py2.py3-none-any.whl littleutils-0.2.4-py3-none-any.whl propcache-0.3.1-py3-none-any.whl
# Go back to root repository
cd /home/o/olbus4/links/scratch/GRIT_MLRC/

echo "Launching experiments with EPOCH_GOAL=$EPOCHS and RAND_SEED=$SEED_BASE"

# On s'assure de nettoyer le param input
rm -f scripts/params/$DATASET.input

# Run 4 training runs with different seeds on 4 GPUs
for i in {0..3}; do
  seed=$(($SEED_BASE + $i))
  echo "python main.py --cfg configs/GRIT/$DATASET-GRIT-RRWP.yaml accelerator cuda:0 train.ckpt_period $DELTA_EPOCH optim.max_epoch $EPOCHS seed $seed" >> scripts/params/$DATASET.input
done

# Lancement parallèle de 4 tâches, chacune sur un GPU distinct
cat scripts/params/$DATASET.input | parallel -j4 --lb 'CUDA_VISIBLE_DEVICES=$(({%} - 1)) bash -c {} > results/$DATASET-GRIT-RRWP/log/slogs/slog_{#}.out 2>&1'

# Calcul de la nouvelle valeur de EPOCH_GOAL
NEXT_EPOCH=$((EPOCHS + DELTA_EPOCH))
export EPOCH_GOAL=$NEXT_EPOCH

# Vérification de la limite
if [ "$NEXT_EPOCH" -gt "$MAX_EPOCH" ]; then
    echo "Limite atteinte : $NEXT_EPOCH > $MAX_EPOCH. Fin des jobs."
    exit 0
fi

# Lancement du prochain script avec la nouvelle valeur
echo "Launching next script with EPOCH_GOAL=$EPOCH_GOAL AND RAND_SEED=$RAND_SEED"
sbatch --export=ALL scripts/$DATASET.sh
