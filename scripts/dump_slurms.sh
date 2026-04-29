for f in bench/results/slurm-*.out; do
    echo "===== $f ====="
    cat "$f"
    echo
done
