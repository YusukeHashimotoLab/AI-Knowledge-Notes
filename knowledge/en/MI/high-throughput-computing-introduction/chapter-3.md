---
title: "Chapter 3: Job Scheduling and Parallelization (SLURM, PBS)"
chapter_title: "Chapter 3: Job Scheduling and Parallelization (SLURM, PBS)"
subtitle: 
reading_time: 25-30 min
difficulty: Advanced
code_examples: 5
exercises: 0
version: 1.0
created_at: 2025-10-17
---

# Chapter 3: Job Scheduling and Parallelization (SLURM, PBS)

Understand dependency management and reproducibility design in FireWorks/AiiDA. Also cover essential logging and observability concepts.

**💡 Note:** Visualize the overall picture with DAG (flowcharts). Determining where to save intermediate artifacts speeds up recovery.

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Create SLURM scripts and submit jobs
  * ✅ Evaluate MPI parallel computing efficiency
  * ✅ Write job management scripts in Python
  * ✅ Design parallel computations for 1000-material scale
  * ✅ Perform tuning through benchmarking

* * *

## 3.1 Job Scheduler Fundamentals

### SLURM vs PBS vs Torque

Feature | SLURM | PBS Pro | Torque  
---|---|---|---  
Developer | SchedMD | Altair | Adaptive Computing  
License | GPL (partial commercial) | Commercial | Open Source  
Adoption | TSUBAME, many TOP500 | NASA, DOE national labs | Many universities  
Commands | `sbatch`, `squeue` | `qsub`, `qstat` | `qsub`, `qstat`  
Recommended Use | Large-scale HPC | Enterprise | Small to medium HPC  
  
### SLURM Basic Concepts
    
    
    ```mermaid
    flowchart TD
        A[User] -->|sbatch| B[Job Queue]
        B --> C{Scheduler}
        C -->|Resource Allocation| D[Compute Node 1]
        C -->|Resource Allocation| E[Compute Node 2]
        C -->|Resource Allocation| F[Compute Node N]
    
        D --> G[Job Completion]
        E --> G
        F --> G
    
        style C fill:#4ecdc4
    ```

**Key Commands** :
    
    
    # Submit job
    sbatch job.sh
    
    # Check job status
    squeue -u username
    
    # Cancel job
    scancel job_id
    
    # Node information
    sinfo
    
    # Job details
    scontrol show job job_id
    

* * *

## 3.2 Creating SLURM Scripts

### Basic SLURM Script
    
    
    #!/bin/bash
    #SBATCH --job-name=vasp_relax       # Job name
    #SBATCH --output=slurm-%j.out       # Standard output (%j=job ID)
    #SBATCH --error=slurm-%j.err        # Standard error
    #SBATCH --nodes=1                   # Number of nodes
    #SBATCH --ntasks-per-node=48        # MPI processes per node
    #SBATCH --cpus-per-task=1           # Threads per task
    #SBATCH --time=24:00:00             # Time limit (24 hours)
    #SBATCH --partition=standard        # Partition (queue)
    #SBATCH --account=project_name      # Project name
    
    # Environment setup
    module purge
    module load intel/2021.2
    module load vasp/6.3.0
    
    # Working directory
    cd $SLURM_SUBMIT_DIR
    
    # Run VASP
    echo "Job started: $(date)"
    echo "Host: $(hostname)"
    echo "Number of nodes: $SLURM_JOB_NUM_NODES"
    echo "Number of processes: $SLURM_NTASKS"
    
    mpirun -np $SLURM_NTASKS vasp_std
    
    echo "Job finished: $(date)"
    

### Parallel Execution with Array Jobs

**Parallel computation of 100 materials** :
    
    
    #!/bin/bash
    #SBATCH --job-name=vasp_array
    #SBATCH --output=logs/slurm-%A_%a.out  # %A=array job ID, %a=task ID
    #SBATCH --error=logs/slurm-%A_%a.err
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=48
    #SBATCH --time=24:00:00
    #SBATCH --array=1-100%10              # Tasks 1-100, max 10 concurrent
    
    # Environment setup
    module load vasp/6.3.0
    
    # Load material list
    MATERIAL_LIST="materials.txt"
    MATERIAL=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $MATERIAL_LIST)
    
    echo "Processing: TaskID=${SLURM\_ARRAY\_TASK\_ID}, Material=${MATERIAL}"
    
    # Working directory
    WORK_DIR="calculations/${MATERIAL}"
    cd $WORK_DIR
    
    # Run VASP
    mpirun -np 48 vasp_std
    
    # Convergence check
    if grep -q "reached required accuracy" OUTCAR; then
        echo "SUCCESS: ${MATERIAL}" >> ../completed.log
    else
        echo "FAILED: ${MATERIAL}" >> ../failed.log
    fi
    

**Example materials.txt** :
    
    
    LiCoO2
    LiNiO2
    LiMnO2
    LiFePO4
    ...(100 lines)
    

### Job Chains with Dependencies
    
    
    # Step 1: Structure relaxation
    JOB1=$(sbatch --parsable relax.sh)
    echo "Relaxation job ID: $JOB1"
    
    # Step 2: Static calculation (run after relaxation)
    JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 static.sh)
    echo "Static calculation job ID: $JOB2"
    
    # Step 3: Band structure (run after static calculation)
    JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 band.sh)
    echo "Band structure job ID: $JOB3"
    
    # Step 4: Data analysis (run after all complete)
    sbatch --dependency=afterok:$JOB3 analysis.sh
    

* * *

## 3.3 MPI Parallel Computing

### Types of Parallelization
    
    
    # 1. Task Parallelism (Recommended: High-throughput computing)
    # Compute 100 materials simultaneously on 100 nodes
    # Scaling efficiency: 100%
    
    # 2. Data Parallelism (VASP: KPAR setting)
    # Divide k-points into 4 groups
    INCAR: KPAR = 4
    # Scaling efficiency: 80-90%
    
    # 3. MPI Parallelism (inter-process communication)
    # Distribute 1 calculation across 48 cores
    mpirun -np 48 vasp_std
    # Scaling efficiency: 50-70%
    

### Measuring Scaling Efficiency
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    
    import subprocess
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    
    def benchmark_scaling(structure, core_counts=[1, 2, 4, 8, 16, 32, 48]):
        """
        Benchmark parallelization efficiency
    
        Parameters:
        -----------
        structure : str
            Test structure
        core_counts : list
            Core counts to test
    
        Returns:
        --------
        efficiency : dict
            Efficiency for each core count
        """
        timings = {}
    
        for n_cores in core_counts:
            # Run VASP
            start = time.time()
    
            result = subprocess.run(
                [f"mpirun -np {n_cores} vasp_std"],
                shell=True,
                cwd=f"benchmark_{n_cores}cores"
            )
    
            elapsed = time.time() - start
            timings[n_cores] = elapsed
    
            print(f"{n_cores} cores: {elapsed:.1f} seconds")
    
        # Calculate efficiency
        base_time = timings[1]
        efficiency = {}
    
        for n_cores, t in timings.items():
            ideal_time = base_time / n_cores
            actual_speedup = base_time / t
            ideal_speedup = n_cores
    
            eff = (actual_speedup / ideal_speedup) * 100
            efficiency[n_cores] = eff
    
        return efficiency, timings
    
    # Plot results
    def plot_scaling(efficiency, timings):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
        cores = list(timings.keys())
        times = list(timings.values())
        effs = [efficiency[c] for c in cores]
    
        # Speedup
        ax1.plot(cores, [timings[1]/t for t in times], 'o-', label='Actual')
        ax1.plot(cores, cores, '--', label='Ideal (linear)')
        ax1.set_xlabel('Number of Cores')
        ax1.set_ylabel('Speedup')
        ax1.set_xscale('log', base=2)
        ax1.set_yscale('log', base=2)
        ax1.legend()
        ax1.grid(True)
    
        # Efficiency
        ax2.plot(cores, effs, 'o-', color='green')
        ax2.axhline(y=80, color='red', linestyle='--', label='80% target')
        ax2.set_xlabel('Number of Cores')
        ax2.set_ylabel('Parallelization Efficiency (%)')
        ax2.set_xscale('log', base=2)
        ax2.legend()
        ax2.grid(True)
    
        plt.tight_layout()
        plt.savefig('scaling_benchmark.png', dpi=300)
        plt.show()
    

### Optimizing VASP Parallelization Parameters
    
    
    def optimize_vasp_parallelization(n_kpoints, n_bands, n_cores=48):
        """
        Optimize VASP parallelization parameters
    
        Parameters:
        -----------
        n_kpoints : int
            Number of k-points
        n_bands : int
            Number of bands
        n_cores : int
            Available cores
    
        Returns:
        --------
        params : dict
            Optimal parameters
        """
        # KPAR: k-point parallelism (most efficient)
        # Power of 2, divisor of k-point count
        kpar_options = [1, 2, 4, 8, 16]
        valid_kpar = [k for k in kpar_options if n_kpoints % k == 0 and k <= n_cores]
    
        # NCORE: band parallelism
        # Typically 4-8 is optimal
        ncore = min(4, n_cores // max(valid_kpar))
    
        # Recommended settings
        recommended = {
            'KPAR': max(valid_kpar),
            'NCORE': ncore,
            'cores_per_kpar_group': n_cores // max(valid_kpar),
        }
    
        print("Recommended parallelization parameters:")
        print(f"  KPAR = {recommended['KPAR']} (k-point parallelism)")
        print(f"  NCORE = {recommended['NCORE']} (band parallelism)")
        print(f"  {recommended['cores_per_kpar_group']} cores per KPAR group")
    
        return recommended
    
    # Usage example
    params = optimize_vasp_parallelization(n_kpoints=64, n_bands=200, n_cores=48)
    # Recommended:
    #   KPAR = 16 (k-point parallelism)
    #   NCORE = 3 (band parallelism)
    #   3 cores per KPAR group
    

* * *

## 3.4 Job Management with Python

### Job Submission and Monitoring
    
    
    import subprocess
    import time
    import re
    
    class SLURMJobManager:
        """SLURM job management class"""
    
        def submit_job(self, script_path):
            """
            Submit a job
    
            Returns:
            --------
            job_id : int
                Job ID
            """
            result = subprocess.run(
                ['sbatch', script_path],
                capture_output=True,
                text=True
            )
    
            # Extract ID from "Submitted batch job 12345"
            match = re.search(r'(\d+)', result.stdout)
            if match:
                job_id = int(match.group(1))
                print(f"Job submitted: ID={job_id}")
                return job_id
            else:
                raise RuntimeError(f"Job submission failed: {result.stderr}")
    
        def get_job_status(self, job_id):
            """
            Get job status
    
            Returns:
            --------
            status : str
                'PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED'
            """
            result = subprocess.run(
                ['squeue', '-j', str(job_id), '-h', '-o', '%T'],
                capture_output=True,
                text=True
            )
    
            if result.stdout.strip():
                return result.stdout.strip()
            else:
                # Not in queue → completed or failed
                result = subprocess.run(
                    ['sacct', '-j', str(job_id), '-X', '-n', '-o', 'State'],
                    capture_output=True,
                    text=True
                )
                return result.stdout.strip()
    
        def wait_for_completion(self, job_id, check_interval=60):
            """
            Wait for job completion
    
            Parameters:
            -----------
            job_id : int
                Job ID
            check_interval : int
                Check interval (seconds)
            """
            while True:
                status = self.get_job_status(job_id)
    
                if status in ['COMPLETED', 'FAILED', 'CANCELLED']:
                    print(f"Job {job_id}: {status}")
                    return status
    
                print(f"Job {job_id}: {status}...waiting")
                time.sleep(check_interval)
    
        def submit_array_job(self, script_path, n_tasks, max_concurrent=10):
            """
            Submit array job
    
            Parameters:
            -----------
            n_tasks : int
                Number of tasks
            max_concurrent : int
                Maximum concurrent tasks
            """
            result = subprocess.run(
                ['sbatch', f'--array=1-{n_tasks}%{max_concurrent}', script_path],
                capture_output=True,
                text=True
            )
    
            match = re.search(r'(\d+)', result.stdout)
            if match:
                job_id = int(match.group(1))
                print(f"Array job submitted: ID={job_id}, Tasks={n_tasks}")
                return job_id
            else:
                raise RuntimeError(f"Submission failed: {result.stderr}")
    
    # Usage example
    manager = SLURMJobManager()
    
    # Single job
    job_id = manager.submit_job('relax.sh')
    status = manager.wait_for_completion(job_id)
    
    # Array job
    array_id = manager.submit_array_job('array_job.sh', n_tasks=100, max_concurrent=20)
    

### Large-Scale Job Management (1000 Materials)
    
    
    import os
    from pathlib import Path
    import json
    
    def manage_large_scale_calculation(materials, batch_size=100):
        """
        Efficiently manage 1000 materials
    
        Parameters:
        -----------
        materials : list
            List of materials
        batch_size : int
            Batch size
        """
        manager = SLURMJobManager()
        n_materials = len(materials)
    
        print(f"Total materials: {n_materials}")
        print(f"Batch size: {batch_size}")
    
        # Divide into batches
        n_batches = (n_materials + batch_size - 1) // batch_size
        print(f"Number of batches: {n_batches}")
    
        job_ids = []
    
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_materials)
            batch_materials = materials[start_idx:end_idx]
    
            print(f"\nBatch {batch_idx+1}/{n_batches}")
            print(f"  Materials: {len(batch_materials)}")
    
            # Create material list for batch
            list_file = f"batch_{batch_idx+1}_materials.txt"
            with open(list_file, 'w') as f:
                for mat in batch_materials:
                    f.write(f"{mat}\n")
    
            # Submit array job
            job_id = manager.submit_array_job(
                'vasp_array.sh',
                n_tasks=len(batch_materials),
                max_concurrent=20
            )
    
            job_ids.append(job_id)
    
        # Monitor progress
        print("\nMonitoring progress...")
        completed = 0
    
        while completed < len(job_ids):
            time.sleep(300)  # Check every 5 minutes
    
            for i, job_id in enumerate(job_ids):
                status = manager.get_job_status(job_id)
    
                if status == 'COMPLETED' and i not in completed_jobs:
                    completed += 1
                    print(f"Batch {i+1} completed ({completed}/{len(job_ids)})")
    
        print("All batches completed!")
    
    # Usage example
    materials = [f"material_{i:04d}" for i in range(1, 1001)]
    manage_large_scale_calculation(materials, batch_size=100)
    

* * *

## 3.5 Exercises

### Problem 1 (Difficulty: easy)

**Problem** : Create a SLURM script with the following conditions:

  * Job name: `si_bandgap`
  * Number of nodes: 2
  * Processes per node: 24
  * Time limit: 12 hours
  * Partition: `gpu`

Solution
    
    
    #!/bin/bash
    #SBATCH --job-name=si_bandgap
    #SBATCH --output=slurm-%j.out
    #SBATCH --error=slurm-%j.err
    #SBATCH --nodes=2
    #SBATCH --ntasks-per-node=24
    #SBATCH --time=12:00:00
    #SBATCH --partition=gpu
    
    module load vasp/6.3.0
    
    cd $SLURM_SUBMIT_DIR
    
    mpirun -np 48 vasp_std
    

### Problem 2 (Difficulty: medium)

**Problem** : Create an array job for 50 structure relaxation calculations, with 10 running concurrently.

Solution
    
    
    #!/bin/bash
    #SBATCH --job-name=relax_array
    #SBATCH --output=logs/slurm-%A_%a.out
    #SBATCH --error=logs/slurm-%A_%a.err
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=48
    #SBATCH --time=24:00:00
    #SBATCH --array=1-50%10
    
    module load vasp/6.3.0
    
    MATERIAL=$(sed -n "${SLURM_ARRAY_TASK_ID}p" materials.txt)
    
    cd calculations/${MATERIAL}
    
    mpirun -np 48 vasp_std
    

### Problem 3 (Difficulty: hard)

**Problem** : Create a Python script to manage VASP calculations for 1000 materials. Requirements:

  1. Process in batches of 50 materials
  2. 20 tasks running concurrently per batch
  3. Log completed and failed tasks
  4. Automatically retry failed tasks

Solution
    
    
    import os
    import time
    import json
    from pathlib import Path
    
    class HighThroughputManager:
        def __init__(self, materials, batch_size=50, max_concurrent=20):
            self.materials = materials
            self.batch_size = batch_size
            self.max_concurrent = max_concurrent
            self.manager = SLURMJobManager()
    
            self.results = {
                'completed': [],
                'failed': [],
                'retry': []
            }
    
        def run_batch(self, batch_materials, batch_id):
            """Run batch"""
            # Create material list
            list_file = f"batch_{batch_id}.txt"
            with open(list_file, 'w') as f:
                for mat in batch_materials:
                    f.write(f"{mat}\n")
    
            # Submit job
            job_id = self.manager.submit_array_job(
                'vasp_array.sh',
                n_tasks=len(batch_materials),
                max_concurrent=self.max_concurrent
            )
    
            # Wait for completion
            status = self.manager.wait_for_completion(job_id)
    
            # Check results
            self.check_results(batch_materials, batch_id)
    
        def check_results(self, batch_materials, batch_id):
            """Check results"""
            for mat in batch_materials:
                outcar = f"calculations/{mat}/OUTCAR"
    
                if not os.path.exists(outcar):
                    self.results['failed'].append(mat)
                    continue
    
                with open(outcar, 'r') as f:
                    content = f.read()
                    if 'reached required accuracy' in content:
                        self.results['completed'].append(mat)
                    else:
                        self.results['failed'].append(mat)
    
            # Save log
            with open(f'batch_{batch_id}_results.json', 'w') as f:
                json.dump(self.results, f, indent=2)
    
        def retry_failed(self, max_retries=2):
            """Retry failed tasks"""
            for retry_count in range(max_retries):
                if not self.results['failed']:
                    break
    
                print(f"\nRetry {retry_count+1}/{max_retries}")
                print(f"  Failed tasks: {len(self.results['failed'])}")
    
                failed_materials = self.results['failed'].copy()
                self.results['failed'] = []
    
                self.run_batch(failed_materials, f'retry_{retry_count+1}')
    
        def execute_all(self):
            """Execute all materials"""
            n_batches = (len(self.materials) + self.batch_size - 1) // self.batch_size
    
            for batch_idx in range(n_batches):
                start = batch_idx * self.batch_size
                end = min((batch_idx + 1) * self.batch_size, len(self.materials))
                batch_materials = self.materials[start:end]
    
                print(f"\nBatch {batch_idx+1}/{n_batches}")
                self.run_batch(batch_materials, batch_idx+1)
    
            # Retry
            self.retry_failed(max_retries=2)
    
            # Final report
            print("\nFinal results:")
            print(f"  Completed: {len(self.results['completed'])}")
            print(f"  Failed: {len(self.results['failed'])}")
    
            return self.results
    
    # Execute
    materials = [f"material_{i:04d}" for i in range(1, 1001)]
    manager = HighThroughputManager(materials, batch_size=50, max_concurrent=20)
    results = manager.execute_all()
    

* * *

## 3.6 Summary

**Key Points** :

  1. **SLURM** : Widely used job scheduler for large-scale HPC
  2. **Array Jobs** : Efficiently execute many materials in parallel
  3. **MPI Parallelism** : Measure and optimize scaling efficiency
  4. **Python Management** : Automate and monitor large-scale calculations

**Next Steps** :

In Chapter 4, we will learn about **workflow management with FireWorks and AiiDA**.

**[Chapter 4: Data Management and Post-Processing →](<chapter-4.html>)**

* * *

**License** : CC BY 4.0 **Date Created** : 2025-10-17 **Author** : Dr. Yusuke Hashimoto, Tohoku University
