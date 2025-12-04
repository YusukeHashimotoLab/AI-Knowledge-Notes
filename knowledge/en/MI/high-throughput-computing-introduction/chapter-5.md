---
title: "Chapter 5: Cloud HPC Utilization and Optimization"
chapter_title: "Chapter 5: Cloud HPC Utilization and Optimization"
subtitle: 
reading_time: 15-20 min
difficulty: Intermediate~Advanced
code_examples: 5
exercises: 0
version: 1.0
created_at: 2025-10-17
---

# Chapter 5: Cloud HPC Utilization and Optimization

Learn the practice of data-based "reproducible research" using NOMAD and DVC. Master the principles of metadata design and publication.

**💡 Supplement:** Recording "who, when, and how" data was created facilitates future verification. Perform anonymization and rights confirmation before publication.

## Learning Objectives

By reading this chapter, you will be able to:

  * ✅ Build AWS Parallel Cluster
  * ✅ Reduce costs by 50% with spot instances
  * ✅ Fully reproduce computational environments with Docker
  * ✅ Design and execute 10,000-material scale projects
  * ✅ Understand security and compliance

* * *

## 5.1 Cloud HPC Options

### Major Cloud Provider Comparison

Service | Provider | Features | Initial Cost | Recommended Use  
---|---|---|---|---  
**AWS Parallel Cluster** | Amazon | Largest scale, rich track record | $0 | Large-scale HPC  
**Google Cloud HPC Toolkit** | Google | Strong AI/ML integration | $0 | Machine learning integration  
**Azure CycleCloud** | Microsoft | Windows integration | $0 | Enterprise  
**TSUBAME** | Tokyo Tech | Top in Japan, academic use | Application-based | Academic research  
**Fugaku** | RIKEN | World Top 500 | Application-based | Ultra-large-scale computing  
  
### Cost Comparison (10,000 materials, 1 material = 12 CPU hours, 48 cores)

Option | Computing Time | Cost | Benefits | Drawbacks  
---|---|---|---|---  
On-premise HPC | 5,760,000 CPU hours | $0 (existing facility) | Free (within allocation) | Wait time, limitations  
AWS On-demand | Same as above | $4,000-6,000 | Immediately available | High cost  
AWS Spot | Same as above | $800-1,500 | 70% cost reduction | Interruption risk  
Google Cloud Preemptible | Same as above | $900-1,600 | Low cost | 24-hour limit  
  
* * *

## 5.2 AWS Parallel Cluster Setup

### Prerequisites
    
    
    # AWS CLI installation
    pip install awscli
    
    # AWS configuration
    aws configure
    # AWS Access Key ID: [YOUR_KEY]
    # AWS Secret Access Key: [YOUR_SECRET]
    # Default region: us-east-1
    # Default output format: json
    
    # Parallel Cluster CLI installation
    pip install aws-parallelcluster
    

### Cluster Configuration File

**config.yaml** :
    
    
    Region: us-east-1
    Image:
      Os: alinux2
    
    HeadNode:
      InstanceType: c5.2xlarge  # 8 vCPU, 16 GB RAM
      Networking:
        SubnetId: subnet-12345678
      Ssh:
        KeyName: my-key-pair
    
    Scheduling:
      Scheduler: slurm
      SlurmQueues:
        - Name: compute
          ComputeResources:
            - Name: c5-48xlarge
              InstanceType: c5.24xlarge  # 96 vCPU
              MinCount: 0
              MaxCount: 100  # Maximum 100 nodes
              DisableSimultaneousMultithreading: true
              Efa:
                Enabled: true  # High-speed network
          Networking:
            SubnetIds:
              - subnet-12345678
            PlacementGroup:
              Enabled: true  # Low-latency placement
    
    SharedStorage:
      - MountDir: /shared
        Name: ebs-shared
        StorageType: Ebs
        EbsSettings:
          VolumeType: gp3
          Size: 1000  # 1 TB
          Encrypted: true
    
      - MountDir: /fsx
        Name: lustre-fs
        StorageType: FsxLustre
        FsxLustreSettings:
          StorageCapacity: 1200  # 1.2 TB
          DeploymentType: SCRATCH_2
    

### Cluster Creation
    
    
    # Create cluster
    pcluster create-cluster \
      --cluster-name vasp-cluster \
      --cluster-configuration config.yaml
    
    # Check creation status
    pcluster describe-cluster --cluster-name vasp-cluster
    
    # SSH connection
    pcluster ssh --cluster-name vasp-cluster -i ~/.ssh/my-key-pair.pem
    

### VASP Environment Setup
    
    
    # After SSH connection to cluster
    
    # Intel OneAPI Toolkit (required for VASP compilation)
    wget https://registrationcenter-download.intel.com/...
    bash l_BaseKit_p_2023.0.0.25537_offline.sh
    
    # VASP compilation (license required)
    cd /shared
    tar -xzf vasp.6.3.0.tar.gz
    cd vasp.6.3.0
    
    # Edit makefile.include (for Intel compiler)
    cp arch/makefile.include.intel makefile.include
    
    # Compile
    make all
    
    # Place executable in shared directory
    cp bin/vasp_std /shared/bin/
    

* * *

## 5.3 Cost Optimization

### Utilizing Spot Instances

**Spot instances** are surplus computing resources available at 60-90% discount from on-demand pricing.

**config.yaml (Spot configuration)** :
    
    
    Scheduling:
      Scheduler: slurm
      SlurmQueues:
        - Name: spot-queue
          CapacityType: SPOT  # Spot instances
          ComputeResources:
            - Name: c5-spot
              InstanceType: c5.24xlarge
              MinCount: 0
              MaxCount: 200
              SpotPrice: 2.50  # Maximum bid price ($/hour)
          Networking:
            SubnetIds:
              - subnet-12345678
    

**Spot Instance Best Practices** :

  1. **Checkpoint** : Save calculations periodically
  2. **Multiple instance types** : Specify alternative types
  3. **Retry configuration** : Automatic restart on interruption

### Auto Scaling
    
    
    Scheduling:
      SlurmSettings:
        ScaledownIdletime: 5  # Terminate after 5 min idle
      SlurmQueues:
        - Name: compute
          ComputeResources:
            - Name: c5-instances
              MinCount: 0      # 0 nodes when idle
              MaxCount: 100    # Maximum 100 nodes
    

### Cost Monitoring
    
    
    import boto3
    from datetime import datetime, timedelta
    
    def get_cluster_cost(cluster_name, days=7):
        """
        Get cluster cost
    
        Parameters:
        -----------
        cluster_name : str
            Cluster name
        days : int
            Number of days back
    
        Returns:
        --------
        cost : float
            Total cost (USD)
        """
        ce_client = boto3.client('ce', region_name='us-east-1')
    
        # Set time period
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
    
        # Cost Explorer API
        response = ce_client.get_cost_and_usage(
            TimePeriod={
                'Start': start_date.strftime('%Y-%m-%d'),
                'End': end_date.strftime('%Y-%m-%d')
            },
            Granularity='DAILY',
            Metrics=['UnblendedCost'],
            Filter={
                'Tags': {
                    'Key': 'parallelcluster:cluster-name',
                    'Values': [cluster_name]
                }
            }
        )
    
        total_cost = 0
        for result in response['ResultsByTime']:
            cost = float(result['Total']['UnblendedCost']['Amount'])
            total_cost += cost
            print(f"{result['TimePeriod']['Start']}: ${cost:.2f}")
    
        print(f"\nTotal cost ({days} days): ${total_cost:.2f}")
        return total_cost
    
    # Usage example
    get_cluster_cost('vasp-cluster', days=7)
    

* * *

## 5.4 Docker/Singularity Containerization

### Creating Dockerfile

**Dockerfile** :
    
    
    FROM ubuntu:20.04
    
    # Basic packages
    RUN apt-get update && apt-get install -y \
        build-essential \
        gfortran \
        openmpi-bin \
        libopenmpi-dev \
        python3 \
        python3-pip \
        wget \
        && rm -rf /var/lib/apt/lists/*
    
    # Python environment
    RUN pip3 install --upgrade pip && \
        pip3 install numpy scipy matplotlib \
        ase pymatgen fireworks
    
    # Intel MKL (numerical computation library)
    RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && \
        apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && \
        echo "deb https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list && \
        apt-get update && \
        apt-get install -y intel-oneapi-mkl
    
    # VASP (license holders only)
    # COPY vasp.6.3.0.tar.gz /tmp/
    # RUN cd /tmp && tar -xzf vasp.6.3.0.tar.gz && \
    #     cd vasp.6.3.0 && make all && \
    #     cp bin/vasp_std /usr/local/bin/
    
    # Working directory
    WORKDIR /calculations
    
    # Default command
    CMD ["/bin/bash"]
    

### Docker Image Build and Push
    
    
    # Build image
    docker build -t my-vasp-env:latest .
    
    # Push to Docker Hub (for sharing)
    docker tag my-vasp-env:latest username/my-vasp-env:latest
    docker push username/my-vasp-env:latest
    
    # Push to Amazon ECR (for AWS)
    aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com
    
    docker tag my-vasp-env:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/my-vasp-env:latest
    docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/my-vasp-env:latest
    

### Executing with Singularity on HPC

HPC systems use Singularity instead of Docker.
    
    
    # Create Singularity image from Docker image
    singularity build vasp-env.sif docker://username/my-vasp-env:latest
    
    # Execute with Singularity container
    singularity exec vasp-env.sif mpirun -np 48 vasp_std
    

**SLURM script (using Singularity)** :
    
    
    #!/bin/bash
    #SBATCH --job-name=vasp-singularity
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=48
    #SBATCH --time=24:00:00
    
    # Singularity image
    IMAGE=/shared/containers/vasp-env.sif
    
    # Execute VASP inside container
    singularity exec $IMAGE mpirun -np 48 vasp_std
    

* * *

## 5.5 Security and Compliance

### Access Control (IAM)

**Principle of Least Privilege** :
    
    
    {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Effect": "Allow",
          "Action": [
            "ec2:DescribeInstances",
            "ec2:DescribeVolumes",
            "ec2:RunInstances",
            "ec2:TerminateInstances"
          ],
          "Resource": "*",
          "Condition": {
            "StringEquals": {
              "aws:RequestedRegion": "us-east-1"
            }
          }
        },
        {
          "Effect": "Allow",
          "Action": [
            "s3:GetObject",
            "s3:PutObject"
          ],
          "Resource": "arn:aws:s3:::my-vasp-bucket/*"
        }
      ]
    }
    

### Data Encryption
    
    
    # config.yaml (encryption configuration)
    SharedStorage:
      - MountDir: /shared
        Name: ebs-encrypted
        StorageType: Ebs
        EbsSettings:
          VolumeType: gp3
          Size: 1000
          Encrypted: true  # Encryption enabled
          KmsKeyId: arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012
    

### Academic License Compliance

Considerations when using commercial software like **VASP** on the cloud:

  1. **License verification** : Check if cloud usage is permitted
  2. **Node locking** : Restrictions on execution on specific nodes
  3. **Concurrent execution limits** : License count restrictions
  4. **Audit logs** : Record usage history

* * *

## 5.6 Case Study: 10,000 Material Screening

### Requirements Definition

**Objective** : Calculate band gaps of 10,000 oxide materials within 6 months

**Constraints** : \- Budget: $5,000 \- Computation time per material: 12 hours (48 cores) \- Total CPU time: 5,760,000 CPU hours

### Architecture Design
    
    
    ```mermaid
    flowchart TD
        A["Material List10,000 items"] --> B["Batch Division100 batches × 100 materials"]
        B --> C["AWS Parallel ClusterSpot Instances"]
        C --> D["SLURM Array Jobs20 concurrent nodes"]
        D --> E["FireWorksWorkflow Management"]
        E --> F["MongoDBResult Storage"]
        F --> G["S3Long-term Storage"]
        G --> H["Analysis & Visualization"]
    ```

### Implementation

**1\. Cluster Configuration**
    
    
    # config-10k-materials.yaml
    Scheduling:
      Scheduler: slurm
      SlurmQueues:
        - Name: spot-compute
          CapacityType: SPOT
          ComputeResources:
            - Name: c5-24xlarge-spot
              InstanceType: c5.24xlarge  # 96 vCPU
              MinCount: 0
              MaxCount: 50  # 50 concurrent nodes = 4,800 cores
              SpotPrice: 2.00
    

**2\. Job Submission Script**
    
    
    def run_10k_project():
        """Execute 10,000 material project"""
    
        # Load material list
        with open('oxide_materials_10k.txt', 'r') as f:
            materials = [line.strip() for line in f]
    
        print(f"Total materials: {len(materials)}")
    
        # Divide into 100 batches
        batch_size = 100
        n_batches = len(materials) // batch_size
    
        manager = SLURMJobManager()
    
        for batch_id in range(n_batches):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            batch_materials = materials[start:end]
    
            # Material list for batch
            list_file = f'batch_{batch_id:03d}.txt'
            with open(list_file, 'w') as f:
                for mat in batch_materials:
                    f.write(f"{mat}\n")
    
            # Submit array job (100 materials, 20 concurrent nodes)
            job_id = manager.submit_array_job(
                'vasp_bandgap.sh',
                n_tasks=100,
                max_concurrent=20
            )
    
            print(f"Batch {batch_id+1}/{n_batches} submitted: Job ID {job_id}")
    
            # Rate limiting (for AWS API limits)
            time.sleep(1)
    
    run_10k_project()
    

**3\. Cost Analysis**
    
    
    def estimate_project_cost():
        """Estimate project cost"""
    
        # Parameters
        n_materials = 10000
        cpu_hours_per_material = 12
        cores_per_job = 48
    
        total_cpu_hours = n_materials * cpu_hours_per_material
    
        # c5.24xlarge: 96 vCPU, $4.08/hour (on-demand)
        ondemand_cost = total_cpu_hours * (4.08 / 96)
        print(f"On-demand: ${ondemand_cost:,.0f}")
    
        # Spot: 70% discount
        spot_cost = ondemand_cost * 0.3
        print(f"Spot: ${spot_cost:,.0f}")
    
        # Storage: EBS 1TB × 6 months
        storage_cost = 0.10 * 1000 * 6  # $0.10/GB/month
        print(f"Storage: ${storage_cost:,.0f}")
    
        # Data transfer: 500GB
        transfer_cost = 500 * 0.09
        print(f"Data transfer: ${transfer_cost:,.0f}")
    
        total_cost = spot_cost + storage_cost + transfer_cost
        print(f"\nTotal cost: ${total_cost:,.0f}")
    
        return total_cost
    
    estimate_project_cost()
    

**Output** :
    
    
    On-demand: $5,100
    Spot: $1,530
    Storage: $600
    Data transfer: $45
    
    Total cost: $2,175
    

* * *

## 5.7 Exercises

### Problem 1 (Difficulty: medium)

**Question** : List three cost reduction strategies for AWS Parallel Cluster and estimate the reduction rate for each.

Sample Answer **1. Using Spot Instances** \- Reduction rate: 70% \- Risk: Possibility of interruption **2. Auto scale-down (5 min idle)** \- Reduction rate: 20-30% (depending on idle time) \- Risk: None **3. Reserved Instances (1-year contract)** \- Reduction rate: 40% \- Risk: Long-term commitment **Total reduction rate**: Maximum 85% (Spot + auto-scale) 

### Problem 2 (Difficulty: hard)

**Question** : For a 5,000 material project with a budget of $1,000 and 3-month timeframe, create an optimal execution plan.

Sample Answer **Parameters**: \- Materials: 5,000 \- CPU time: 5,000 × 12 hours = 60,000 hours \- Budget: $1,000 \- Duration: 3 months = 90 days **Working backwards from constraints**: Cost constraint: 
    
    
    $1,000 = Compute cost + Storage cost + Transfer cost
    $1,000 ≈ $800 (compute) + $150 (storage) + $50 (transfer)
    

c5.24xlarge spot: $1.22/hour 
    
    
    Available time = $800 / $1.22 = 656 hours
    Concurrent nodes = 60,000 / 656 / 24 = 3.8 ≈ 4 nodes
    

**Execution plan**: 1\. Spot instances: c5.24xlarge × 4 nodes 2\. Concurrent execution: 16 materials (48 cores each) 3\. Per day: 16 materials × 2 batches = 32 materials 4\. Completion time: 5,000 / 32 = 156 days **Issue**: Duration exceeds 3 months (90 days) **Solutions**: \- Increase concurrent nodes to 8 → Cost $1,600 (over budget) \- Or reduce CPU time per material to 8 hours (trade-off with accuracy) 

* * *

## 5.8 Summary

In this chapter, we learned about cloud HPC utilization and cost optimization.

**Key Points** :

  1. **AWS Parallel Cluster** : Easily build large-scale HPC environments
  2. **Spot Instances** : 70% cost reduction
  3. **Docker/Singularity** : Complete environment reproduction
  4. **Cost Management** : Estimation and monitoring
  5. **Security** : Encryption and access control

**Congratulations on completing the series!**

You have now completed all 5 chapters of High Throughput Computing. From fundamental concepts in Chapter 1 to cloud implementation in Chapter 5, you should have acquired practical skills.

**Next Steps** :

  1. Execute small-scale projects (100-1000 materials)
  2. Measure and optimize costs
  3. Publish results to NOMAD
  4. Write papers and present at conferences

**[Return to Series Top](<./index.html>)**

* * *

## References

  1. Amazon Web Services (2023). "AWS Parallel Cluster User Guide." https://docs.aws.amazon.com/parallelcluster/

  2. Kurtzer, G. M., et al. (2017). "Singularity: Scientific containers for mobility of compute." _PLOS ONE_ , 12(5), e0177459.

  3. Merkel, D. (2014). "Docker: lightweight Linux containers for consistent development and deployment." _Linux Journal_ , 2014(239), 2.

  4. NOMAD Laboratory (2023). "NOMAD Repository - FAIR Data Sharing." https://nomad-lab.eu/

  5. Jain, A., et al. (2015). "FireWorks: a dynamic workflow system designed for high-throughput applications." _Concurrency and Computation: Practice and Experience_ , 27(17), 5037-5059.

* * *

**License** : CC BY 4.0 **Created** : 2025-10-17 **Author** : Dr. Yusuke Hashimoto, Tohoku University

* * *

**You have completed the High Throughput Computing introduction series!**

We look forward to your research that accelerates materials discovery and contributes to the world.
