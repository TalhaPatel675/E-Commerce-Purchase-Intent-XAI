# AWS deployment notes

## 1. Container registry (ECR)
```powershell
aws ecr create-repository --repository-name purchase-intent-api --region us-east-1
aws ecr create-repository --repository-name purchase-intent-dashboard --region us-east-1
$ACC = (Get-STSCallerIdentity).Account
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$ACC.dkr.ecr.us-east-1.amazonaws.com"
docker build -f docker/Dockerfile.api -t purchase-intent-api .
docker tag purchase-intent-api:latest "$ACC.dkr.ecr.us-east-1.amazonaws.com/purchase-intent-api:latest"
docker push "$ACC.dkr.ecr.us-east-1.amazonaws.com/purchase-intent-api:latest"
```

## 2. ECS Fargate
* Cluster: `purchase-intent-cluster`
* Two services, container port 8000 (api) and 8501 (dashboard)
* Use `aws/task-definition-template.json` for the api container — replace `REPLACE_WITH_ECR_URI`, subnet id, security-group id.
* Set environment variables from `.env` (do NOT commit real secrets).

## 3. IAM minimum permissions
* `AmazonEC2ContainerRegistryReadOnly` for ECS task execution role
* `AmazonS3ReadOnlyAccess` if the dashboard stream model artifacts from S3
* Nothing else committed to source.

## 4. EC2 + compose alternative
```sh
sudo apt update && sudo apt install -y docker.io docker-compose-plugin
git clone <repo> && cd purchase-intent-xai
docker compose up --build
```

## 5. libgomp1 Windows reminders
The Dockerfile installs `libgomp1` (LightGBM system dep). On Windows installs, the same lib is bundled in modern `lightgbm` wheels but the **Microsoft Visual C++ Redistributable 2015-2022** must be present: <https://aka.ms/vs/17/release/vc_redist.x64.exe>.
