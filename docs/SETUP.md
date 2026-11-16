# Eenmalige setup

## 1. GitHub-repo

1. Maak een nieuwe repo `hertveldtdaan/heart-risk-mlops` aan op GitHub (zonder README).
2. In deze map:
   ```bash
   git init -b main
   git add . && git commit -m "Heart risk MLOps project"
   git remote add origin https://github.com/hertveldtdaan/heart-risk-mlops.git
   git push -u origin main
   ```
   `data/`, `model/`, `outputs/` en `.venv/` staan in `.gitignore`. De dataset (140 MB) gaat naar Azure ML als data asset, niet naar git.

## 2. Azure (nieuwe resource group en workspace)

```bash
az login
bash scripts/azure-setup.sh
```

Het script maakt aan:

- resource group `mlops-home-project` en workspace `hertveldt-daan-heart` (francecentral);
- data asset `heart-disease-2022:1` (upload van de CSV);
- app registration `github-heart-risk-mlops` met een service principal en een **OIDC federated credential** (geen client secret). Let op: GitHub stuurt het subject in het "immutable" formaat met numerieke ID's, `repo:HertveldtDaan@<owner_id>/heart-risk-mlops@<repo_id>:environment:azure-ml`, en Entra ID vergelijkt hoofdlettergevoelig. Het script haalt die ID's op via de GitHub-API. De repo moet dus al bestaan wanneer je het script draait;
- de rollen *AzureML Data Scientist* en *AzureML Compute Operator*, enkel op de workspace.

Andere namen kies je met omgevingsvariabelen, bv. `GITHUB_REPO=owner/naam bash scripts/azure-setup.sh`.

## 3. GitHub-instellingen

- **Settings → Environments → New environment `azure-ml`**, met de 6 variabelen die het script op het einde toont (`AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`, `AZURE_ML_WORKSPACE`, `AZURE_LOCATION`).
- **Settings → Secrets and variables → Actions → New repository secret** `POSTGRES_PASSWORD`.
- **Settings → Actions → General → Workflow permissions**: *Read and write* (voor het pushen naar GHCR).

## 4. Self-hosted runner (deploy-job)

Je runner `daan-pc` is aan de labo-repo gekoppeld. Registreer daarom een tweede runner voor deze repo, in een aparte map:

1. Repo → Settings → Actions → Runners → **New self-hosted runner** (Windows x64).
2. Voer de getoonde commando's uit in bv. `C:\actions-runner-heart`.
3. Start de runner met `.\run.cmd` voordat je een deploy-run start.

## 5. k3d-cluster

```bash
k3d cluster start mlops
k3d cluster edit mlops --port-add 30082:30082@loadbalancer   # NodePort van de Heart Risk API
```

Daarna is de API na een deploy bereikbaar op <http://localhost:30082/docs>.
