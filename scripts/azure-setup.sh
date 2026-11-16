#!/usr/bin/env bash
# One-time Azure setup for the heart-risk MLOps project (run in Git Bash after `az login`).
#
#   1. Resource group + Azure ML workspace (francecentral)
#   2. Versioned data asset heart-disease-2022:1 (the raw CSV, uploaded once)
#   3. App registration + service principal for GitHub Actions, with an OIDC
#      federated credential (no client secret stored anywhere)
#   4. Least-privilege roles for that service principal on the workspace only
#
# The compute cluster, environments and components are NOT created here:
# the GitHub Actions pipeline creates/updates them (infrastructure as code).
set -euo pipefail
export MSYS_NO_PATHCONV=1   # Git Bash: don't rewrite "/subscriptions/..." into a Windows path

LOCATION="${LOCATION:-francecentral}"
GROUP="${GROUP:-mlops-home-project}"
WORKSPACE="${WORKSPACE:-hertveldt-daan-heart}"
GITHUB_REPO="${GITHUB_REPO:-hertveldtdaan/heart-risk-mlops}"   # owner/repo
GITHUB_ENVIRONMENT="${GITHUB_ENVIRONMENT:-azure-ml}"
APP_NAME="${APP_NAME:-github-heart-risk-mlops}"
DATA_FILE="${DATA_FILE:-data/raw/heart_2022_with_nans.csv}"

SUBSCRIPTION_ID=$(az account show --query id -o tsv)
TENANT_ID=$(az account show --query tenantId -o tsv)
echo "Subscription: $(az account show --query name -o tsv) ($SUBSCRIPTION_ID)"

az extension add --name ml --upgrade -y >/dev/null

echo "== 1. Resource group + workspace"
az group create --name "$GROUP" --location "$LOCATION" -o none
if ! az ml workspace show --name "$WORKSPACE" --resource-group "$GROUP" -o none 2>/dev/null; then
  az ml workspace create --name "$WORKSPACE" --resource-group "$GROUP" --location "$LOCATION" -o none
fi
az configure --defaults group="$GROUP" workspace="$WORKSPACE" location="$LOCATION"
WORKSPACE_ID=$(az ml workspace show --name "$WORKSPACE" --query id -o tsv)

echo "== 2. Data asset heart-disease-2022:1"
if ! az ml data show --name heart-disease-2022 --version 1 -o none 2>/dev/null; then
  az ml data create --name heart-disease-2022 --version 1 --type uri_file \
    --path "$DATA_FILE" \
    --description "CDC BRFSS 2022 - Personal Key Indicators of Heart Disease (Kaggle kamilpytlak), raw with NaNs" \
    --set tags.source=kaggle/kamilpytlak/personal-key-indicators-of-heart-disease tags.year=2022 -o none
fi

echo "== 3. App registration + service principal + OIDC federated credential"
APP_ID=$(az ad app list --display-name "$APP_NAME" --query "[0].appId" -o tsv)
if [ -z "$APP_ID" ]; then
  APP_ID=$(az ad app create --display-name "$APP_NAME" --query appId -o tsv)
fi
az ad sp show --id "$APP_ID" -o none 2>/dev/null || az ad sp create --id "$APP_ID" -o none
SP_OBJECT_ID=$(az ad sp show --id "$APP_ID" --query id -o tsv)

# GitHub now sends an "immutable" subject with the numeric owner and repo IDs:
#   repo:Owner@<owner_id>/repo@<repo_id>:environment:<env>
# (it doesn't change when the account or repo is renamed). The match is case-sensitive.
REPO_INFO=$(curl -s "https://api.github.com/repos/${GITHUB_REPO}")
OWNER=$(echo "$REPO_INFO" | python -c "import json,sys; print(json.load(sys.stdin)['owner']['login'])")
OWNER_ID=$(echo "$REPO_INFO" | python -c "import json,sys; print(json.load(sys.stdin)['owner']['id'])")
REPO=$(echo "$REPO_INFO" | python -c "import json,sys; print(json.load(sys.stdin)['name'])")
REPO_ID=$(echo "$REPO_INFO" | python -c "import json,sys; print(json.load(sys.stdin)['id'])")
SUBJECT="repo:${OWNER}@${OWNER_ID}/${REPO}@${REPO_ID}:environment:${GITHUB_ENVIRONMENT}"

CRED_NAME="github-${GITHUB_ENVIRONMENT}-immutable-ids"
if ! az ad app federated-credential show --id "$APP_ID" --federated-credential-id "$CRED_NAME" -o none 2>/dev/null; then
  az ad app federated-credential create --id "$APP_ID" --parameters "{
    \"name\": \"$CRED_NAME\",
    \"issuer\": \"https://token.actions.githubusercontent.com\",
    \"subject\": \"$SUBJECT\",
    \"audiences\": [\"api://AzureADTokenExchange\"]
  }" -o none
fi
echo "Federated credential subject: $SUBJECT"

echo "== 4. Roles on the workspace (least privilege, nothing on the subscription)"
for ROLE in "AzureML Data Scientist" "AzureML Compute Operator"; do
  az role assignment create --assignee-object-id "$SP_OBJECT_ID" --assignee-principal-type ServicePrincipal \
    --role "$ROLE" --scope "$WORKSPACE_ID" -o none
done

cat <<EOF

Done. Add these VARIABLES to the GitHub environment '$GITHUB_ENVIRONMENT'
(Settings -> Environments -> $GITHUB_ENVIRONMENT -> Environment variables):

  AZURE_CLIENT_ID        $APP_ID
  AZURE_TENANT_ID        $TENANT_ID
  AZURE_SUBSCRIPTION_ID  $SUBSCRIPTION_ID
  AZURE_RESOURCE_GROUP   $GROUP
  AZURE_ML_WORKSPACE     $WORKSPACE
  AZURE_LOCATION         $LOCATION

And a repository SECRET (Settings -> Secrets and variables -> Actions):
  POSTGRES_PASSWORD      <a strong password of your choice>
EOF
