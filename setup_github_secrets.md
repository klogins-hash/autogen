# GitHub Secrets Setup for Automated Deployment

## Required GitHub Secrets

To enable automated deployment, add these secrets to your GitHub repository:

### 1. SSH Configuration Secrets

Go to: `https://github.com/klogins-hash/autogen/settings/secrets/actions`

Add these secrets:

```
SSH_HOST = 206.189.203.174
SSH_USER = root
SSH_PRIVATE_KEY = [Your SSH private key content]
```

### 2. Get Your SSH Private Key

```bash
# Display your SSH private key
cat ~/.ssh/hitsdifferent_key

# Copy the entire output (including -----BEGIN and -----END lines)
```

### 3. Add Secrets via GitHub CLI (Alternative)

```bash
# Set SSH host
gh secret set SSH_HOST --body "206.189.203.174"

# Set SSH user
gh secret set SSH_USER --body "root"

# Set SSH private key (interactive)
gh secret set SSH_PRIVATE_KEY < ~/.ssh/hitsdifferent_key
```

## Deployment Triggers

The automated deployment will trigger on:

- ✅ **Push to main branch** - Full deployment with testing
- ✅ **Push to feature/enhanced-magentic-one** - Development deployment
- ✅ **Pull request to main** - Testing only (no deployment)
- ✅ **Changes to enhanced components** - Selective deployment

## Deployment Process

### Automatic Steps:
1. **Test Suite** - Run all tests across Python versions
2. **Build Package** - Create deployment package
3. **SSH Deploy** - Deploy to your server automatically
4. **Verify** - Test deployment on remote server
5. **Notify** - Report deployment status

### What Gets Deployed:
- Enhanced MCP server
- All autogen-ext components
- Updated dependencies
- Test scripts
- Documentation

## Manual Deployment Override

If you need to deploy manually:

```bash
# Trigger deployment workflow
gh workflow run deploy-enhanced-magentic-one.yml

# Or push to trigger automatic deployment
git push origin feature/enhanced-magentic-one
```

## Monitoring Deployments

View deployment status:
- GitHub Actions: `https://github.com/klogins-hash/autogen/actions`
- SSH Server: `ssh hitsdifferent "ls -la ~/enhanced-magentic-one/"`

## Rollback Process

If deployment fails, automatic rollback:
```bash
ssh hitsdifferent << 'EOF'
cd ~/enhanced-magentic-one
rm -rf current
mv backup current
ln -sf enhanced-magentic-one/current ~/magentic-mcp
EOF
```
