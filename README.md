# 🎨 Academic Project Page Template

[![build](https://github.com/denkiwakame/academic-project-template/actions/workflows/build.yaml/badge.svg)](https://github.com/denkiwakame/academic-project-template/actions/workflows/build.yaml) [![lint](https://github.com/denkiwakame/academic-project-template/actions/workflows/lint.yaml/badge.svg)](https://github.com/denkiwakame/academic-project-template/actions/workflows/lint.yaml)
[![license](https://img.shields.io/badge/LICENSE-CC--BY--SA4.0-important.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

## ✨ Features

- 🚀 Built with [React](https://react.dev/) + [UIKit](https://getuikit.com/)
- 📝 Just one file! Write all metadata & content in [template.yaml](template.yaml)
- 📝 Markdown support with full styling capabilities
- ➗ Mathematical equations with KaTeX
- 🎨 Customizable themes
- 🔄 Auto-deployment with GitHub Actions
- 📱 Fully responsive design

🌐 **Demo**: [default-theme](https://denkiwakame.github.io/academic-project-template) | [classic-theme](https://github.com/user-attachments/assets/b8bc1e19-01b1-41a8-aff7-69047bb6ece2)

### ⚙️ Configuration

Simply define everything in **a single `template.yaml` file**—**metadata**, **Markdown**, **HTML**, and even **KaTeX** equations. No extra setup required!

```yaml
theme: default # default || dark
organization: 'Your Lab'
title: 'Project Title'
resources:
  paper: https://example.com/paper
  code: https://github.com/your/repo
# ... other configurations
```
With just this file, your project page is ready to go! 🚀

## 🚦 Prerequisites

### System Requirements

Supported platforms:

- 🪟 Windows (WSL)
- 🐧 Linux
- 🍎 macOS

### 📦 Node.js Setup

Recommended: Install Node.js using [Volta](https://volta.sh/)

```bash
# Install Volta
curl https://get.volta.sh/ | bash

# After restarting your shell...
volta install node@22.13.1
volta pin node@22.13.1

# Verify installation
node --version  # v22.13.1
npm --version   # 10.9.2
```

## 🛠️ Development Guide

### 📥 Installation

```sh
npm install
```

### 💻 Development Server

```sh
npm run dev
```

### 🏗️ Build

```sh
npm run clean
npm run build
npm run preview
```

#### 🤖 Puppeteer Dependencies (for react-snap pre-rendering)

Required packages for Headless Chrome (pre-rendering):

```bash
sudo apt install -y libgtk2.0-0 libgtk-3-0 libnotify-dev libgconf-2-4 libnss3 libxss1 libasound2 libxtst6 xauth xvfb libgbm-dev fonts-ipafont
```

## 🎨 Customization

- 💅 Customize styles without writing CSS: modify [UIKit variables](https://github.com/uikit/uikit/blob/develop/src/scss/variables.scss) in `src/scss/theme.scss`
- 🧩 Extend components with:
  - [UIKit Components](https://getuikit.com/docs/introduction)
  - [React Icons](https://react-icons.github.io/react-icons/)

## 📁 Project Structure

```
template.yaml         # Configuration file
index.html            # Root HTML
src/
├── components/       # React components
├── js/
│   └── styles.js     # Style settings
├── pages/
│   └── index.jsx     # Main template
└── scss/             # Theme settings
    └── theme.scss
public/
└── 001.png           # Media files
```

## 🚀 Release Your Project Page

### Automatic Deployment with GitHub Actions

1. Go to your repository's Settings > Pages
2. Under Build and Deployment > Source, select `GitHub Actions`
3. Configure your remote repository:

```bash
git remote add origin your-repo-url
git push origin project-page
```

> [!NOTE]
> The implementation of the paper is published in the main branch, and the code for the project page is expected to be pushed to a separate branch.
> If you want to change the branch settings, please modify `.github/workflows/deploy.yaml`.

🚨 TroubleShooting

<details>
<summary>Branch "project-page" is not allowed to deploy to github-pages due to environment protection rules</summary>
Navigate to Settings > Environments > github-pages > 🗑️
or set rules properly (add `project-page` to deployable branch).
https://docs.github.com/ja/actions/managing-workflow-runs-and-deployments/managing-deployments/managing-environments-for-deployment#deployment-protection-rules
  
![image](https://github.com/user-attachments/assets/ddaa751d-cedc-4665-86a1-8afd88e04e52)

</details>

📚 For more details:

- [GitHub Pages Documentation](https://docs.github.com/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)
- [actions-deploy-pages](https://github.com/actions/deploy-pages)

## 🐶 Husky & Typo Checking

### Automatic Typo Checking

This project uses Husky to automatically check for typos during commits. The check is performed using the `typos` tool and only runs on staged files (files that have been `git add`ed).
If you want to ignore specific typos as a dictionary, please add the original and corrected versions to `_typos.toml`. ([doc](https://github.com/crate-ci/typos/tree/master?tab=readme-ov-file#false-positives))


```bash
[default.extend-words]
# Don't correct the surname "Teh"
teh = "teh"
```

### Manual Typo Checking

To manually check for typos without committing:

```bash
npx typos
```

This command checks all files in the project, regardless of whether they are staged for commit or not.

### Disabling Git Hooks

To temporarily disable all git hooks (including lint, format, and typo validation) during commit:

```bash
export HUSKY=0
```

You can re-enable hooks by unsetting the variable or starting a new terminal session.

## 📄 License

[CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
