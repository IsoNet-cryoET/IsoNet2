# isoapp

An Electron application with React

## Recommended IDE Setup

- [VSCode](https://code.visualstudio.com/) + [ESLint](https://marketplace.visualstudio.com/items?itemName=dbaeumer.vscode-eslint) + [Prettier](https://marketplace.visualstudio.com/items?itemName=esbenp.prettier-vscode)

## Project Setup

### Install

```bash
$ npm install
```

### Development

```bash
$ npm run dev
```

### Build

```bash
# For windows
$ npm run build:win

# For macOS
$ npm run build:mac

# For Linux
$ npm run build:linux
```

## Release packaging

The GUI follows the IsoNet2 release version in the root `pyproject.toml`; the internal package metadata uses the same version. The public Linux GUI asset is named `isoapp.AppImage`, without a separate GUI version in its filename.

Pushing an `IsoNet-X.Y.Z` tag runs `.github/workflows/release.yml`, builds the GUI from that tag, uploads the AppImage and its SHA256 checksum to a draft release, then publishes it. Release notes come from the tagged `RELEASE_NOTES.md`. The workflow can also be run manually for an existing tag.
