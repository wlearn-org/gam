'use strict'

const fs = require('fs')
const path = require('path')

const scriptsDir = __dirname
const pkgDir = path.resolve(scriptsDir, '..')
const repoDir = path.resolve(pkgDir, '..')
const srcDir = path.join(repoDir, 'src')
const csrcDir = path.join(pkgDir, 'csrc')

if (!fs.existsSync(srcDir)) {
  console.error(`sync-csrc: source directory not found: ${srcDir}`)
  process.exit(1)
}

const files = fs.readdirSync(srcDir)
  .filter((name) => /\.(c|h)$/.test(name))
  .sort()

if (files.length === 0) {
  console.error(`sync-csrc: no C sources found in ${srcDir}`)
  process.exit(1)
}

fs.rmSync(csrcDir, { recursive: true, force: true })
fs.mkdirSync(csrcDir, { recursive: true })

for (const name of files) {
  fs.copyFileSync(path.join(srcDir, name), path.join(csrcDir, name))
}

console.log(`sync-csrc: copied ${files.length} files to ${csrcDir}`)
