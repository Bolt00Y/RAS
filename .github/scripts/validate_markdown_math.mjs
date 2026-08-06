#!/usr/bin/env node

/**
 * Validate GitHub Markdown `math` fenced blocks with MathJax.
 *
 * GitHub renders Markdown mathematics with MathJax. This checker uses the same
 * TeX engine family and treats MathJax error output as a CI failure.
 */

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { mathjax } from "mathjax-full/js/mathjax.js";
import { TeX } from "mathjax-full/js/input/tex.js";
import { SVG } from "mathjax-full/js/output/svg.js";
import { liteAdaptor } from "mathjax-full/js/adaptors/liteAdaptor.js";
import { RegisterHTMLHandler } from "mathjax-full/js/handlers/html.js";
import { AllPackages } from "mathjax-full/js/input/tex/AllPackages.js";

const root = path.resolve(process.argv[2] ?? ".");
const skippedDirectories = new Set([
  ".git",
  "node_modules",
  ".venv",
  "venv",
  "dist",
  "build",
]);
const legacyDelimiters = ["\\(", "\\)", "\\[", "\\]"];

function walkMarkdown(directory) {
  const files = [];
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    if (entry.isDirectory() && skippedDirectories.has(entry.name)) continue;
    const fullPath = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...walkMarkdown(fullPath));
    else if (entry.isFile() && entry.name.endsWith(".md")) files.push(fullPath);
  }
  return files.sort();
}

function fenceMatch(line) {
  return line.match(/^(\s*)(`{3,}|~{3,})(.*)$/);
}

function extractMathBlocks(filePath) {
  const source = fs.readFileSync(filePath, "utf8");
  const lines = source.split(/\r?\n/);
  const blocks = [];
  const errors = [];

  let openFence = null;
  let openInfo = "";
  let openLine = 0;
  let buffer = [];

  for (let index = 0; index < lines.length; index += 1) {
    const lineNumber = index + 1;
    const line = lines[index];
    const match = fenceMatch(line);

    if (openFence !== null) {
      const isClosingFence =
        match !== null &&
        match[2][0] === openFence[0] &&
        match[2].length >= openFence.length &&
        match[3].trim() === "";

      if (isClosingFence) {
        if (openInfo === "math") {
          blocks.push({
            expression: buffer.join("\n").trim(),
            line: openLine,
          });
        }
        openFence = null;
        openInfo = "";
        openLine = 0;
        buffer = [];
      } else if (openInfo === "math") {
        buffer.push(line);
      }
      continue;
    }

    if (match !== null) {
      openFence = match[2];
      openInfo = match[3].trim().toLowerCase();
      openLine = lineNumber;
      buffer = [];
      continue;
    }

    if (line.trim() === "$$") {
      errors.push(`${filePath}:${lineNumber}: standalone double-dollar delimiter is forbidden; use a math fence`);
    }
    for (const delimiter of legacyDelimiters) {
      if (line.includes(delimiter)) {
        errors.push(`${filePath}:${lineNumber}: legacy delimiter ${JSON.stringify(delimiter)} is forbidden`);
      }
    }
  }

  if (openFence !== null) {
    errors.push(`${filePath}:${openLine}: unclosed fenced block`);
  }

  for (const block of blocks) {
    if (block.expression.length === 0) {
      errors.push(`${filePath}:${block.line}: empty math fenced block`);
    }
  }

  return { blocks, errors };
}

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const tex = new TeX({ packages: AllPackages });
const svg = new SVG({ fontCache: "none" });
const document = mathjax.document("", { InputJax: tex, OutputJax: svg });

function validateExpression(expression) {
  try {
    const node = document.convert(expression, { display: true });
    const rendered = adaptor.outerHTML(node);
    const hasMathJaxError =
      /<merror\b/i.test(rendered) ||
      /data-mjx-error/i.test(rendered) ||
      /data-mml-node="mtext"[^>]*(?:fill|stroke)="red"/i.test(rendered) ||
      /(?:fill|stroke)="red"[^>]*data-mml-node="mtext"/i.test(rendered);
    return hasMathJaxError ? "MathJax produced an error node" : null;
  } catch (error) {
    return error instanceof Error ? error.message : String(error);
  }
}

const markdownFiles = walkMarkdown(root);
const failures = [];
let formulaCount = 0;

for (const filePath of markdownFiles) {
  const relativePath = path.relative(root, filePath);
  const { blocks, errors } = extractMathBlocks(filePath);
  failures.push(...errors.map((message) => message.replace(filePath, relativePath)));

  for (const block of blocks) {
    formulaCount += 1;
    const error = validateExpression(block.expression);
    if (error !== null) {
      failures.push(`${relativePath}:${block.line}: ${error}\n${block.expression}`);
    }
  }
}

if (failures.length > 0) {
  console.error("GitHub Markdown math validation failed:");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log(
  `Validated ${formulaCount} math fenced block(s) across ${markdownFiles.length} Markdown file(s).`,
);
