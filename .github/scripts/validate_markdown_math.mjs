#!/usr/bin/env node

/**
 * Validate Markdown mathematics with MathJax.
 *
 * New repository convention:
 *   - inline mathematics: $...$
 *   - display mathematics: standalone $$ delimiter lines
 *
 * Existing `math` fenced blocks are still parsed for backwards compatibility.
 * Both display and inline expressions are sent through MathJax and any error
 * node is treated as a CI failure.
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
const inlineMathPattern = /(?<!\\)\$(?!\$)(.+?)(?<!\\)\$/g;
const inlineCodePattern = /(?<!`)`[^`\n]+`(?!`)/g;

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

function extractMath(filePath) {
  const source = fs.readFileSync(filePath, "utf8");
  const lines = source.split(/\r?\n/);
  const expressions = [];
  const errors = [];

  let openFence = null;
  let openInfo = "";
  let openFenceLine = 0;
  let fenceBuffer = [];

  let displayOpen = false;
  let displayLine = 0;
  let displayBuffer = [];

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
          expressions.push({
            expression: fenceBuffer.join("\n").trim(),
            line: openFenceLine,
            display: true,
            kind: "math fence",
          });
        }
        openFence = null;
        openInfo = "";
        openFenceLine = 0;
        fenceBuffer = [];
      } else if (openInfo === "math") {
        fenceBuffer.push(line);
      }
      continue;
    }

    if (displayOpen) {
      if (line.trim() === "$$") {
        expressions.push({
          expression: displayBuffer.join("\n").trim(),
          line: displayLine,
          display: true,
          kind: "double-dollar block",
        });
        displayOpen = false;
        displayLine = 0;
        displayBuffer = [];
      } else {
        displayBuffer.push(line);
      }
      continue;
    }

    if (match !== null) {
      openFence = match[2];
      openInfo = match[3].trim().toLowerCase();
      openFenceLine = lineNumber;
      fenceBuffer = [];
      continue;
    }

    if (line.trim() === "$$") {
      displayOpen = true;
      displayLine = lineNumber;
      displayBuffer = [];
      continue;
    }

    if (line.includes("$$")) {
      errors.push(
        `${filePath}:${lineNumber}: display-math delimiters must be standalone $$ lines`,
      );
    }

    for (const delimiter of legacyDelimiters) {
      if (line.includes(delimiter)) {
        errors.push(
          `${filePath}:${lineNumber}: legacy delimiter ${JSON.stringify(delimiter)} is forbidden`,
        );
      }
    }

    const lineWithoutCode = line.replace(inlineCodePattern, "");
    inlineMathPattern.lastIndex = 0;
    for (const inlineMatch of lineWithoutCode.matchAll(inlineMathPattern)) {
      expressions.push({
        expression: inlineMatch[1].trim(),
        line: lineNumber,
        display: false,
        kind: "inline math",
      });
    }
  }

  if (openFence !== null) {
    errors.push(`${filePath}:${openFenceLine}: unclosed fenced block`);
  }
  if (displayOpen) {
    errors.push(`${filePath}:${displayLine}: unclosed double-dollar block`);
  }

  for (const item of expressions) {
    if (item.expression.length === 0) {
      errors.push(`${filePath}:${item.line}: empty ${item.kind}`);
    }
  }

  return { expressions, errors };
}

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const tex = new TeX({ packages: AllPackages });
const svg = new SVG({ fontCache: "none" });
const document = mathjax.document("", { InputJax: tex, OutputJax: svg });

function validateExpression(expression, display) {
  try {
    const node = document.convert(expression, { display });
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
let displayCount = 0;
let inlineCount = 0;

for (const filePath of markdownFiles) {
  const relativePath = path.relative(root, filePath);
  const { expressions, errors } = extractMath(filePath);
  failures.push(...errors.map((message) => message.replace(filePath, relativePath)));

  for (const item of expressions) {
    if (item.display) displayCount += 1;
    else inlineCount += 1;
    const error = validateExpression(item.expression, item.display);
    if (error !== null) {
      failures.push(
        `${relativePath}:${item.line}: ${item.kind}: ${error}\n${item.expression}`,
      );
    }
  }
}

if (failures.length > 0) {
  console.error("GitHub Markdown math validation failed:");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log(
  `Validated ${displayCount} display formula(s) and ${inlineCount} inline formula(s) across ${markdownFiles.length} Markdown file(s).`,
);
