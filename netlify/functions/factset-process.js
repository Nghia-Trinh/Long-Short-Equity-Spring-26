"use strict";

const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

const REPO_ROOT = path.resolve(__dirname, "..", "..");
const PYTHON_ENTRY = path.join(REPO_ROOT, "equity_pipeline", "pipeline.py");

function json(statusCode, payload) {
  return {
    statusCode,
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  };
}

function parseBoolean(value, defaultValue = false) {
  if (typeof value === "boolean") return value;
  if (typeof value !== "string") return defaultValue;
  const normalized = value.trim().toLowerCase();
  if (["1", "true", "yes", "on"].includes(normalized)) return true;
  if (["0", "false", "no", "off"].includes(normalized)) return false;
  return defaultValue;
}

function runCommand(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: options.cwd || REPO_ROOT,
      env: options.env || process.env,
      shell: false,
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString("utf-8");
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString("utf-8");
    });
    child.on("error", reject);
    child.on("close", (code) => {
      resolve({ code, stdout, stderr });
    });
  });
}

function parseBody(event) {
  if (!event.body) return {};
  try {
    return JSON.parse(event.body);
  } catch (error) {
    throw new Error("Request body must be valid JSON.");
  }
}

function buildTickersFile(tickers) {
  if (!Array.isArray(tickers) || tickers.length === 0) {
    throw new Error("`tickers` must be a non-empty array.");
  }
  const normalized = tickers
    .map((item) => String(item || "").trim().toUpperCase())
    .filter((item) => item.length > 0);
  if (normalized.length === 0) {
    throw new Error("`tickers` must include at least one non-empty symbol.");
  }
  const unique = [...new Set(normalized)];
  const filename = `tickers_${Date.now()}_${Math.random().toString(16).slice(2)}.txt`;
  const targetPath = path.join("/tmp", filename);
  fs.writeFileSync(targetPath, `${unique.join("\n")}\n`, "utf-8");
  return targetPath;
}

function buildEnv(overrides = {}) {
  const env = { ...process.env };
  for (const [key, value] of Object.entries(overrides || {})) {
    if (value === undefined || value === null) continue;
    env[String(key)] = String(value);
  }
  return env;
}

exports.handler = async (event) => {
  if (event.httpMethod !== "POST") {
    return json(405, { error: "Method not allowed. Use POST." });
  }

  try {
    if (!fs.existsSync(PYTHON_ENTRY)) {
      return json(500, { error: `Pipeline entry not found: ${PYTHON_ENTRY}` });
    }

    const body = parseBody(event);
    const tickersFile = buildTickersFile(body.tickers);

    const outputDir = String(body.outputDir || `/tmp/factset_pipeline_${Date.now()}`);
    const months = Number.isFinite(Number(body.months)) ? Number(body.months) : 6;
    const skipNews = parseBoolean(body.skipNews, true);
    const factsetEnrich = parseBoolean(body.factsetEnrich, true);
    const factsetIncludeRaw = parseBoolean(body.factsetIncludeRaw, false);
    const factsetExchange = body.factsetExchange ? String(body.factsetExchange) : null;

    const args = [
      PYTHON_ENTRY,
      "--tickers",
      tickersFile,
      "--output",
      outputDir,
      "--months",
      String(months),
    ];
    if (skipNews) args.push("--skip-news");
    if (factsetEnrich) args.push("--factset-enrich");
    if (factsetIncludeRaw) args.push("--factset-include-raw");
    if (factsetExchange) {
      args.push("--factset-exchange", factsetExchange);
    }

    const env = buildEnv(body.env || {});
    const result = await runCommand("python3", args, { env });
    const metadataPath = path.join(outputDir, "run_metadata.json");

    let metadata = null;
    if (fs.existsSync(metadataPath)) {
      try {
        metadata = JSON.parse(fs.readFileSync(metadataPath, "utf-8"));
      } catch (error) {
        metadata = { parse_error: String(error) };
      }
    }

    if (result.code !== 0) {
      return json(500, {
        ok: false,
        exitCode: result.code,
        outputDir,
        metadataPath,
        metadata,
        stdoutTail: result.stdout.slice(-8000),
        stderrTail: result.stderr.slice(-8000),
      });
    }

    return json(200, {
      ok: true,
      outputDir,
      metadataPath,
      metadata,
      tickersFile,
      request: {
        tickers: body.tickers,
        months,
        skipNews,
        factsetEnrich,
        factsetIncludeRaw,
        factsetExchange,
      },
      stdoutTail: result.stdout.slice(-8000),
      stderrTail: result.stderr.slice(-8000),
    });
  } catch (error) {
    return json(400, { ok: false, error: String(error.message || error) });
  }
};
