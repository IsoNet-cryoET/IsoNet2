import { handleProcess, spawnWithRuntime } from "../process.js";

const starJsonCache = new Map();

function cloneJson(data) {
    return typeof structuredClone === "function"
        ? structuredClone(data)
        : JSON.parse(JSON.stringify(data));
}

function parseStarLine(line) {
    const regex = /'[^']*'|"[^"]*"|\S+/g;
    const matches = line.match(regex);
    if (!matches) return [];
    return matches.map((token) => {
        if (
            (token.startsWith("'") && token.endsWith("'")) ||
            (token.startsWith('"') && token.endsWith('"'))
        ) {
            return token.slice(1, -1);
        }
        return token;
    });
}

function normalizeStarColumn(token) {
    const noPrefix = token.startsWith("_") ? token.slice(1) : token;
    return noPrefix.split("#")[0].trim();
}

function coerceStarValue(value) {
    if (value == null) return null;
    const trimmed = String(value).trim();
    if (trimmed === "") return "";

    if (/^[+-]?\d+$/.test(trimmed)) {
        const parsedInt = Number.parseInt(trimmed, 10);
        if (Number.isSafeInteger(parsedInt)) return parsedInt;
    }

    if (/^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(trimmed)) {
        const parsedFloat = Number(trimmed);
        if (!Number.isNaN(parsedFloat)) return parsedFloat;
    }

    return trimmed;
}

function rowsToColumnJson(rows, columns) {
    const columnJson = {};

    columns.forEach((column) => {
        columnJson[column] = {};
    });

    rows.forEach((row, rowIndex) => {
        columns.forEach((column) => {
            columnJson[column][rowIndex] = row[column] ?? null;
        });
    });

    if (!columnJson.rlnIndex) {
        columnJson.rlnIndex = {};
        rows.forEach((_row, rowIndex) => {
            columnJson.rlnIndex[rowIndex] = rowIndex + 1;
        });
    }

    return [columnJson];
}

function parseStarTextFast(text) {
    const columns = [];
    const rows = [];
    let inLoop = false;
    let sawRows = false;

    for (const rawLine of text.split(/\r?\n/)) {
        const line = rawLine.trim();
        if (!line || line.startsWith("#")) continue;
        if (line.startsWith(";")) {
            throw new Error("Unsupported STAR multiline field in fast parser");
        }

        const lower = line.toLowerCase();
        if (lower.startsWith("data_")) {
            if (sawRows) break;
            continue;
        }
        if (lower === "loop_") {
            if (sawRows) break;
            inLoop = true;
            continue;
        }

        if (!inLoop) continue;

        if (line.startsWith("_")) {
            if (sawRows) break;
            const parts = parseStarLine(line);
            if (parts.length === 0) continue;
            columns.push(normalizeStarColumn(parts[0]));
            continue;
        }

        if (columns.length === 0) continue;

        const values = parseStarLine(line);
        if (values.length === 0) continue;
        if (values.length !== columns.length) {
            throw new Error(
                `Expected ${columns.length} values but found ${values.length}`,
            );
        }

        const row = {};
        for (let index = 0; index < columns.length; index += 1) {
            row[columns[index]] = coerceStarValue(values[index]);
        }
        rows.push(row);
        sawRows = true;
    }

    if (columns.length === 0 || rows.length === 0) {
        throw new Error("No STAR rows found in fast parser");
    }

    return rowsToColumnJson(rows, columns);
}

async function parseStarWithPython(filePath) {
    const fs = await import("fs");
    const os = await import("os");
    const path = await import("path");

    const tempDir = await fs.promises.mkdtemp(
        path.join(os.tmpdir(), "isoapp-star-"),
    );
    const jsonFile = path.join(tempDir, "star.json");
    const command = `isonet.py star2json --star_file "${filePath}" --json_file "${jsonFile}"`;
    const pythonProcess = spawnWithRuntime(command);

    return await new Promise((resolve, reject) => {
        let stderr = "";

        pythonProcess.stderr.on("data", (data) => {
            stderr += data.toString();
        });

        pythonProcess.on("error", async (error) => {
            await fs.promises.rm(tempDir, { recursive: true, force: true });
            reject(error);
        });

        pythonProcess.on("close", async (code) => {
            try {
                if (code !== 0) {
                    throw new Error(
                        stderr || `star2json exited with code ${code}`,
                    );
                }

                const jsonText = await fs.promises.readFile(jsonFile, "utf8");
                const output = jsonText
                    .split(/\r?\n/)
                    .filter(Boolean)
                    .map((line) => JSON.parse(line));

                if (output.length === 0) {
                    throw new Error("No JSON output produced by star2json");
                }

                resolve(output);
            } catch (error) {
                reject(error);
            } finally {
                await fs.promises.rm(tempDir, { recursive: true, force: true });
            }
        });
    });
}

async function loadStarJson(filePath) {
    const fs = await import("fs");

    if (!fs.existsSync(filePath)) {
        throw new Error(`File not found: ${filePath}`);
    }

    const stat = await fs.promises.stat(filePath);
    const cacheKey = `${filePath}:${stat.size}:${stat.mtimeMs}`;
    const cached = starJsonCache.get(cacheKey);
    if (cached) {
        return cloneJson(cached);
    }

    let output;
    try {
        const text = await fs.promises.readFile(filePath, "utf8");
        output = parseStarTextFast(text);
    } catch (fastError) {
        console.warn(`Fast STAR parse failed for ${filePath}:`, fastError);
        output = await parseStarWithPython(filePath);
    }

    for (const key of Array.from(starJsonCache.keys())) {
        if (key.startsWith(`${filePath}:`)) {
            starJsonCache.delete(key);
        }
    }
    starJsonCache.set(cacheKey, output);

    return cloneJson(output);
}

export default function other({ getMainWindow }) {
    return {
        async run(event, data) {
            handleProcess(event, data);
        },
        async parseStarFile(_, filePath) {
            const output = await loadStarJson(filePath);
            return { ok: true, output };
        },
        async updateStar(_, data) {
            const filePath = ".to_star.json";
            let updateStarProcess = null;
            const fs = await import("fs");

            fs.writeFileSync(
                filePath,
                JSON.stringify(data.convertedJson, null, 2),
            );
            updateStarProcess = spawnWithRuntime(
                `isonet.py json2star --json_file "${filePath}" --star_name "${data.star_name}"`,
            );
            updateStarProcess.on("close", (code) => {
                console.log(`Python process exited with code ${code}`);
                updateStarProcess = null;
            });
        },
        async appClose(_, ok) {
            const win = getMainWindow();
            if (!win) return;
            if (ok) {
                win.__forceClose = true;
                win.close();
            }
        },
    };
}
