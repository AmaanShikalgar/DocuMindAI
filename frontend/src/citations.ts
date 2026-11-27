/** gpt-oss sometimes cites in its own format, like 【1†L1-L4】, or groups numbers as [1, 3].
 * Normalise every variant to [1][3] so citations stay clickable. */
export function normalizeCitations(text: string): string {
  return text
    .replace(/【\s*(\d+)(?:†[^】]*)?\s*】/g, "[$1]")
    .replace(/\[(\d+)†[^\]]*\]/g, "[$1]")
    .replace(/\[(\d+(?:\s*,\s*\d+)+)\]/g, (_, list: string) => list.split(",").map((n) => `[${n.trim()}]`).join(""));
}

/** 1-based source numbers an answer cites, in order of first use, limited to 1..max. */
export function citedNumbers(text: string, max: number): number[] {
  const seen: number[] = [];
  for (const m of normalizeCitations(text).matchAll(/\[(\d+)\]/g)) {
    const n = +m[1];
    if (n >= 1 && n <= max && !seen.includes(n)) seen.push(n);
  }
  return seen;
}
